# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _score_exception_default() -> dict:
    return {"score": 0.0, "reward_score_exception": 1.0}


def _score_budget_default() -> dict:
    return {"score": 0.0, "reward_batch_timeout": 1.0}


def _math_score_process_exception_default() -> dict:
    return {
        "score": 0.0,
        "math_score_process_exception": 1.0,
        "math_score_timeout": 0.0,
        "math_score_exception": 0.0,
        "math_score_fast_reject": 0.0,
        "math_score_skipped_by_format": 0.0,
        "math_score_slow": 0.0,
        "math_score_elapsed_s": 0.0,
    }


def _math_score_batch_timeout_default() -> dict:
    out = _math_score_process_exception_default()
    out["math_score_process_exception"] = 0.0
    out["math_score_batch_timeout"] = 1.0
    return out


def _score_process_context():
    method = os.environ.get("HYBRID_V2_MATH_SCORE_START_METHOD", "spawn").strip().lower()
    if method not in {"spawn", "forkserver"}:
        method = "spawn"
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context("spawn")


def _terminate_process_pool(executor) -> None:
    processes = list((getattr(executor, "_processes", None) or {}).values())
    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)
    except Exception:
        pass
    for proc in processes:
        try:
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def _compute_score_one(self, item: dict):
        return self.compute_score(
            data_source=item["data_source"],
            solution_str=item["response_str"],
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )

    def _attach_precomputed_math_scores(self, items: list[dict]) -> None:
        if not items or not _env_bool("HYBRID_V2_PRECOMPUTE_MATH_SCORE", "0"):
            return

        os.environ["RGRPO_LIGHT_IMPORT"] = "1"
        from rgrpo.math_score_light import compute_math_score_payload

        workers = max(1, _env_int("HYBRID_V2_MATH_SCORE_PROCESS_WORKERS", 16))
        workers = min(workers, len(items))
        per_item_timeout = _env_float("HYBRID_V2_MATH_SCORE_TIMEOUT", 5.0)
        slow_log_sec = _env_float("HYBRID_V2_SCORE_SLOW_LOG_SEC", 2.0)
        batch_budget = _env_float("HYBRID_V2_REWARD_BATCH_TIMEOUT", 0.0)
        timeout_action = os.environ.get("HYBRID_V2_REWARD_BATCH_TIMEOUT_ACTION", "raise").strip().lower()
        process_exception_action = os.environ.get(
            "HYBRID_V2_REWARD_PROCESS_EXCEPTION_ACTION", "raise"
        ).strip().lower()

        payloads = []
        for item in items:
            extra_info = item.get("extra_info")
            item["extra_info"] = dict(extra_info) if isinstance(extra_info, dict) else {}
            item["math_score_info"] = {}
            payloads.append(
                {
                    "solution_str": item["response_str"],
                    "ground_truth": item["ground_truth"],
                    "timeout": per_item_timeout,
                    "slow_log_sec": slow_log_sec,
                }
            )

        def store_math_result(item: dict, result: dict) -> None:
            score = float(result.get("score", 0.0))
            info = dict(result)
            info.pop("score", None)
            info["math_score_raw"] = score
            item["math_score_info"] = info
            item["extra_info"]["precomputed_task_reward"] = score

        start = time.monotonic()
        executor = cf.ProcessPoolExecutor(max_workers=workers, mp_context=_score_process_context())
        future_to_idx = {}
        process_exceptions = 0
        timed_out = 0
        try:
            for idx, payload in enumerate(payloads):
                future_to_idx[executor.submit(compute_math_score_payload, payload)] = idx

            wait_timeout = batch_budget if batch_budget > 0 else None
            done, not_done = cf.wait(
                future_to_idx.keys(),
                timeout=wait_timeout,
                return_when=cf.ALL_COMPLETED,
            )

            for future in done:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception:
                    result = _math_score_process_exception_default()
                    process_exceptions += 1
                store_math_result(items[idx], result)

            if not_done:
                timed_out = len(not_done)
                for future in not_done:
                    idx = future_to_idx[future]
                    future.cancel()
                    result = _math_score_batch_timeout_default()
                    store_math_result(items[idx], result)
                _terminate_process_pool(executor)
            else:
                executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            _terminate_process_pool(executor)
            raise

        elapsed = time.monotonic() - start
        if timed_out:
            msg = (
                "[reward_manager] math score batch timeout: "
                f"items={len(items)} workers={workers} timed_out={timed_out} elapsed_s={elapsed:.2f}"
            )
            if timeout_action == "raise":
                raise RuntimeError(msg)
            print(msg, file=sys.stderr, flush=True)

        if process_exceptions:
            msg = (
                "[reward_manager] math score process exceptions: "
                f"items={len(items)} workers={workers} exceptions={process_exceptions} elapsed_s={elapsed:.2f}"
            )
            if process_exception_action == "raise":
                raise RuntimeError(msg)
            print(msg, file=sys.stderr, flush=True)

        timeouts = sum(float(item["math_score_info"].get("math_score_timeout", 0.0)) for item in items)
        fast_rejects = sum(float(item["math_score_info"].get("math_score_fast_reject", 0.0)) for item in items)
        skipped_format = sum(
            float(item["math_score_info"].get("math_score_skipped_by_format", 0.0)) for item in items
        )
        slow = sum(float(item["math_score_info"].get("math_score_slow", 0.0)) for item in items)
        if timeouts or fast_rejects or skipped_format or slow or _env_bool("HYBRID_V2_REWARD_SCORE_LOG_EVERY_BATCH", "0"):
            print(
                (
                    "[reward_manager] math score batch: "
                    f"items={len(items)} workers={workers} elapsed_s={elapsed:.2f} "
                    f"timeouts={int(timeouts)} fast_rejects={int(fast_rejects)} "
                    f"skipped_format={int(skipped_format)} slow={int(slow)}"
                ),
                file=sys.stderr,
                flush=True,
            )

    def _score_items(self, items: list[dict]) -> list:
        batch_budget = _env_float(
            "HYBRID_V2_REWARD_BATCH_TIMEOUT",
            _env_float("VERL_REWARD_SCORE_BATCH_TIMEOUT", 0.0),
        )
        deadline = time.monotonic() + batch_budget if batch_budget > 0 else None
        timeout_action = os.environ.get("HYBRID_V2_REWARD_BATCH_TIMEOUT_ACTION", "raise").strip().lower()

        results = []
        skipped = 0
        for item in items:
            if deadline is not None and time.monotonic() >= deadline:
                if timeout_action == "raise":
                    raise RuntimeError(
                        f"[reward_manager] reward batch wall-clock budget exhausted; skipped_from={len(results)}"
                    )
                results.append(_score_budget_default())
                skipped += 1
                continue
            try:
                results.append(self._compute_score_one(item))
            except Exception as exc:
                if os.environ.get("FOURWAY_CONFIG"):
                    # A judge infrastructure failure is not an incorrect answer.
                    # Retain the exact failed input before propagating the error.
                    from pathlib import Path
                    import traceback
                    from fourway.contracts import atomic_json, config
                    method = os.environ.get("FOURWAY_METHOD", "unknown")
                    destination = (Path(config()["output"]) / "diagnostics" / "reward_errors"
                                   / f"{method}_{os.getpid()}_{time.time_ns()}.json")
                    atomic_json(destination, {
                        "method": method, "pid": os.getpid(),
                        "data_source": str(item.get("data_source", "")),
                        "ground_truth": str(item.get("ground_truth", "")),
                        "response_text": item.get("response_str", ""),
                        "extra_info": str(item.get("extra_info", "")),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    })
                    raise RuntimeError(f"Reward judge failed; refusing zero reward. Diagnostic: {destination}") from exc
                results.append(_score_exception_default())
        if skipped:
            print(
                f"[reward_manager] reward batch wall-clock budget exhausted; skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )
        return results

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        decoded_items = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            decoded_items.append(
                {
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                    "valid_response_length": valid_response_length,
                }
            )

        self._attach_precomputed_math_scores(decoded_items)
        scores = self._score_items(decoded_items)

        def append_reward_extra(score_dict, row_idx: int) -> None:
            if score_dict:
                for key in score_dict.keys():
                    if key not in reward_extra_info:
                        reward_extra_info[key] = [0.0] * row_idx
            for key in list(reward_extra_info.keys()):
                value = score_dict.get(key, 0.0) if score_dict else 0.0
                reward_extra_info[key].append(value)

        for i, (item, score) in enumerate(zip(decoded_items, scores)):
            valid_response_length = item["valid_response_length"]
            data_source = item["data_source"]
            prompt_str = item["prompt_str"]
            response_str = item["response_str"]
            ground_truth = item["ground_truth"]

            if isinstance(score, dict):
                score = dict(score)
                score.update(item.get("math_score_info", {}))
                reward = score["score"]
                # Store the information including original reward
                append_reward_extra(score, i)
            else:
                reward = score
                append_reward_extra(item.get("math_score_info", None), i)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
