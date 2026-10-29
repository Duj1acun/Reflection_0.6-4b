"""Pre-trigger baseline GRPO for reflection training.

This mode is enabled only when ``PRETRIGGER_BASELINE_GRPO_ENABLE=1`` and this
module is imported as the custom reward function.  It installs two runtime
patches in the current training process:

* rollout: when an original trajectory emits ``<reflection>...</reflection>``,
  sample K free baseline continuations from the token prefix immediately before
  the reflection tag;
* advantage: for each reflection event, normalize the original reflected
  trajectory against those K pre-trigger baselines.  Direct no-reflection
  samples can be kept as a separate same-prompt GRPO objective with environment
  flags.

No existing training entrypoint is changed by importing this module unless the
environment gate above is set.
"""

from __future__ import annotations

import os
import json
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from .core_hybrid_grpo_v2 import (
    REFL_PATTERN,
    _REFLECTION_CLOSE_RE,
    _REFLECTION_OPEN_RE,
    _reflection_format_valid,
    compute_math_score,
)


_PATCHED = False
_ORIGINAL_REFLECTION_GRPO_LOOP = None
_ORIGINAL_COMPUTE_ADVANTAGE = None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _efficiency_reward(n_iters: int, phi: int) -> float:
    phi = max(1, int(phi))
    n_iters = max(1, int(n_iters))
    used_extra_iters = max(0, n_iters - 1)
    reward = 1.0 - (used_extra_iters / phi) ** 2
    return max(0.0, min(1.0, float(reward)))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _grpo_norm(
    rewards: Sequence[float],
    binary_values: Optional[Sequence[float]] = None,
    norm_by_std: bool = True,
    std_floor: float = 1e-6,
    epsilon: float = 1e-6,
) -> List[float]:
    if not rewards:
        return []
    vals = np.asarray(rewards, dtype=np.float32)
    if vals.size < 2:
        return [0.0 for _ in rewards]
    centered = vals - float(np.mean(vals))
    if not norm_by_std:
        return [float(x) for x in centered]
    if binary_values is not None:
        binary_arr = np.asarray(binary_values, dtype=np.float32)
        if binary_arr.shape[0] == vals.shape[0] and np.isfinite(binary_arr).all():
            binary_arr = (binary_arr > 0.5).astype(np.float32)
            if np.all(binary_arr == binary_arr[0]):
                return [float(x) for x in centered]
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    if std <= float(std_floor):
        return [0.0 for _ in rewards]
    return [float(x) for x in (centered / (std + epsilon))]


def _normalize_spans(spans: Any, max_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in _as_list(spans):
        try:
            start, end = item
        except Exception:
            continue
        start = min(max(0, int(start)), max_len)
        end = min(max(start, int(end)), max_len)
        if end > start:
            out.append((start, end))
    return out


def _float_array(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return np.full(length, default, dtype=np.float32)
    if arr.shape[0] != length:
        return np.full(length, default, dtype=np.float32)
    return arr


def _object_array(value: Any, length: int) -> np.ndarray:
    if value is None:
        arr = np.empty(length, dtype=object)
        arr[:] = [None] * length
        return arr
    try:
        arr = np.asarray(value, dtype=object)
    except Exception:
        out = np.empty(length, dtype=object)
        out[:] = [None] * length
        return out
    if arr.ndim > 1 and arr.shape[0] == length:
        out = np.empty(length, dtype=object)
        out[:] = [arr[i].tolist() for i in range(length)]
        return out
    if arr.shape[0] != length:
        out = np.empty(length, dtype=object)
        out[:] = [None] * length
        return out
    return arr


def _binary_score(value: Any) -> float:
    try:
        return 1.0 if float(value) > 0.5 else 0.0
    except Exception:
        return 0.0


def _reflection_markup_stats(generated: str) -> Tuple[int, bool]:
    text = generated or ""
    complete_count = len(REFL_PATTERN.findall(text))
    has_markup = bool(
        complete_count
        or _REFLECTION_OPEN_RE.search(text)
        or _REFLECTION_CLOSE_RE.search(text)
    )
    return complete_count, has_markup


def _pretrigger_require_reflection_format() -> bool:
    raw = os.environ.get(
        "PRETRIGGER_REQUIRE_REFLECTION_FORMAT",
        os.environ.get("CF_REFL_REQUIRE_REFLECTION_FORMAT", "1"),
    )
    return _as_bool(raw)


def _format_valid(generated: str) -> bool:
    _complete_count, has_markup = _reflection_markup_stats(generated)
    if not _pretrigger_require_reflection_format() or not has_markup:
        return True
    return bool(_reflection_format_valid(generated))


def compute_pretrigger_baseline_grpo_score(
    solution_str: str = None,
    ground_truth: Any = None,
    generated: str = None,
    data_source: str = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Return final-answer correctness gated by reflection format validity."""

    del data_source, kwargs
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""
    if ground_truth is None:
        ground_truth = ""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", ground_truth.get("ground_truth", ""))

    num_reflections, has_reflection_markup = _reflection_markup_stats(generated)
    info_num_reflections = _safe_int(
        extra_info.get("num_reflections", extra_info.get("n_reflections", num_reflections)),
        num_reflections,
    )
    num_reflections = max(num_reflections, info_num_reflections)

    env_phi = os.environ.get("PRETRIGGER_REWARD_PHI")
    phi_source = env_phi if env_phi is not None else extra_info.get("phi", 5)
    phi = max(1, _safe_int(phi_source, 5))

    default_n_iters = 1 + num_reflections
    n_iters = _safe_int(extra_info.get("n_iters", default_n_iters), default_n_iters)
    n_iters = max(default_n_iters, n_iters)
    if _as_bool(extra_info.get("terminated_by_cap", False)):
        n_iters = max(n_iters, phi)

    reflection_format_valid = _format_valid(generated)
    format_reward = 1.0 if reflection_format_valid else 0.0
    skip_math_on_invalid_format = _as_bool(
        os.environ.get("PRETRIGGER_SKIP_MATH_ON_INVALID_FORMAT", "1")
    )
    math_score_skipped_by_format = bool(
        skip_math_on_invalid_format and has_reflection_markup and format_reward <= 0.5
    )
    has_precomputed_task_reward = "precomputed_task_reward" in extra_info
    if math_score_skipped_by_format:
        task_reward = 0.0
    elif has_precomputed_task_reward:
        task_reward = _binary_score(extra_info.get("precomputed_task_reward", 0.0))
    else:
        task_reward = float(compute_math_score(solution_str=generated, ground_truth=ground_truth))
    is_correct = task_reward > 0.5

    use_eff_reward = _as_bool(os.environ.get("PRETRIGGER_USE_EFF_REWARD", "1"))
    eff_reward = _efficiency_reward(n_iters=n_iters, phi=phi) if use_eff_reward else 1.0
    reflection_freq_penalty_coef = max(
        0.0,
        _safe_float(os.environ.get("PRETRIGGER_REFLECTION_FREQ_PENALTY", "0.0"), 0.0),
    )
    reflection_freq_penalty = (
        reflection_freq_penalty_coef * float(num_reflections)
        if is_correct and num_reflections > 0
        else 0.0
    )

    score_before_freq_penalty = float(task_reward * eff_reward * format_reward)
    final_score = max(0.0, score_before_freq_penalty - reflection_freq_penalty)

    return {
        "score": float(final_score),
        "score_before_format_gate": float(task_reward),
        "score_before_freq_penalty": float(score_before_freq_penalty),
        "task_reward": task_reward,
        "eff_reward": float(eff_reward),
        "format_reward": float(format_reward),
        "math_score_precomputed": float(has_precomputed_task_reward),
        "math_score_skipped_by_format": float(math_score_skipped_by_format),
        "reflection_freq_penalty": float(reflection_freq_penalty),
        "reflection_format_valid": float(reflection_format_valid),
        "has_reflection_markup": float(has_reflection_markup),
        "format_gate_zero": float(has_reflection_markup and format_reward <= 0.5),
        "num_reflections": int(num_reflections),
        "n_reflections": int(num_reflections),
        "n_iters": int(n_iters),
        "phi": int(phi),
        "has_reflection": float(num_reflections > 0),
        "is_correct": float(is_correct),
    }


def pretrigger_baseline_reflection_grpo_loop(self, gen_batch, actor_rollout_wg, is_train: bool = True):
    """Reflection rollout with pre-trigger baseline continuations."""

    if not _as_bool(os.environ.get("PRETRIGGER_BASELINE_GRPO_ENABLE", "0")):
        return _ORIGINAL_REFLECTION_GRPO_LOOP(self, gen_batch, actor_rollout_wg, is_train)
    if not is_train:
        return _ORIGINAL_REFLECTION_GRPO_LOOP(self, gen_batch, actor_rollout_wg, is_train)

    result, _ = self.no_env_loop(gen_batch, actor_rollout_wg, is_train)
    if "rollout_log_probs" in result.batch.keys():
        result.batch.pop("rollout_log_probs")

    max_iters = int(self.config.algorithm.get("reflection_grpo", {}).get("max_iters", 4))
    max_iters = max(1, max_iters)
    remaining_reflections = max(0, max_iters - 1)
    k_baseline = max(
        1,
        _safe_int(
            os.environ.get(
                "PRETRIGGER_BASELINE_K",
                self.config.actor_rollout_ref.rollout.get("pretrigger_baseline_k", 8),
            ),
            8,
        ),
    )
    drop_direct_original = _as_bool(os.environ.get("PRETRIGGER_DROP_DIRECT_ORIGINAL", "1"))
    # When enabled, pre-trigger baseline continuations get NO recursive
    # reflection rounds (remaining_reflections forced to 0).  This turns each
    # baseline into a single-pass "direct attempt from the pre-trigger prefix"
    # instead of a trajectory that recursively reflects like the original, which
    # restores the intended "reflect vs. don't-reflect" contrast.  Default off
    # to preserve existing behaviour.
    baseline_no_reflection = _as_bool(
        os.environ.get("PRETRIGGER_BASELINE_NO_REFLECTION", "0")
    )
    max_branch_prompts_per_call = max(
        1, _safe_int(os.environ.get("PRETRIGGER_MAX_BRANCH_PROMPTS_PER_CALL", "64"), 64)
    )
    max_events_per_step = max(
        0, _safe_int(os.environ.get("PRETRIGGER_MAX_EVENTS_PER_STEP", "0"), 0)
    )
    max_events_per_source = max(
        0, _safe_int(os.environ.get("PRETRIGGER_MAX_EVENTS_PER_SOURCE", "0"), 0)
    )
    event_sample_mode = os.environ.get("PRETRIGGER_EVENT_SAMPLE_MODE", "random").strip().lower()
    if event_sample_mode not in {"random", "first"}:
        event_sample_mode = "random"
    reflection_cont_prompt_mode = os.environ.get(
        "PRETRIGGER_REFLECTION_CONT_PROMPT_MODE",
        "reflection_only",
    ).strip().lower()
    if reflection_cont_prompt_mode in {"reset", "reflection", "reflection_only"}:
        reflection_cont_prompt_mode = "reflection_only"
    elif reflection_cont_prompt_mode in {"history", "full_history"}:
        reflection_cont_prompt_mode = "full_history"
    else:
        reflection_cont_prompt_mode = "reflection_only"

    max_cont_prompt = int(
        os.environ.get(
            "PRETRIGGER_BASELINE_MAX_PROMPT_TOKENS",
            str(self.config.data.max_prompt_length + self.config.data.max_response_length // 2),
        )
    )
    response_length = int(self.config.data.max_response_length)
    rollout_max_model_len = _safe_int(
        self.config.actor_rollout_ref.rollout.get("max_model_len", 0), 0
    )
    default_max_total_response_length = max(1, response_length * max_iters)
    context_response_budget = 0
    if rollout_max_model_len > 0:
        context_response_budget = max(0, rollout_max_model_len - max_cont_prompt)
        if context_response_budget > 0:
            default_max_total_response_length = min(
                default_max_total_response_length,
                context_response_budget,
            )
    max_total_response_length = max(
        1,
        _safe_int(
            os.environ.get(
                "PRETRIGGER_MAX_TOTAL_RESPONSE_LENGTH",
                str(default_max_total_response_length),
            ),
            default_max_total_response_length,
        ),
    )
    if context_response_budget > 0:
        max_total_response_length = min(max_total_response_length, context_response_budget)

    def _infer_vocab_size() -> Optional[int]:
        """Exclusive upper bound for valid prompt token ids.

        A reconstructed prompt id is rejected by vLLM's `_validate_token_prompt`
        when it exceeds `tokenizer.max_token_id` (= max(max(get_vocab()),
        tokenizer.vocab_size)). The model's padded `config.json` vocab_size
        (e.g. 151936 for Qwen2.5) is *larger* than that bound, so using it
        alone lets the dead padding-region ids (151665..151935) through and
        the run crashes exactly as before. The valid limit is therefore the
        **minimum** of the model embedding size and the tokenizer's id range.
        `tokenizer.vocab_size` must NOT be used as the limit on its own: it
        excludes added special tokens (e.g. <|im_start|>) that are legitimately
        present in chat prompts.
        """
        limits: List[int] = []

        # 1) Model embedding rows: an id >= this indexes out of bounds.
        try:
            model_path = str(self.config.actor_rollout_ref.model.path)
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
                model_vocab_size = int(model_config.get("vocab_size", 0))
                if model_vocab_size > 0:
                    limits.append(model_vocab_size)
        except Exception:
            pass

        # 2) Tokenizer id range == what vLLM actually enforces. Compute the
        #    max known token id and use (max_id + 1) as the exclusive bound.
        tok_max_id: Optional[int] = None
        try:
            vocab = self.tokenizer.get_vocab()
            if vocab:
                tok_max_id = max(int(v) for v in vocab.values())
        except Exception:
            tok_max_id = None
        for getter in (
            lambda: getattr(self.tokenizer, "vocab_size", None),  # -> id (n-1)
            lambda: len(self.tokenizer),                           # -> id (n-1)
        ):
            try:
                value = getter()
            except Exception:
                value = None
            if value is None:
                continue
            try:
                value = int(value)
            except Exception:
                continue
            if value > 0:
                tok_max_id = (value - 1) if tok_max_id is None else max(tok_max_id, value - 1)
        if tok_max_id is not None and tok_max_id >= 0:
            limits.append(tok_max_id + 1)

        return min(limits) if limits else None

    vocab_size = _infer_vocab_size()
    invalid_token_stats = {"count": 0, "printed": 0}

    def _is_valid_token_id(token_id: Any) -> bool:
        try:
            token_id = int(token_id)
        except Exception:
            return False
        if token_id < 0:
            return False
        if vocab_size is not None and token_id >= vocab_size:
            return False
        return True

    def _fallback_prompt_token_id() -> int:
        for token_id in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 0):
            if isinstance(token_id, int) and _is_valid_token_id(token_id):
                return int(token_id)
            if isinstance(token_id, (list, tuple)):
                for item in token_id:
                    if _is_valid_token_id(item):
                        return int(item)
        return 0

    def _pad_token_id() -> int:
        pad_id = self.tokenizer.pad_token_id
        if _is_valid_token_id(pad_id):
            return int(pad_id)
        return _fallback_prompt_token_id()

    def _sanitize_token_ids(ids, context: str) -> List[int]:
        clean = []
        dropped = 0
        for token_id in _to_py_list(ids):
            if _is_valid_token_id(token_id):
                clean.append(int(token_id))
            else:
                dropped += 1
        if dropped:
            invalid_token_stats["count"] += dropped
            if invalid_token_stats["printed"] < 5:
                invalid_token_stats["printed"] += 1
                print(
                    (
                        f"[PRETRIGGER] dropped invalid token ids: context={context} "
                        f"dropped={dropped} vocab_size={vocab_size}"
                    ),
                    flush=True,
                )
        return clean

    def _get_base_stop_token_ids():
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            return [eos_token_id] if _is_valid_token_id(eos_token_id) else []
        if isinstance(eos_token_id, (list, tuple)):
            return [int(tid) for tid in eos_token_id if _is_valid_token_id(tid)]
        return []

    def _truncate_prompt(prompt_ids):
        prompt_ids = _sanitize_token_ids(prompt_ids, "prompt")
        if len(prompt_ids) > max_cont_prompt:
            return list(prompt_ids[-max_cont_prompt:])
        return list(prompt_ids)

    def _to_py_list(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def _strip_response_ids(resp_row, resp_mask=None):
        ids = _to_py_list(resp_row)
        if resp_mask is not None:
            mask = _to_py_list(resp_mask)
            valid_len = 0
            for idx, keep in enumerate(mask[: len(ids)]):
                if int(keep) > 0:
                    valid_len = idx + 1
            return _sanitize_token_ids(ids[:valid_len], "response")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return _sanitize_token_ids(ids, "response")
        eos_ids = set(_get_base_stop_token_ids())
        if pad_id in eos_ids:
            for idx, token_id in enumerate(ids):
                if token_id in eos_ids:
                    return _sanitize_token_ids(ids[: idx + 1], "response")
            return _sanitize_token_ids(ids, "response")
        end = len(ids)
        while end > 0 and ids[end - 1] == pad_id:
            end -= 1
        return _sanitize_token_ids(ids[:end], "response")

    def _contains_eos(ids):
        eos_ids = set(_get_base_stop_token_ids())
        return bool(eos_ids and any(token_id in eos_ids for token_id in ids))

    def _stage_truncated(stage_ids):
        if not stage_ids or _contains_eos(stage_ids):
            return False
        return len(stage_ids) >= int(self.config.data.max_response_length)

    def _can_append_full(state, ids_len):
        return len(state["ids"]) + int(ids_len) <= max_total_response_length

    def _append_ids_capped(state, ids):
        ids = _sanitize_token_ids(ids, "append")
        remaining = max_total_response_length - len(state["ids"])
        if remaining <= 0:
            if ids:
                state["truncated"] = True
            return 0
        take = min(len(ids), remaining)
        if take > 0:
            state["ids"].extend(ids[:take])
        if take < len(ids):
            state["truncated"] = True
        return take

    def _sample_event_items(items, cap, rng):
        items = list(items)
        if cap <= 0 or len(items) <= cap:
            return items
        if event_sample_mode == "first":
            return items[:cap]
        selected = np.sort(rng.choice(len(items), size=cap, replace=False))
        return [items[int(idx)] for idx in selected]

    def _reflection_cont_prompt_ids(reset_prompt_ids, full_history_prompt_ids, reflection_ids):
        if reflection_cont_prompt_mode == "full_history":
            return _truncate_prompt(full_history_prompt_ids)
        return _truncate_prompt(list(reset_prompt_ids) + list(reflection_ids))

    def _build_prompt_batch(prompt_id_list, meta_info, n):
        if not prompt_id_list:
            return None
        sanitized_prompt_id_list = []
        for ids in prompt_id_list:
            ids = _truncate_prompt(ids)
            if not ids:
                ids = [_fallback_prompt_token_id()]
            sanitized_prompt_id_list.append(ids)
        max_len = max(len(ids) for ids in sanitized_prompt_id_list)
        samples = []
        for ids in sanitized_prompt_id_list:
            pad_len = max_len - len(ids)
            padded_ids = [_pad_token_id()] * pad_len + list(ids)
            attn = [0] * pad_len + [1] * len(ids)
            ids_t = torch.tensor(padded_ids, dtype=torch.long)
            attn_t = torch.tensor(attn, dtype=torch.long)
            pos_t = compute_position_id_with_mask(attn_t.unsqueeze(0))[0]
            samples.append(
                {
                    "input_ids": ids_t,
                    "attention_mask": attn_t,
                    "position_ids": pos_t,
                    "raw_prompt_ids": np.array(ids, dtype=np.int64),
                }
            )

        prompt_dict = collate_fn(samples)
        batch_meta = dict(meta_info)
        batch_meta["n"] = n
        existing_stop_token_ids = batch_meta.get("stop_token_ids", None)
        has_existing_stop = existing_stop_token_ids is not None and len(_as_list(existing_stop_token_ids)) > 0
        base_stop_token_ids = existing_stop_token_ids if has_existing_stop else _get_base_stop_token_ids()
        if base_stop_token_ids:
            batch_meta["stop_token_ids"] = base_stop_token_ids
        else:
            batch_meta.pop("stop_token_ids", None)
        return DataProto.from_single_dict(prompt_dict, meta_info=batch_meta)

    def _generate_from_prompt_ids(prompt_id_list, n, warn_prefix):
        if not prompt_id_list:
            return []
        prompt_batch = _build_prompt_batch(prompt_id_list, result.meta_info, n)
        prompt_padded, prompt_pad_size = pad_dataproto_to_divisor(
            prompt_batch, actor_rollout_wg.world_size
        )
        try:
            generated_padded = actor_rollout_wg.generate_sequences(prompt_padded)
        except Exception as exc:
            print(
                f"{warn_prefix} generate_sequences failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return []
        generated = unpad_dataproto(generated_padded, pad_size=prompt_pad_size * max(1, n))

        expected_rows = len(prompt_id_list) * max(1, n)
        total_rows = len(generated)
        usable_rows = min(total_rows, expected_rows)
        usable_rows -= usable_rows % max(1, n)
        if total_rows != expected_rows:
            print(
                f"{warn_prefix} row mismatch: got={total_rows}, expected={expected_rows}, "
                f"prompts={len(prompt_id_list)}, n={n}",
                flush=True,
            )
        if usable_rows <= 0:
            return []
        if usable_rows != expected_rows:
            print(
                f"{warn_prefix} truncating rows to {usable_rows} (multiple of n={n})",
                flush=True,
            )

        resp_ids = generated.batch["responses"][:usable_rows]
        resp_masks = None
        if "attention_mask" in generated.batch.keys():
            generated_attention_mask = generated.batch["attention_mask"][:usable_rows]
            if generated_attention_mask.shape[-1] >= resp_ids.shape[-1]:
                resp_masks = generated_attention_mask[:, -resp_ids.shape[-1] :]
        grouped_rows = []
        usable_prompt_count = usable_rows // max(1, n)
        for prompt_idx in range(usable_prompt_count):
            prompt_rows = []
            for sample_idx in range(max(1, n)):
                row_idx = prompt_idx * max(1, n) + sample_idx
                stripped_ids = _strip_response_ids(
                    resp_ids[row_idx],
                    None if resp_masks is None else resp_masks[row_idx],
                )
                prompt_rows.append(
                    {
                        "response_ids": stripped_ids,
                        "response_text": self.tokenizer.decode(
                            stripped_ids, skip_special_tokens=True
                        ),
                    }
                )
            grouped_rows.append(prompt_rows)
        if n == 1:
            return [rows[0] for rows in grouped_rows]
        return grouped_rows

    def _generate_from_prompt_ids_chunked(prompt_id_list, n, warn_prefix):
        if not prompt_id_list:
            return []
        if len(prompt_id_list) <= max_branch_prompts_per_call:
            return _generate_from_prompt_ids(prompt_id_list, n=n, warn_prefix=warn_prefix)
        grouped = []
        for start in range(0, len(prompt_id_list), max_branch_prompts_per_call):
            end = min(start + max_branch_prompts_per_call, len(prompt_id_list))
            grouped.extend(
                _generate_from_prompt_ids(
                    prompt_id_list[start:end],
                    n=n,
                    warn_prefix=f"{warn_prefix} chunk={start // max_branch_prompts_per_call + 1}",
                )
            )
        return grouped

    def _parse_reflection_stage(stage_text, stage_ids):
        match = REFL_PATTERN.search(stage_text)
        if match is None:
            return None
        stage_len = len(stage_ids)

        def _bounds_from_offsets():
            try:
                encoded = self.tokenizer(
                    stage_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
            except Exception:
                return None
            offsets = encoded.get("offset_mapping", None)
            token_ids = encoded.get("input_ids", None)
            if offsets is None or token_ids is None:
                return None
            if len(offsets) > stage_len:
                return None
            if list(token_ids) != list(stage_ids[: len(token_ids)]):
                return None

            start_char = int(match.start())
            end_char = int(match.end())
            pre_end = None
            refl_end = None
            for tok_idx, (_start, end) in enumerate(offsets):
                end = int(end)
                if pre_end is None and end > start_char:
                    pre_end = tok_idx
                if refl_end is None and end >= end_char:
                    refl_end = tok_idx + 1
                    break
            if pre_end is None:
                pre_end = len(offsets)
            if refl_end is None:
                refl_end = len(offsets)
            return pre_end, refl_end

        def _bounds_from_decode_scan():
            start_char = int(match.start())
            end_char = int(match.end())
            pre_end = 0
            refl_end = stage_len
            for tok_idx in range(stage_len + 1):
                text_prefix = self.tokenizer.decode(
                    stage_ids[:tok_idx], skip_special_tokens=True
                )
                text_len = len(text_prefix)
                if text_len <= start_char:
                    pre_end = tok_idx
                if text_len >= end_char:
                    refl_end = tok_idx
                    break
            return pre_end, refl_end

        bounds = _bounds_from_offsets()
        if bounds is None:
            bounds = _bounds_from_decode_scan()
        pre_end_tok, refl_end_tok = bounds
        reflection_tag = match.group(0).rstrip() + "\n"
        reflection_ids = self.tokenizer.encode(reflection_tag, add_special_tokens=False)
        pre_end_tok = min(pre_end_tok, stage_len)
        refl_end_tok = min(max(pre_end_tok, refl_end_tok), stage_len)
        return {
            "reflection_ids": reflection_ids,
            "pre_end_tok": pre_end_tok,
            "refl_end_tok": refl_end_tok,
        }

    def _complete_rows_recursively(
        base_prompt_ids_list,
        rows,
        remaining_list,
        reset_prompt_ids_list=None,
    ):
        if reset_prompt_ids_list is None:
            reset_prompt_ids_list = base_prompt_ids_list
        states = [
            {
                "ids": [],
                "num_reflections": 0,
                "terminated_by_cap": False,
                "truncated": False,
                "generation_failed": False,
            }
            for _ in rows
        ]
        active = []
        for idx, row in enumerate(rows):
            active.append(
                {
                    "branch_idx": idx,
                    "stage_text": row["response_text"],
                    "stage_ids": row["response_ids"],
                    "base_prompt_ids": list(base_prompt_ids_list[idx]),
                    "reset_prompt_ids": list(reset_prompt_ids_list[idx]),
                    "remaining_reflections": int(remaining_list[idx]),
                }
            )

        while active:
            prompt_specs = []
            prompt_ids_batch = []
            for node in active:
                branch_idx = node["branch_idx"]
                states[branch_idx]["truncated"] = (
                    states[branch_idx]["truncated"] or _stage_truncated(node["stage_ids"])
                )
                parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])
                if parsed is None:
                    _append_ids_capped(states[branch_idx], node["stage_ids"])
                    continue
                if node["remaining_reflections"] <= 0:
                    appended = _append_ids_capped(states[branch_idx], node["stage_ids"])
                    if appended >= int(parsed["refl_end_tok"]):
                        states[branch_idx]["num_reflections"] += 1
                    states[branch_idx]["terminated_by_cap"] = True
                    continue

                if (
                    not _can_append_full(states[branch_idx], parsed["refl_end_tok"])
                    or len(states[branch_idx]["ids"]) + int(parsed["refl_end_tok"])
                    >= max_total_response_length
                ):
                    appended = _append_ids_capped(states[branch_idx], node["stage_ids"])
                    if appended >= int(parsed["refl_end_tok"]):
                        states[branch_idx]["num_reflections"] += 1
                    states[branch_idx]["truncated"] = True
                    continue

                _append_ids_capped(
                    states[branch_idx], node["stage_ids"][: parsed["refl_end_tok"]]
                )
                states[branch_idx]["num_reflections"] += 1
                cont_prompt_ids = _reflection_cont_prompt_ids(
                    node["reset_prompt_ids"],
                    node["base_prompt_ids"] + states[branch_idx]["ids"],
                    parsed["reflection_ids"],
                )
                prompt_specs.append(
                    {
                        "branch_idx": branch_idx,
                        "base_prompt_ids": node["base_prompt_ids"],
                        "reset_prompt_ids": node["reset_prompt_ids"],
                        "remaining_reflections": node["remaining_reflections"] - 1,
                    }
                )
                prompt_ids_batch.append(cont_prompt_ids)

            if not prompt_specs:
                break
            next_rows = _generate_from_prompt_ids_chunked(
                prompt_ids_batch, n=1, warn_prefix="[PRETRIGGER-BASELINE-CONT]"
            )
            next_active = []
            for prompt_idx, spec in enumerate(prompt_specs):
                if prompt_idx >= len(next_rows):
                    branch_idx = spec["branch_idx"]
                    states[branch_idx]["truncated"] = True
                    states[branch_idx]["generation_failed"] = True
                    continue
                row = next_rows[prompt_idx]
                next_active.append(
                    {
                        "branch_idx": spec["branch_idx"],
                        "stage_text": row["response_text"],
                        "stage_ids": row["response_ids"],
                        "base_prompt_ids": spec["base_prompt_ids"],
                        "reset_prompt_ids": spec["reset_prompt_ids"],
                        "remaining_reflections": spec["remaining_reflections"],
                    }
                )
            active = next_active

        completed = []
        for state in states:
            completed.append(
                {
                    "response_ids": list(state["ids"]),
                    "response_text": self.tokenizer.decode(state["ids"], skip_special_tokens=True),
                    "num_reflections": int(state["num_reflections"]),
                    "terminated_by_cap": bool(state["terminated_by_cap"]),
                    "truncated": bool(state["truncated"]),
                    "generation_failed": bool(state["generation_failed"]),
                }
            )
        return completed

    def _rebuild_training_batch(sample_specs):
        prompt_batch = _build_prompt_batch(
            [spec["prompt_ids"] for spec in sample_specs],
            result.meta_info,
            n=1,
        )
        prompt_ids = prompt_batch.batch["input_ids"].to(result.batch["input_ids"].device)
        prompt_attention_mask = prompt_batch.batch["attention_mask"].to(
            result.batch["attention_mask"].device
        )
        prompt_position_ids = prompt_batch.batch["position_ids"].to(
            result.batch["position_ids"].device
        )

        batch_size_local = len(sample_specs)
        max_resp_len = max(1, max(len(spec["response_ids"]) for spec in sample_specs))
        final_response_ids = torch.full(
            (batch_size_local, max_resp_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        final_loss_mask = torch.zeros(
            (batch_size_local, max_resp_len),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        for idx, spec in enumerate(sample_specs):
            ids = spec["response_ids"]
            if ids:
                final_response_ids[idx, : len(ids)] = torch.tensor(
                    ids, dtype=prompt_ids.dtype, device=prompt_ids.device
                )
            for start, end in spec.get("loss_spans", []):
                start = min(max(0, int(start)), len(ids))
                end = min(max(start, int(end)), len(ids))
                if end > start:
                    final_loss_mask[idx, start:end] = 1

        eos_token_id = result.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        response_attention_mask = verl_F.get_response_mask(
            response_id=final_response_ids,
            eos_token=eos_token_id,
            dtype=prompt_attention_mask.dtype,
        )
        final_loss_mask = final_loss_mask * response_attention_mask
        delta_position_id = torch.arange(
            1, final_response_ids.size(1) + 1, device=prompt_position_ids.device
        ).unsqueeze(0).repeat(batch_size_local, 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

        rebuilt = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": final_response_ids,
                "input_ids": torch.cat((prompt_ids, final_response_ids), dim=-1),
                "attention_mask": torch.cat((prompt_attention_mask, response_attention_mask), dim=-1),
                "position_ids": torch.cat((prompt_position_ids, response_position_ids), dim=-1),
                "loss_mask": final_loss_mask,
            },
            meta_info=result.meta_info,
        )
        result.batch = rebuilt.batch

    response_ids = result.batch["responses"]
    batch_size = response_ids.shape[0]
    if batch_size == 0:
        return result, 0.0

    full_input_ids = result.batch["input_ids"]
    full_attn_mask = result.batch["attention_mask"]
    prompt_len_padded = full_input_ids.shape[1] - response_ids.shape[1]
    response_attn_mask = full_attn_mask[:, prompt_len_padded:]
    initial_response_id_lists = [
        _strip_response_ids(response_ids[i], response_attn_mask[i]) for i in range(batch_size)
    ]
    initial_response_texts = [
        self.tokenizer.decode(ids, skip_special_tokens=True) for ids in initial_response_id_lists
    ]
    header_prompt_ids = {}
    for idx in range(batch_size):
        prompt_ids_padded = full_input_ids[idx, :prompt_len_padded]
        prompt_attn = full_attn_mask[idx, :prompt_len_padded]
        header_prompt_ids[idx] = prompt_ids_padded[prompt_attn.bool()].tolist()

    reward_model_info = result.non_tensor_batch.get("reward_model", None)
    ground_truths = []
    for idx in range(batch_size):
        if reward_model_info is not None:
            rm_i = reward_model_info[idx]
            gt = rm_i.get("ground_truth", "") if isinstance(rm_i, dict) else str(rm_i)
        else:
            gt = ""
        ground_truths.append(gt)

    states = {
        idx: {
            "ids": [],
            "events": [],
            "num_reflections": 0,
            "terminated_by_cap": False,
            "truncated": False,
            "generation_failed": False,
        }
        for idx in range(batch_size)
    }
    active = [
        {
            "sample_idx": idx,
            "stage_text": initial_response_texts[idx],
            "stage_ids": initial_response_id_lists[idx],
            "header_ids": header_prompt_ids[idx],
            "context_prompt_ids": header_prompt_ids[idx],
            "remaining_reflections": remaining_reflections,
        }
        for idx in range(batch_size)
    ]

    baseline_samples = []
    event_count = 0
    round_idx = 0

    while active:
        round_idx += 1
        cont_specs = []
        cont_prompt_ids_batch = []

        for node in active:
            sample_idx = node["sample_idx"]
            state = states[sample_idx]
            state["truncated"] = state["truncated"] or _stage_truncated(node["stage_ids"])
            parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])

            if parsed is None:
                _append_ids_capped(state, node["stage_ids"])
                continue

            if node["remaining_reflections"] <= 0:
                appended = _append_ids_capped(state, node["stage_ids"])
                if appended >= int(parsed["refl_end_tok"]):
                    state["num_reflections"] += 1
                state["terminated_by_cap"] = True
                continue

            if (
                not _can_append_full(state, parsed["refl_end_tok"])
                or len(state["ids"]) + int(parsed["refl_end_tok"]) >= max_total_response_length
            ):
                appended = _append_ids_capped(state, node["stage_ids"])
                if appended >= int(parsed["refl_end_tok"]):
                    state["num_reflections"] += 1
                state["truncated"] = True
                continue

            prefix_start = len(state["ids"])
            refl_start = prefix_start + parsed["pre_end_tok"]
            refl_end = prefix_start + parsed["refl_end_tok"]
            event_id = event_count
            event_count += 1

            pretrigger_prompt_ids = _truncate_prompt(
                node["context_prompt_ids"] + node["stage_ids"][: parsed["pre_end_tok"]]
            )
            event_record = {
                "event_id": int(event_id),
                "source_idx": int(sample_idx),
                "span_start": int(refl_start),
                "span_end": -1,
                "baseline_prompt_ids": list(pretrigger_prompt_ids),
                "remaining_reflections": int(node["remaining_reflections"]),
            }
            state["events"].append(event_record)

            _append_ids_capped(state, node["stage_ids"][: parsed["refl_end_tok"]])
            state["num_reflections"] += 1
            cont_prompt_ids = _reflection_cont_prompt_ids(
                node["header_ids"],
                node["context_prompt_ids"] + node["stage_ids"][: parsed["refl_end_tok"]],
                parsed["reflection_ids"],
            )
            cont_specs.append(
                {
                    "sample_idx": sample_idx,
                    "header_ids": node["header_ids"],
                    "context_prompt_ids": cont_prompt_ids,
                    "remaining_reflections": node["remaining_reflections"] - 1,
                }
            )
            cont_prompt_ids_batch.append(cont_prompt_ids)

        if not cont_prompt_ids_batch:
            break
        next_rows = _generate_from_prompt_ids_chunked(
            cont_prompt_ids_batch, n=1, warn_prefix="[PRETRIGGER-ORIGINAL-CONT]"
        )
        next_active = []
        for prompt_idx, spec in enumerate(cont_specs):
            if prompt_idx >= len(next_rows):
                state = states[int(spec["sample_idx"])]
                state["truncated"] = True
                state["generation_failed"] = True
                continue
            row = next_rows[prompt_idx]
            next_active.append(
                {
                    "sample_idx": spec["sample_idx"],
                    "stage_text": row["response_text"],
                    "stage_ids": row["response_ids"],
                    "header_ids": spec["header_ids"],
                    "context_prompt_ids": spec["context_prompt_ids"],
                    "remaining_reflections": spec["remaining_reflections"],
                }
            )
        active = next_active

    candidate_events = []
    for idx in range(batch_size):
        state = states[idx]
        events = state["events"]
        for event_idx, event in enumerate(events):
            if event_idx + 1 < len(events):
                event["span_end"] = int(events[event_idx + 1]["span_start"])
            else:
                event["span_end"] = int(len(state["ids"]))
            if int(event["span_end"]) > int(event["span_start"]):
                candidate_events.append(event)

    event_rng = np.random.default_rng()
    source_limited_events = list(candidate_events)
    if max_events_per_source > 0:
        grouped_by_source: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for event in candidate_events:
            grouped_by_source[int(event["source_idx"])].append(event)
        source_limited_events = []
        for source_idx in sorted(grouped_by_source):
            picked = _sample_event_items(
                grouped_by_source[source_idx],
                max_events_per_source,
                event_rng,
            )
            source_limited_events.extend(picked)
        source_limited_events.sort(key=lambda event: int(event["event_id"]))

    selected_events = _sample_event_items(source_limited_events, max_events_per_step, event_rng)
    selected_events.sort(key=lambda event: int(event["event_id"]))
    selected_event_ids = {int(event["event_id"]) for event in selected_events}

    if selected_events:
        baseline_prompt_ids_batch = [event["baseline_prompt_ids"] for event in selected_events]
        baseline_groups = _generate_from_prompt_ids_chunked(
            baseline_prompt_ids_batch,
            n=k_baseline,
            warn_prefix="[PRETRIGGER-BASELINE]",
        )
        flat_rows = []
        flat_prompts = []
        flat_reset_prompts = []
        flat_remaining = []
        flat_specs = []
        for prompt_idx, spec in enumerate(selected_events):
            group = baseline_groups[prompt_idx] if prompt_idx < len(baseline_groups) else []
            for row in group:
                flat_rows.append(row)
                flat_prompts.append(spec["baseline_prompt_ids"])
                flat_reset_prompts.append(header_prompt_ids[int(spec["source_idx"])])
                flat_remaining.append(
                    0 if baseline_no_reflection else spec["remaining_reflections"]
                )
                flat_specs.append(spec)

        completed = _complete_rows_recursively(
            flat_prompts,
            flat_rows,
            flat_remaining,
            flat_reset_prompts,
        )
        for spec, prompt_ids, completed_row in zip(flat_specs, flat_prompts, completed):
            baseline_samples.append(
                {
                    "prompt_ids": list(prompt_ids),
                    "response_ids": list(completed_row["response_ids"]),
                    "source_idx": int(spec["source_idx"]),
                    "role": "pretrigger_baseline",
                    "event_id": int(spec["event_id"]),
                    "event_ids": [int(spec["event_id"])],
                    "event_spans": [],
                    "loss_spans": [(0, len(completed_row["response_ids"]))],
                    "num_reflections": 0,
                    "baseline_num_reflections": int(completed_row["num_reflections"]),
                    "terminated_by_cap": bool(completed_row["terminated_by_cap"]),
                    "truncated": bool(completed_row["truncated"]),
                    "generation_failed": bool(completed_row["generation_failed"]),
                }
            )

    expanded_samples = []
    for idx in range(batch_size):
        state = states[idx]
        events = state["events"]
        selected_source_events = [
            event for event in events if int(event["event_id"]) in selected_event_ids
        ]

        if selected_source_events:
            spans = [
                (int(event["span_start"]), int(event["span_end"]))
                for event in selected_source_events
                if int(event["span_end"]) > int(event["span_start"])
            ]
            expanded_samples.append(
                {
                    "prompt_ids": list(header_prompt_ids[idx]),
                    "response_ids": list(state["ids"]),
                    "source_idx": int(idx),
                    "role": "event_reflection_original",
                    "event_id": -1,
                    "event_ids": [int(event["event_id"]) for event in selected_source_events],
                    "event_spans": spans,
                    "loss_spans": spans,
                    "num_reflections": int(state["num_reflections"]),
                    "baseline_num_reflections": 0,
                    "terminated_by_cap": bool(state["terminated_by_cap"]),
                    "truncated": bool(state["truncated"]),
                    "generation_failed": bool(state["generation_failed"]),
                }
            )
        else:
            is_direct_no_reflection = not events and int(state["num_reflections"]) <= 0
            if not drop_direct_original and is_direct_no_reflection:
                expanded_samples.append(
                    {
                        "prompt_ids": list(header_prompt_ids[idx]),
                        "response_ids": list(state["ids"]),
                        "source_idx": int(idx),
                        "role": "direct",
                        "event_id": -1,
                        "event_ids": [],
                        "event_spans": [],
                        "loss_spans": [(0, len(state["ids"]))],
                        "num_reflections": int(state["num_reflections"]),
                        "baseline_num_reflections": 0,
                        "terminated_by_cap": bool(state["terminated_by_cap"]),
                        "truncated": bool(state["truncated"]),
                        "generation_failed": bool(state["generation_failed"]),
                    }
                )

    expanded_samples.extend(baseline_samples)
    if not expanded_samples:
        # No reflection event was found in this rollout batch.  Keep the direct
        # samples as zero-advantage placeholders so the trainer does not receive
        # an empty DataProto, but mark them with a role ignored by the advantage
        # patch.  This preserves the "do not train direct answers" objective.
        for idx in range(batch_size):
            state = states[idx]
            expanded_samples.append(
                {
                    "prompt_ids": list(header_prompt_ids[idx]),
                    "response_ids": list(state["ids"]),
                    "source_idx": int(idx),
                    "role": "direct_dropped",
                    "event_id": -1,
                    "event_ids": [],
                    "event_spans": [],
                    "loss_spans": [(0, len(state["ids"]))],
                    "num_reflections": int(state["num_reflections"]),
                    "baseline_num_reflections": 0,
                    "terminated_by_cap": bool(state["terminated_by_cap"]),
                    "truncated": bool(state["truncated"]),
                    "generation_failed": bool(state["generation_failed"]),
                }
            )

    source_indices = np.array([sample["source_idx"] for sample in expanded_samples], dtype=np.int64)
    expanded_batch_size = len(expanded_samples)
    orig_non_tensor_batch = dict(result.non_tensor_batch)

    _rebuild_training_batch(expanded_samples)

    new_non_tensor_batch = {}
    managed_keys = {
        "uid",
        "traj_uid",
        "tool_callings",
        "raw_prompt_ids",
        "raw_prompt",
        "pretrigger_sample_role",
        "pretrigger_event_id",
        "pretrigger_event_ids",
        "pretrigger_event_spans",
        "pretrigger_source_idx",
        "pretrigger_loss_span_count",
        "pretrigger_baseline_k",
        "pretrigger_drop_direct_original",
        "pretrigger_baseline_num_reflections",
        "num_reflections",
        "n_reflections",
        "has_reflection",
        "terminated_by_cap",
        "truncated",
        "generation_failed",
    }
    for key, value in orig_non_tensor_batch.items():
        if key in managed_keys:
            continue
        if key == "raw_prompt_ids":
            continue
        if isinstance(value, np.ndarray) and len(value) == batch_size:
            new_non_tensor_batch[key] = value[source_indices]
            continue
        if isinstance(value, list) and len(value) == batch_size:
            new_non_tensor_batch[key] = np.array([value[i] for i in source_indices], dtype=object)
            continue
        new_non_tensor_batch[key] = value

    uids = np.asarray(orig_non_tensor_batch.get("uid", np.arange(batch_size)), dtype=object)
    role_arr = np.empty(expanded_batch_size, dtype=object)
    role_arr[:] = [sample["role"] for sample in expanded_samples]
    event_ids_obj = np.empty(expanded_batch_size, dtype=object)
    event_ids_obj[:] = [sample["event_ids"] for sample in expanded_samples]
    event_spans_obj = np.empty(expanded_batch_size, dtype=object)
    event_spans_obj[:] = [sample["event_spans"] for sample in expanded_samples]

    new_non_tensor_batch["uid"] = np.array([uids[i] for i in source_indices], dtype=object)
    new_non_tensor_batch["traj_uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(expanded_batch_size)], dtype=object
    )
    new_non_tensor_batch["tool_callings"] = np.zeros(expanded_batch_size, dtype=np.float32)
    new_non_tensor_batch["raw_prompt_ids"] = np.array(
        [np.array(sample["prompt_ids"], dtype=np.int64) for sample in expanded_samples],
        dtype=object,
    )
    new_non_tensor_batch["raw_prompt"] = np.array(
        [
            self.tokenizer.decode(sample["prompt_ids"], skip_special_tokens=True)
            for sample in expanded_samples
        ],
        dtype=object,
    )
    new_non_tensor_batch["pretrigger_sample_role"] = role_arr
    new_non_tensor_batch["pretrigger_event_id"] = np.array(
        [sample["event_id"] for sample in expanded_samples], dtype=object
    )
    new_non_tensor_batch["pretrigger_event_ids"] = event_ids_obj
    new_non_tensor_batch["pretrigger_event_spans"] = event_spans_obj
    new_non_tensor_batch["pretrigger_source_idx"] = source_indices.astype(np.float32)
    new_non_tensor_batch["pretrigger_loss_span_count"] = np.array(
        [len(sample["loss_spans"]) for sample in expanded_samples], dtype=np.float32
    )
    new_non_tensor_batch["pretrigger_baseline_k"] = np.full(
        expanded_batch_size, float(k_baseline), dtype=np.float32
    )
    new_non_tensor_batch["pretrigger_drop_direct_original"] = np.full(
        expanded_batch_size, float(drop_direct_original), dtype=np.float32
    )
    new_non_tensor_batch["pretrigger_baseline_num_reflections"] = np.array(
        [sample["baseline_num_reflections"] for sample in expanded_samples], dtype=np.float32
    )
    new_non_tensor_batch["num_reflections"] = np.array(
        [sample["num_reflections"] for sample in expanded_samples], dtype=np.float32
    )
    new_non_tensor_batch["n_reflections"] = new_non_tensor_batch["num_reflections"]
    new_non_tensor_batch["has_reflection"] = np.array(
        [float(sample["role"] == "event_reflection_original") for sample in expanded_samples],
        dtype=np.float32,
    )
    new_non_tensor_batch["terminated_by_cap"] = np.array(
        [sample["terminated_by_cap"] for sample in expanded_samples], dtype=bool
    )
    new_non_tensor_batch["truncated"] = np.array(
        [sample["truncated"] for sample in expanded_samples], dtype=bool
    )
    new_non_tensor_batch["generation_failed"] = np.array(
        [sample["generation_failed"] for sample in expanded_samples], dtype=bool
    )
    result.non_tensor_batch = new_non_tensor_batch
    result.meta_info["pretrigger_total_events"] = int(event_count)
    result.meta_info["pretrigger_candidate_events"] = int(len(candidate_events))
    result.meta_info["pretrigger_selected_events"] = int(len(selected_events))
    result.meta_info["pretrigger_max_events_per_step"] = int(max_events_per_step)
    result.meta_info["pretrigger_max_events_per_source"] = int(max_events_per_source)
    result.meta_info["pretrigger_event_sample_mode"] = str(event_sample_mode)
    result.meta_info["pretrigger_reflection_cont_prompt_mode"] = str(
        reflection_cont_prompt_mode
    )
    result.meta_info["pretrigger_max_total_response_length"] = int(max_total_response_length)

    print(
        "[PRETRIGGER-GRPO] "
        f"initial={batch_size} expanded={expanded_batch_size} events={event_count} "
        f"candidates={len(candidate_events)} selected={len(selected_events)} "
        f"baselines={len(baseline_samples)} k={k_baseline} "
        f"max_events={max_events_per_step} max_per_source={max_events_per_source} "
        f"max_total_resp={max_total_response_length} "
        f"sample_mode={event_sample_mode} cont_mode={reflection_cont_prompt_mode} "
        f"drop_direct={int(drop_direct_original)} "
        f"baseline_no_reflect={int(baseline_no_reflection)} "
        f"rounds={round_idx}",
        flush=True,
    )
    return result, 0.0


def compute_pretrigger_baseline_advantage(
    data,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    step_advantage_w=1.0,
    gigpo_mode="mean_std_norm",
    gigpo_enable_similarity=False,
    gigpo_similarity_thresh=0.95,
    **kwargs,
):
    """Drop-in replacement for ray_trainer.compute_advantage."""

    if not _as_bool(os.environ.get("PRETRIGGER_BASELINE_GRPO_ENABLE", "0")):
        return _ORIGINAL_COMPUTE_ADVANTAGE(
            data,
            adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            multi_turn=multi_turn,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            step_advantage_w=step_advantage_w,
            gigpo_mode=gigpo_mode,
            gigpo_enable_similarity=gigpo_enable_similarity,
            gigpo_similarity_thresh=gigpo_similarity_thresh,
            **kwargs,
        )

    from verl.trainer.ppo import ray_trainer as _ray_trainer

    if adv_estimator != _ray_trainer.AdvantageEstimator.GRPO:
        return _ORIGINAL_COMPUTE_ADVANTAGE(
            data,
            adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            multi_turn=multi_turn,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            step_advantage_w=step_advantage_w,
            gigpo_mode=gigpo_mode,
            gigpo_enable_similarity=gigpo_enable_similarity,
            gigpo_similarity_thresh=gigpo_similarity_thresh,
            **kwargs,
        )

    nt = data.non_tensor_batch
    if "pretrigger_sample_role" not in nt:
        return _ORIGINAL_COMPUTE_ADVANTAGE(
            data,
            adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            multi_turn=multi_turn,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            step_advantage_w=step_advantage_w,
            gigpo_mode=gigpo_mode,
            gigpo_enable_similarity=gigpo_enable_similarity,
            gigpo_similarity_thresh=gigpo_similarity_thresh,
            **kwargs,
        )

    if "response_mask" not in data.batch:
        data.batch["response_mask"] = _ray_trainer.compute_response_mask(data)

    batch_size = data.batch["token_level_rewards"].shape[0]
    response_len = data.batch["token_level_rewards"].shape[1]
    device = data.batch["token_level_rewards"].device
    dtype = data.batch["token_level_rewards"].dtype

    score_source = nt.get("score", None)
    if score_source is None:
        score_source = nt.get("task_reward", None)
    scores = _float_array(score_source, batch_size, default=np.nan)
    missing = np.isnan(scores)
    if np.any(missing):
        summed = data.batch["token_level_rewards"].detach().sum(dim=-1).cpu().numpy()
        scores[missing] = summed[missing]
    binary_source = nt.get("is_correct", None)
    if binary_source is None:
        binary_source = nt.get("task_reward", None)
    binary_scores = _float_array(binary_source, batch_size, default=np.nan)

    roles = np.asarray(nt.get("pretrigger_sample_role"), dtype=object)
    uids = np.asarray(nt.get("uid", np.arange(batch_size)), dtype=object)
    event_id_arr = _object_array(nt.get("pretrigger_event_id", None), batch_size)
    event_ids_arr = _object_array(nt.get("pretrigger_event_ids", None), batch_size)
    event_spans_arr = _object_array(nt.get("pretrigger_event_spans", None), batch_size)
    baseline_num_reflections_arr = _float_array(
        nt.get("pretrigger_baseline_num_reflections", None), batch_size, default=0.0
    )
    generation_failed_arr = np.asarray(
        nt.get("generation_failed", np.zeros(batch_size, dtype=bool)), dtype=bool
    )
    truncated_arr = np.asarray(nt.get("truncated", np.zeros(batch_size, dtype=bool)), dtype=bool)
    terminated_by_cap_arr = np.asarray(
        nt.get("terminated_by_cap", np.zeros(batch_size, dtype=bool)), dtype=bool
    )

    response_mask = data.batch["response_mask"].to(dtype=dtype, device=device)
    if "loss_mask" in data.batch:
        train_mask = data.batch["loss_mask"][:, -response_len:].to(dtype=dtype, device=device)
    else:
        train_mask = response_mask

    advantages = torch.zeros_like(data.batch["token_level_rewards"], dtype=dtype)
    std_floor = _safe_float(os.environ.get("PRETRIGGER_EVENT_STD_FLOOR", "1e-6"), 1e-6)
    direct_weight = _safe_float(os.environ.get("PRETRIGGER_DIRECT_AUX_WEIGHT", "0.0"), 0.0)

    event_to_items: Dict[str, List[Tuple[str, int, int, int, float, float]]] = defaultdict(list)
    for idx, role in enumerate(roles):
        role = str(role)
        valid_len = int(response_mask[idx].sum().item())
        if valid_len <= 0:
            continue
        if role == "event_reflection_original":
            eids = _as_list(event_ids_arr[idx])
            spans = _normalize_spans(event_spans_arr[idx], response_len)
            for event_id, (start, end) in zip(eids, spans):
                event_to_items[str(event_id)].append(
                    ("original", idx, start, end, float(scores[idx]), float(binary_scores[idx]))
                )
        elif role == "pretrigger_baseline":
            event_to_items[str(event_id_arr[idx])].append(
                ("baseline", idx, 0, valid_len, float(scores[idx]), float(binary_scores[idx]))
            )

    event_adv_values: List[float] = []
    event_rewards: List[float] = []
    event_pool_sizes: List[int] = []
    reflected_rewards: List[float] = []
    baseline_rewards: List[float] = []
    useful_flags: List[float] = []
    harmful_flags: List[float] = []
    zero_signal_flags: List[float] = []
    original_item_count = 0
    baseline_item_count = 0

    for items in event_to_items.values():
        if not items:
            continue
        rewards = [item[4] for item in items]
        binary_rewards = [item[5] for item in items]
        advs = _grpo_norm(
            rewards,
            binary_values=binary_rewards,
            norm_by_std=bool(norm_adv_by_std_in_grpo),
            std_floor=std_floor,
        )
        event_pool_sizes.append(len(items))
        event_rewards.extend(rewards)
        zero_signal_flags.append(float(all(abs(r - rewards[0]) <= 1e-6 for r in rewards)))

        original_reward = None
        baseline_vals = []
        for item in items:
            if item[0] == "original" and original_reward is None:
                original_reward = item[4]
            elif item[0] == "baseline":
                baseline_vals.append(item[4])
        if original_reward is not None:
            reflected_rewards.append(float(original_reward))
            if baseline_vals:
                baseline_mean = float(np.mean(baseline_vals))
                baseline_rewards.extend([float(v) for v in baseline_vals])
                useful_flags.append(float(original_reward > baseline_mean + 1e-6))
                harmful_flags.append(float(original_reward + 1e-6 < baseline_mean))

        for (kind, idx, start, end, _reward, _binary_reward), adv in zip(items, advs):
            if end <= start:
                continue
            advantages[idx, start:end] += float(adv)
            event_adv_values.append(float(adv))
            if kind == "original":
                original_item_count += 1
            else:
                baseline_item_count += 1

    uid_to_direct: Dict[str, List[int]] = defaultdict(list)
    if abs(float(direct_weight)) > 1e-12:
        for idx, role in enumerate(roles):
            if str(role) == "direct":
                uid_to_direct[str(uids[idx])].append(idx)

    direct_adv_values: List[float] = []
    direct_rewards: List[float] = []
    direct_pool_sizes: List[int] = []
    for indices in uid_to_direct.values():
        if len(indices) < 2:
            continue
        rewards = [float(scores[idx]) for idx in indices]
        binary_rewards = [float(binary_scores[idx]) for idx in indices]
        advs = _grpo_norm(
            rewards,
            binary_values=binary_rewards,
            norm_by_std=bool(norm_adv_by_std_in_grpo),
            std_floor=std_floor,
        )
        direct_pool_sizes.append(len(indices))
        direct_rewards.extend(rewards)
        for idx, adv in zip(indices, advs):
            valid_len = int(response_mask[idx].sum().item())
            if valid_len <= 0:
                continue
            weighted_adv = float(direct_weight * adv)
            advantages[idx, :valid_len] += weighted_adv
            direct_adv_values.append(weighted_adv)

    advantages = advantages * train_mask
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages

    valid = train_mask > 0
    if valid.any():
        valid_adv = advantages[valid].detach().float().cpu().numpy()
        zero_token_ratio = float(np.mean(np.abs(valid_adv) < 1e-6)) if valid_adv.size else 1.0
    else:
        zero_token_ratio = 1.0

    total_events_meta = _safe_int(
        data.meta_info.get("pretrigger_total_events", len(event_to_items)), len(event_to_items)
    )
    candidate_events_meta = _safe_int(
        data.meta_info.get("pretrigger_candidate_events", len(event_to_items)), len(event_to_items)
    )
    selected_events_meta = _safe_int(
        data.meta_info.get("pretrigger_selected_events", len(event_to_items)), len(event_to_items)
    )
    max_events_meta = _safe_int(data.meta_info.get("pretrigger_max_events_per_step", 0), 0)
    max_events_per_source_meta = _safe_int(
        data.meta_info.get("pretrigger_max_events_per_source", 0), 0
    )
    max_total_response_length_meta = _safe_int(
        data.meta_info.get("pretrigger_max_total_response_length", 0), 0
    )
    cont_mode_meta = str(
        data.meta_info.get("pretrigger_reflection_cont_prompt_mode", "reflection_only")
    )
    cont_mode_is_reflection_only = 1.0 if cont_mode_meta == "reflection_only" else 0.0
    baseline_mask = np.asarray(
        [str(role) == "pretrigger_baseline" for role in roles], dtype=bool
    )
    baseline_reflection_counts = baseline_num_reflections_arr[baseline_mask]
    baseline_has_internal_reflection = baseline_reflection_counts > 0
    baseline_depth_1 = baseline_reflection_counts == 1
    baseline_depth_2 = baseline_reflection_counts == 2
    baseline_depth_3plus = baseline_reflection_counts >= 3
    direct_count = int(np.sum(np.asarray([str(role) == "direct" for role in roles], dtype=np.int32)))
    direct_dropped_count = int(
        np.sum(np.asarray([str(role) == "direct_dropped" for role in roles], dtype=np.int32))
    )

    metrics = {
        "pretrigger_grpo/events/count": int(len(event_to_items)),
        "pretrigger_grpo/events/total_count": int(total_events_meta),
        "pretrigger_grpo/events/candidate_count": int(candidate_events_meta),
        "pretrigger_grpo/events/selected_count": int(selected_events_meta),
        "pretrigger_grpo/events/max_per_step": int(max_events_meta),
        "pretrigger_grpo/events/max_per_source": int(max_events_per_source_meta),
        "pretrigger_grpo/response/max_total_length": int(max_total_response_length_meta),
        "pretrigger_grpo/events/selected_fraction": (
            float(selected_events_meta / max(1, candidate_events_meta))
            if candidate_events_meta > 0
            else 0.0
        ),
        "pretrigger_grpo/cont_mode/reflection_only": float(cont_mode_is_reflection_only),
        "pretrigger_grpo/events/pool_size_mean": float(np.mean(event_pool_sizes))
        if event_pool_sizes
        else 0.0,
        "pretrigger_grpo/events/reward_mean": float(np.mean(event_rewards))
        if event_rewards
        else 0.0,
        "pretrigger_grpo/events/original_count": int(original_item_count),
        "pretrigger_grpo/events/baseline_count": int(baseline_item_count),
        "pretrigger_grpo/reflected/acc": float(np.mean(reflected_rewards))
        if reflected_rewards
        else 0.0,
        "pretrigger_grpo/baseline/acc": float(np.mean(baseline_rewards))
        if baseline_rewards
        else 0.0,
        "pretrigger_grpo/baseline/internal_reflection_rate": float(
            np.mean(baseline_has_internal_reflection)
        )
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/internal_reflection_count_mean": float(
            np.mean(baseline_reflection_counts)
        )
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/depth_0_rate": float(
            np.mean(baseline_reflection_counts <= 0)
        )
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/depth_1_rate": float(np.mean(baseline_depth_1))
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/depth_2_rate": float(np.mean(baseline_depth_2))
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/depth_3plus_rate": float(np.mean(baseline_depth_3plus))
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/truncated_rate": float(np.mean(truncated_arr[baseline_mask]))
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/baseline/terminated_by_cap_rate": float(
            np.mean(terminated_by_cap_arr[baseline_mask])
        )
        if baseline_reflection_counts.size
        else 0.0,
        "pretrigger_grpo/delta_acc": (
            float(np.mean(reflected_rewards)) - float(np.mean(baseline_rewards))
            if reflected_rewards and baseline_rewards
            else 0.0
        ),
        "pretrigger_grpo/useful_rate": float(np.mean(useful_flags)) if useful_flags else 0.0,
        "pretrigger_grpo/harmful_rate": float(np.mean(harmful_flags)) if harmful_flags else 0.0,
        "pretrigger_grpo/local_zero_signal_rate": float(np.mean(zero_signal_flags))
        if zero_signal_flags
        else 0.0,
        "pretrigger_grpo/advantage/event_mean": float(np.mean(event_adv_values))
        if event_adv_values
        else 0.0,
        "pretrigger_grpo/advantage/direct_aux_mean": float(np.mean(direct_adv_values))
        if direct_adv_values
        else 0.0,
        "pretrigger_grpo/advantage/zero_token_ratio": zero_token_ratio,
        "pretrigger_grpo/direct/count": int(direct_count),
        "pretrigger_grpo/direct_aux/weight": float(direct_weight),
        "pretrigger_grpo/direct_aux/pool_count": int(len(direct_pool_sizes)),
        "pretrigger_grpo/direct_aux/pool_size_mean": float(np.mean(direct_pool_sizes))
        if direct_pool_sizes
        else 0.0,
        "pretrigger_grpo/direct_aux/acc": float(np.mean(direct_rewards))
        if direct_rewards
        else 0.0,
        "pretrigger_grpo/direct_dropped/count": int(direct_dropped_count),
        "pretrigger_grpo/truncated/rate": float(np.mean(truncated_arr))
        if truncated_arr.size
        else 0.0,
        "pretrigger_grpo/terminated_by_cap/rate": float(np.mean(terminated_by_cap_arr))
        if terminated_by_cap_arr.size
        else 0.0,
        "pretrigger_grpo/generation_failed/count": int(np.sum(generation_failed_arr)),
        "pretrigger_grpo/generation_failed/rate": float(np.mean(generation_failed_arr))
        if generation_failed_arr.size
        else 0.0,
        "pretrigger_grpo/std_floor": float(std_floor),
    }

    existing_metrics = data.meta_info.get("hybrid_metrics", {})
    if isinstance(existing_metrics, dict):
        existing_metrics.update(metrics)
        data.meta_info["hybrid_metrics"] = existing_metrics
    else:
        data.meta_info["hybrid_metrics"] = metrics

    return data


def install_pretrigger_baseline_grpo_patches() -> None:
    """Install runtime patches exactly once."""

    global _PATCHED, _ORIGINAL_REFLECTION_GRPO_LOOP, _ORIGINAL_COMPUTE_ADVANTAGE
    if _PATCHED:
        return
    if not _as_bool(os.environ.get("PRETRIGGER_BASELINE_GRPO_ENABLE", "0")):
        return

    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl.trainer.ppo import ray_trainer as _ray_trainer

    _ORIGINAL_REFLECTION_GRPO_LOOP = TrajectoryCollector.reflection_grpo_loop
    _ORIGINAL_COMPUTE_ADVANTAGE = _ray_trainer.compute_advantage

    TrajectoryCollector.reflection_grpo_loop = pretrigger_baseline_reflection_grpo_loop
    _ray_trainer.compute_advantage = compute_pretrigger_baseline_advantage
    _PATCHED = True
    print("[PRETRIGGER-GRPO] Installed rollout and advantage runtime patches.")


install_pretrigger_baseline_grpo_patches()
