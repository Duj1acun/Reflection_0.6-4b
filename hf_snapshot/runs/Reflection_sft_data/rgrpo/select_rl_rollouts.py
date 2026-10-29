#!/usr/bin/env python3
"""Score math questions via direct (and optionally reflection-prompt) rollouts.

Two selection modes:
  * quota mode (--target-total): bucket every scored question into
    easy (direct_correct == n) / medium (0 < direct_correct < n) /
    hard (direct_correct == 0), and keep exactly the requested quota per
    bucket. Candidates are read and scored in chunks so scoring can stop
    early once every quota is filled.
  * legacy mode (no --target-total): single-pass accuracy-band filtering,
    kept for backward compatibility with earlier rescue-style selection.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm import tqdm

DIRECT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
REFLECTION_SYSTEM = (
    "You are a reasoning assistant. When you make an error during reasoning, "
    "use <reflection>...</reflection> to diagnose the mistake, then continue "
    "reasoning based on the reflection. Put your final answer within \\boxed{}."
)
TIERS = ("easy", "medium", "hard")


def _score_worker(connection) -> None:
    from rgrpo.math_score_light import compute_math_score_payload

    try:
        connection.send({"ready": True})
        while True:
            item = connection.recv()
            if item is None:
                return
            generated, ground_truth, timeout = item
            result = compute_math_score_payload(
                {"solution_str": generated, "ground_truth": ground_truth, "timeout": timeout}
            )
            connection.send(result)
    except (EOFError, OSError):
        return
    finally:
        connection.close()


class HardTimeoutScorer:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.retry_timeout = timeout * 2
        self.startup_timeout = 30.0
        self.context = mp.get_context("spawn")
        self.process = None
        self.parent_connection = None
        self._start_worker()

    def _start_worker(self) -> None:
        self.parent_connection, child_connection = self.context.Pipe()
        self.process = self.context.Process(target=_score_worker, args=(child_connection,))
        self.process.start()
        child_connection.close()
        if not self.parent_connection.poll(self.startup_timeout):
            self.process.terminate()
            self.process.join(timeout=1)
            self.parent_connection.close()
            raise RuntimeError("score worker failed to start within 30 seconds")
        ready = self.parent_connection.recv()
        if not isinstance(ready, dict) or not ready.get("ready"):
            raise RuntimeError("score worker sent an invalid startup response")

    def _restart_worker(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        if self.parent_connection is not None:
            self.parent_connection.close()
        self._start_worker()

    def _score_once(self, generated: str, ground_truth: str, timeout: float) -> tuple[float, bool]:
        try:
            self.parent_connection.send((generated, ground_truth, timeout))
            if not self.parent_connection.poll(timeout + 2.0):
                self._restart_worker()
                return 0.0, True
            result = self.parent_connection.recv()
            return float(result.get("score", 0.0)), bool(result.get("math_score_timeout", 0.0))
        except (EOFError, OSError):
            self._restart_worker()
            return 0.0, True

    def score(self, generated: str, ground_truth: str) -> float:
        score, timed_out = self._score_once(generated, ground_truth, self.timeout)
        if not timed_out:
            return score
        score, _ = self._score_once(generated, ground_truth, self.retry_timeout)
        return score

    def close(self) -> None:
        try:
            if self.parent_connection is not None:
                self.parent_connection.send(None)
        except (EOFError, OSError, BrokenPipeError):
            pass
        if self.process is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        if self.parent_connection is not None:
            self.parent_connection.close()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def prompt(tokenizer, system: str, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def rl_row(row: dict[str, Any], index: int, data_source: str, tier: str | None = None) -> dict[str, Any]:
    extra_info = {"index": index, "source_id": str(index)}
    if "source" in row:
        extra_info["source"] = row["source"]
    if tier is not None:
        extra_info["difficulty_tier"] = tier
    return {
        "data_source": data_source,
        "question": [{"role": "user", "content": row["question"].strip()}],
        "reward_model": {"ground_truth": str(row["answer"]).strip()},
        "extra_info": extra_info,
    }


def tier_for(direct_correct: int, n: int) -> str:
    if direct_correct == 0:
        return "hard"
    if direct_correct == n:
        return "easy"
    return "medium"


def passes_filter(
    direct_correct: int,
    n: int,
    min_accuracy: float,
    max_accuracy: float,
    exclude_extremes_only: bool,
) -> bool:
    if exclude_extremes_only:
        return 0 < direct_correct < n
    accuracy = direct_correct / n
    return min_accuracy <= accuracy <= max_accuracy


def score_rows(rows, direct_outputs, reflection_outputs, n: int, scorer: HardTimeoutScorer) -> list[dict[str, Any]]:
    records = []
    for row, direct_output, reflection_output in tqdm(
        zip(rows, direct_outputs, reflection_outputs),
        total=len(rows),
        desc="Scoring rows",
        unit="row",
    ):
        gold = str(row["answer"])
        direct_scores = [int(scorer.score(out.text, gold) > 0.5) for out in direct_output.outputs]
        direct_correct = sum(direct_scores)
        record = {
            "question": row["question"], "answer": gold,
            "direct_accuracy": direct_correct / n, "direct_correct": direct_correct,
            "tier": tier_for(direct_correct, n),
        }
        if reflection_output is not None:
            reflection_scores = [int(scorer.score(out.text, gold) > 0.5) for out in reflection_output.outputs]
            reflection_correct = sum(reflection_scores)
            record["reflection_correct"] = reflection_correct
            record["reflection_accuracy"] = reflection_correct / n
            record["reflection_improvement_observed"] = int(reflection_correct > direct_correct)
        else:
            record["reflection_correct"] = None
            record["reflection_accuracy"] = None
            record["reflection_improvement_observed"] = None
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=8, help="Samples per condition and question.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=4000, help="Quota mode: rows scored per vLLM batch.")
    parser.add_argument(
        "--skip-reflection", action="store_true",
        help="Only sample the direct condition (halves rollout cost); reflection fields are recorded as null.",
    )
    parser.add_argument("--min-direct-accuracy", type=float, default=0.2)
    parser.add_argument("--max-direct-accuracy", type=float, default=0.8)
    parser.add_argument(
        "--require-reflection-improvement",
        action="store_true",
        help="Legacy mode only: keep only questions where reflection-prompt accuracy exceeds direct accuracy.",
    )
    parser.add_argument(
        "--exclude-extremes-only",
        action="store_true",
        help="Legacy mode only: ignore accuracy bounds; keep any question with 0 < direct_correct < n.",
    )
    parser.add_argument(
        "--target-total", type=int, default=None,
        help="Enable quota mode: build exactly this many rows split across easy/medium/hard tiers.",
    )
    parser.add_argument("--easy-fraction", type=float, default=0.17)
    parser.add_argument("--medium-fraction", type=float, default=0.55)
    parser.add_argument("--hard-fraction", type=float, default=0.28)
    parser.add_argument("--data-source-tag", type=str, default="math_rescue_filtered")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument(
        "--score-timeout", type=float, default=15.0,
        help="Initial per-answer scoring timeout; a timed-out answer is retried with double this budget.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data/rollout_selection"))
    args = parser.parse_args()
    if not 0 <= args.min_direct_accuracy <= args.max_direct_accuracy <= 1:
        raise ValueError("Direct-accuracy bounds must satisfy 0 <= min <= max <= 1")
    if args.skip_reflection and args.require_reflection_improvement:
        raise ValueError("--require-reflection-improvement needs reflection sampling; drop --skip-reflection")
    if args.n < 1 or args.chunk_size < 1 or args.score_timeout <= 0:
        raise ValueError("n, chunk-size, and score-timeout must be positive")

    from vllm import LLM, SamplingParams

    rows = load_rows(args.input, args.limit)
    if not rows:
        raise ValueError("No candidate questions loaded")
    llm = LLM(model=args.model_path, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    scorer = HardTimeoutScorer(args.score_timeout)

    def generate_scores(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
        direct_out = llm.generate([prompt(tokenizer, DIRECT_SYSTEM, row["question"]) for row in chunk], sampling, use_tqdm=True)
        if args.skip_reflection:
            reflection_out = [None] * len(chunk)
        else:
            reflection_out = llm.generate([prompt(tokenizer, REFLECTION_SYSTEM, row["question"]) for row in chunk], sampling, use_tqdm=True)
        return score_rows(chunk, direct_out, reflection_out, args.n, scorer)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.target_total is not None:
            fractions = {"easy": args.easy_fraction, "medium": args.medium_fraction, "hard": args.hard_fraction}
            if any(fraction < 0 for fraction in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 0.01:
                raise ValueError(f"easy/medium/hard fractions must be >=0 and sum to ~1.0, got {fractions}")
            quotas = {"easy": round(args.target_total * fractions["easy"]), "hard": round(args.target_total * fractions["hard"])}
            quotas["medium"] = args.target_total - quotas["easy"] - quotas["hard"]

            buckets: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
            all_records, position = [], 0
            scores_path = args.output_dir / "rollout_scores.jsonl"
            with scores_path.open("w", encoding="utf-8") as score_handle:
                while position < len(rows) and any(len(buckets[tier]) < quotas[tier] for tier in TIERS):
                    chunk = rows[position: position + args.chunk_size]
                    position += len(chunk)
                    chunk_records = generate_scores(chunk)
                    for row, record in zip(chunk, chunk_records):
                        tier = record["tier"]
                        record["selected"] = len(buckets[tier]) < quotas[tier]
                        if record["selected"]:
                            buckets[tier].append(row)
                        all_records.append(record)
                        score_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    score_handle.flush()

            combined = [(row, tier) for tier in TIERS for row in buckets[tier]]
            random.Random(args.seed).shuffle(combined)
            selected = [rl_row(row, index, args.data_source_tag, tier) for index, (row, tier) in enumerate(combined)]

            Dataset.from_list(selected).to_parquet(str(args.output_dir / f"{args.data_source_tag}.parquet"))
            achieved = {tier: len(buckets[tier]) for tier in TIERS}
            summary = {
                "mode": "quota", "candidates_available": len(rows), "candidates_processed": position,
                "target_total": args.target_total, "quotas": quotas, "achieved": achieved,
                "shortfall": {tier: quotas[tier] - achieved[tier] for tier in TIERS if achieved[tier] < quotas[tier]},
                "n": args.n, "skip_reflection": args.skip_reflection, "score_timeout": args.score_timeout,
            }
        else:
            records = generate_scores(rows)
            selected = []
            for row, record in zip(rows, records):
                reflection_improvement = bool(record["reflection_improvement_observed"])
                record["selected"] = passes_filter(
                    record["direct_correct"], args.n, args.min_direct_accuracy, args.max_direct_accuracy, args.exclude_extremes_only
                ) and (reflection_improvement or not args.require_reflection_improvement)
                if record["selected"]:
                    selected.append(rl_row(row, len(selected), args.data_source_tag))

            with (args.output_dir / "rollout_scores.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            Dataset.from_list(selected).to_parquet(str(args.output_dir / f"{args.data_source_tag}.parquet"))
            summary = {
                "mode": "legacy", "candidates": len(rows), "selected": len(selected), "n": args.n,
                "direct_accuracy_bounds": [args.min_direct_accuracy, args.max_direct_accuracy],
                "exclude_extremes_only": args.exclude_extremes_only, "score_timeout": args.score_timeout,
            }

        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
    finally:
        scorer.close()


if __name__ == "__main__":
    main()