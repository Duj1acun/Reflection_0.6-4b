#!/usr/bin/env python3
"""Re-select from an existing rollout_scores.jsonl without re-running vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from rgrpo.select_rl_rollouts import passes_filter, rl_row


def load_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-file", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="Samples per condition used in the original rollout run.")
    parser.add_argument("--min-direct-accuracy", type=float, default=0.2)
    parser.add_argument("--max-direct-accuracy", type=float, default=0.8)
    parser.add_argument("--exclude-extremes-only", action="store_true")
    parser.add_argument("--require-reflection-improvement", action="store_true")
    parser.add_argument("--data-source-tag", type=str, default="math_rescue_filtered")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_scores(args.scores_file)
    if not records:
        raise ValueError(f"No records found in {args.scores_file}")

    selected = []
    for record in records:
        reflection_improvement = (
            record.get("reflection_correct") is not None
            and record["reflection_correct"] / args.n > record["direct_correct"] / args.n
        )
        if passes_filter(
            record["direct_correct"], args.n, args.min_direct_accuracy, args.max_direct_accuracy, args.exclude_extremes_only
        ) and (reflection_improvement or not args.require_reflection_improvement):
            selected.append(rl_row(record, len(selected), args.data_source_tag))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(selected).to_parquet(str(args.output_dir / f"{args.data_source_tag}.parquet"))
    summary = {
        "source_scores_file": str(args.scores_file),
        "candidates": len(records),
        "selected": len(selected),
        "n": args.n,
        "direct_accuracy_bounds": [args.min_direct_accuracy, args.max_direct_accuracy],
        "exclude_extremes_only": args.exclude_extremes_only,
        "require_reflection_improvement": args.require_reflection_improvement,
    }
    (args.output_dir / "refilter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
