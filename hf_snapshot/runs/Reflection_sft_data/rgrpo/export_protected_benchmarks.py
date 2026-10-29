#!/usr/bin/env python3
"""Export locally cached eval-only benchmarks to JSONL for RL data protection.

These splits must NEVER be used as RL training candidates. Exporting them to
JSONL lets prepare_rl_candidates.py / audit_rl_data.py de-duplicate against
them using the same normalized-text matching as other protected sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# (repo, config, split, output_name, question_field)
BENCHMARKS = [
    ("openai/gsm8k", "main", "test", "gsm8k_test.jsonl", "question"),
    ("math-ai/amc23", None, "test", "amc23_test.jsonl", "question"),
    ("jeggers/logiqa2_formatted", None, "test", "logiqa2_test.jsonl", "question"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data/protected_exports"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for repo, config, split, output_name, question_field in BENCHMARKS:
        dataset = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        output_path = args.output_dir / output_name
        with output_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps({"question": row[question_field]}, ensure_ascii=False) + "\n")
        print(f"{repo} ({split}): {len(dataset)} rows -> {output_path}")


if __name__ == "__main__":
    main()
