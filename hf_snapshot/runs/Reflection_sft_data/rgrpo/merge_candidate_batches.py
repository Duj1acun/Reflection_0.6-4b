#!/usr/bin/env python3
"""Merge sequential prepare_rl_candidates.py batches into one candidate pool.

Each batch must have been built by chaining --protected onto every prior
batch's candidates.jsonl (see prepare_rl_candidates.py). This script re-checks
that invariant (zero cross-batch duplicate normalized questions) before
merging, rather than assuming the chaining was done correctly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

from rgrpo.prepare_rl_candidates import to_verl_row


def normalize(text: Any) -> str:
    text = re.sub(r"\\(?:left|right)\b", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def load_batch(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True, help="Path to a batch's candidates.jsonl, repeatable in batch order.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.batch) < 2:
        raise ValueError("Provide at least 2 --batch paths to merge")

    merged: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for batch_path in args.batch:
        rows = load_batch(batch_path)
        for row in rows:
            key = normalize(row["question"])
            if key in seen:
                raise RuntimeError(
                    f"Duplicate question across batches: {batch_path} duplicates a question "
                    f"already introduced by {seen[key]}. Fix batch construction (chain --protected) and retry."
                )
            seen[key] = batch_path
            merged.append(row)

    random.Random(args.seed).shuffle(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "candidates.jsonl"
    parquet_path = args.output_dir / "candidates.parquet"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Dataset.from_list([to_verl_row(row, index) for index, row in enumerate(merged)]).to_parquet(str(parquet_path))
    report = {
        "batches": [str(path) for path in args.batch],
        "batch_counts": [len(load_batch(path)) for path in args.batch],
        "total_merged": len(merged),
        "sources": {source: sum(row["source"] == source for row in merged) for source in {row["source"] for row in merged}},
    }
    (args.output_dir / "merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
