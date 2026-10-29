#!/usr/bin/env python3
"""Convert local LIMO and MATH-500 files to the verl GRPO JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def convert_row(row: dict[str, Any], data_source: str, index: int) -> dict[str, Any]:
    question = row.get("question", row.get("problem"))
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{data_source} row {index} has no question or problem")
    if answer is None or not str(answer).strip():
        raise ValueError(f"{data_source} row {index} has no answer")

    return {
        "data_source": data_source,
        "question": [{"role": "user", "content": question.strip()}],
        "reward_model": {"ground_truth": str(answer).strip()},
        "extra_info": {
            "index": index,
            "source_solution": row.get("solution", ""),
            "source_id": row.get("unique_id", str(index)),
        },
    }


def convert_file(input_path: Path, output_path: Path, data_source: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_rows = []
    count = 0
    with input_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            converted_rows.append(convert_row(row, data_source, index))
            count += 1
    Dataset.from_list(converted_rows).to_parquet(str(output_path))
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limo-input", type=Path, default=Path("/mnt/data/datasets/GAIR-LIMO/limo.jsonl"))
    parser.add_argument("--math500-input", type=Path, default=Path("/mnt/data/datasets/MATH-500/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data"))
    args = parser.parse_args()

    limo_count = convert_file(args.limo_input, args.output_dir / "limo_train.parquet", "limo")
    math500_count = convert_file(args.math500_input, args.output_dir / "math500_val.parquet", "math500")
    print(f"Wrote {limo_count} LIMO training rows and {math500_count} MATH-500 validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()