#!/usr/bin/env python3
"""Build an auditable, deduplicated 8k math-RL candidate set."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


def normalize(text: Any) -> str:
    text = re.sub(r"\\(?:left|right)\b", "", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def boxed_answer(solution: str) -> str:
    matches = list(re.finditer(r"\\boxed\s*\{", solution or ""))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, chars = 1, []
    for char in solution[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
    return ""


def math_rows() -> list[dict[str, Any]]:
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval")
    rows = []
    for index, row in enumerate(dataset["train"]):
        answer = boxed_answer(row["solution"])
        if answer:
            rows.append({
                "source": "math_lighteval_train",
                "source_index": index,
                "question": row["problem"],
                "answer": answer,
                "solution": row["solution"],
                "level": row.get("level", ""),
                "type": row.get("type", ""),
            })
    return rows


def big_math_rows() -> list[dict[str, Any]]:
    dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    rows = []
    for index, row in enumerate(dataset):
        answer = str(row.get("answer", "")).strip()
        problem = str(row.get("problem", "")).strip()
        if problem and answer:
            rows.append({
                "source": "big_math_rl_verified",
                "source_index": index,
                "question": problem,
                "answer": answer,
                "solution": "",
                "level": "",
                "type": str(row.get("domain", "")),
                "source_solve_rate": row.get("llama8b_solve_rate"),
            })
    return rows


def gsm8k_rows() -> list[dict[str, Any]]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for index, row in enumerate(dataset):
        answer = row["answer"].split("####")[-1].strip()
        rows.append({
            "source": "gsm8k_train",
            "source_index": index,
            "question": row["question"].strip(),
            "answer": answer,
            "solution": row["answer"],
            "level": "",
            "type": "",
        })
    return rows


def load_protected(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("question", row.get("problem", ""))
            if isinstance(value, list):
                value = "\n".join(str(item.get("content", "")) for item in value)
            values.add(normalize(value))
    return values


def to_verl_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "data_source": row["source"],
        "question": [{"role": "user", "content": row["question"].strip()}],
        "reward_model": {"ground_truth": row["answer"].strip()},
        "extra_info": {
            "index": index,
            "source": row["source"],
            "source_index": row["source_index"],
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "source_solve_rate": row.get("source_solve_rate"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data/math8000"))
    parser.add_argument("--protected", type=Path, action="append", default=[
        Path("/mnt/data/datasets/MATH-500/test.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/sft_training/data/generated/merged/03_sft/reasoning_outputs.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/sft_training/data/generated/merged/03_sft/reflection_data_prepared_perfect.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/rl_data/protected_exports/gsm8k_test.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/rl_data/protected_exports/amc23_test.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/rl_data/protected_exports/logiqa2_test.jsonl"),
    ])
    parser.add_argument("--include-gsm8k", action="store_true", help="Add GSM8K train split as a supplemental source.")
    args = parser.parse_args()
    if args.target < 1:
        raise ValueError("--target must be positive")

    protected = set().union(*(load_protected(path) for path in args.protected))
    math = math_rows()
    supplemental = big_math_rows()
    if args.include_gsm8k:
        supplemental += gsm8k_rows()
    selected, seen = [], set(protected)
    stats = {"math_loaded": len(math), "supplemental_loaded": len(supplemental), "duplicate_or_protected": 0}

    for row in math + supplemental:
        key = normalize(row["question"])
        if not key or key in seen:
            stats["duplicate_or_protected"] += 1
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) == args.target:
            break

    if len(selected) < args.target:
        raise RuntimeError(f"Only constructed {len(selected)} independent rows, target={args.target}")

    random.Random(args.seed).shuffle(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "candidates.jsonl"
    parquet_path = args.output_dir / "candidates.parquet"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    Dataset.from_list([to_verl_row(row, index) for index, row in enumerate(selected)]).to_parquet(str(parquet_path))
    report = {"target": args.target, "selected": len(selected), "sources": {source: sum(row["source"] == source for row in selected) for source in {row["source"] for row in selected}}, "stats": stats, "protected_files": [str(path) for path in args.protected]}
    (args.output_dir / "construction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()