#!/usr/bin/env python3
"""Audit RL candidates against protected data and remove duplicate-like prompts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


def question(row: dict[str, Any]) -> str:
    value = row.get("question", row.get("problem", ""))
    if isinstance(value, list):
        return "\n".join(str(message.get("content", "")) for message in value)
    return str(value)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


def word_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def protected_questions(paths: Iterable[Path]) -> list[tuple[str, str]]:
    records = []
    for path in paths:
        for row in load_jsonl(path):
            text = question(row)
            if text.strip():
                records.append((str(path), text))
    return records


def audit(candidates: list[dict[str, Any]], protected: list[tuple[str, str]], similarity: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    protected_norm = {normalize(text) for _, text in protected}
    protected_compact = {compact(text) for _, text in protected}
    protected_by_shingle: dict[tuple[str, ...], set[int]] = {}
    protected_compacted = [compact(text) for _, text in protected]
    for protected_index, text in enumerate(protected_compacted):
        for shingle in word_shingles(text):
            protected_by_shingle.setdefault(shingle, set()).add(protected_index)
    retained, seen_norm, seen_compact = [], set(), set()
    stats = Counter()

    for row in candidates:
        text = question(row)
        normalized, compacted = normalize(text), compact(text)
        if not normalized:
            stats["missing_question"] += 1
        elif normalized in seen_norm or compacted in seen_compact:
            stats["candidate_duplicate"] += 1
        elif normalized in protected_norm or compacted in protected_compact:
            stats["protected_exact_or_normalized"] += 1
        else:
            candidate_ids: set[int] = set()
            for shingle in word_shingles(text):
                candidate_ids.update(protected_by_shingle.get(shingle, set()))
            is_similar = any(
                SequenceMatcher(None, compacted, protected_compacted[protected_index]).ratio() >= similarity
                for protected_index in candidate_ids
                if 0.8 <= len(compacted) / max(1, len(protected_compacted[protected_index])) <= 1.25
            )
            if is_similar:
                stats["protected_high_similarity"] += 1
                continue
            retained.append(row)
            seen_norm.add(normalized)
            seen_compact.add(compacted)
            stats["retained"] += 1
    return retained, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("/mnt/data/datasets/GAIR-LIMO/limo.jsonl"))
    parser.add_argument("--protected", type=Path, action="append", default=[
        Path("/mnt/data/datasets/MATH-500/test.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/sft_training/data/generated/merged/03_sft/reasoning_outputs.jsonl"),
        Path("/mnt/data/runs/Reflection_sft_data/sft_training/data/generated/merged/03_sft/reflection_data_prepared_perfect.jsonl"),
    ])
    parser.add_argument("--similarity", type=float, default=0.92)
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data/limo_audited.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("/mnt/data/runs/Reflection_sft_data/rl_data/limo_audit_report.json"))
    args = parser.parse_args()

    missing = [path for path in args.protected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected datasets not found: {missing}")
    if not 0 < args.similarity <= 1:
        raise ValueError("--similarity must be in (0, 1]")

    candidates = load_jsonl(args.candidates)
    protected = protected_questions(args.protected)
    retained, stats = audit(candidates, protected, args.similarity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "candidate_file": str(args.candidates),
        "candidate_count": len(candidates),
        "protected_files": [str(path) for path in args.protected],
        "protected_question_count": len(protected),
        "similarity_threshold": args.similarity,
        "stats": stats,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()