"""Summarize pass@N reasoning rebuttal eval outputs."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "artifacts" / "evaluation"))
MODELS = os.environ.get("MODELS", "refl_dg520 sft grpo base").split()
DATASETS = [
    d.strip()
    for d in os.environ.get("EVAL_DATASETS", "bbh_all").split(",")
    if d.strip()
]


def fmt_pct(x):
    if x is None:
        return "NA"
    return f"{100 * float(x):.1f}"


def fmt_num(x):
    if x is None:
        return "NA"
    return f"{float(x):.1f}"


def fmt_mean_std(values: list[float], pct: bool = True) -> str:
    if not values:
        return "NA"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if pct:
        return f"{100 * mean:.1f}±{100 * std:.1f}"
    return f"{mean:.2f}±{std:.2f}"


def seed_dirs() -> list[Path]:
    dirs = sorted(p for p in RESULTS_ROOT.glob("seed_*") if p.is_dir())
    return dirs if dirs else [RESULTS_ROOT]


def load_summary(src: Path, model: str, dataset: str) -> dict[str, Any] | None:
    path = src / f"{model}__{dataset}.summary.json"
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def main() -> None:
    sources = seed_dirs()
    rows = []
    aggregate: dict[str, Any] = {
        "results_root": str(RESULTS_ROOT),
        "seeds": [p.name.removeprefix("seed_") for p in sources if p != RESULTS_ROOT],
        "models": MODELS,
        "datasets": DATASETS,
        "items": {},
    }
    for model in MODELS:
        for dataset in DATASETS:
            summaries = [
                s
                for src in sources
                for s in [load_summary(src, model, dataset)]
                if s is not None
            ]
            if not summaries:
                rows.append([model, dataset, "0", "MISSING", "", "", "", "", "", "", "", ""])
                continue

            n = summaries[0].get("n_samples_per_q", "?")
            key = f"{model}__{dataset}"
            item = {
                "n_seeds": len(summaries),
                "n_questions": summaries[0].get("n_questions"),
                "n_samples_per_q": n,
                "mean_at_n": [float(s["mean_at_n"]) for s in summaries if s.get("mean_at_n") is not None],
                "pass_at_n": [float(s["pass_at_n"]) for s in summaries if s.get("pass_at_n") is not None],
                "maj_at_n": [float(s["maj_at_n"]) for s in summaries if s.get("maj_at_n") is not None],
                "reflection_trigger_rate": [
                    float(s["reflection_trigger_rate"])
                    for s in summaries
                    if s.get("reflection_trigger_rate") is not None
                ],
                "avg_n_iters": [
                    float(s["avg_n_iters"]) for s in summaries if s.get("avg_n_iters") is not None
                ],
                "avg_prompt_tokens": [
                    float(s["avg_prompt_tokens"])
                    for s in summaries
                    if s.get("avg_prompt_tokens") is not None
                ],
                "avg_decoded_tokens": [
                    float(s["avg_decoded_tokens"])
                    for s in summaries
                    if s.get("avg_decoded_tokens") is not None
                ],
                "prompt_truncated_rate": [
                    float(s["prompt_truncated_rate"])
                    for s in summaries
                    if s.get("prompt_truncated_rate") is not None
                ],
            }
            aggregate["items"][key] = item
            rows.append(
                [
                    model,
                    dataset,
                    str(len(summaries)),
                    str(summaries[0].get("n_questions", "")),
                    fmt_mean_std(item["mean_at_n"]),
                    fmt_mean_std(item["pass_at_n"]),
                    fmt_mean_std(item["maj_at_n"]),
                    fmt_mean_std(item["reflection_trigger_rate"]),
                    fmt_mean_std(item["avg_n_iters"], pct=False),
                    fmt_mean_std(item["avg_prompt_tokens"], pct=False),
                    fmt_mean_std(item["avg_decoded_tokens"], pct=False),
                    fmt_mean_std(item["prompt_truncated_rate"]),
                    f"@{n}",
                ]
            )

    headers = [
        "Model",
        "Dataset",
        "#Seeds",
        "#Q",
        "avg@N",
        "pass@N",
        "maj@N",
        "Trigger",
        "Avg iters",
        "Prompt tok",
        "Decoded tok",
        "Trunc",
        "N",
    ]
    lines = ["# Reasoning Rebuttal Eval Summary", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULTS_ROOT / "summary_reasoning.md").open("w") as f:
        f.write("\n".join(lines))
    with (RESULTS_ROOT / "summary_reasoning.json").open("w") as f:
        json.dump({"rows": rows, "headers": headers, "aggregate": aggregate}, f, indent=2)
    print(RESULTS_ROOT / "summary_reasoning.md")
    print(RESULTS_ROOT / "summary_reasoning.json")


if __name__ == "__main__":
    main()
