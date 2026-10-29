#!/usr/bin/env python3
"""Offline rebuttal analyses from existing REFLECT+ evaluation artifacts.

This script intentionally does not run training or inference. It consumes
``test/results/*.summary.json`` and ``test/results/*.samples.jsonl`` and writes
Markdown/JSON tables under ``rebuttal/generated``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


MAIN_MODELS = ["base", "sft", "grpo", "grpo500", "refl_dg520"]
MAIN_DATASETS = ["math500", "amc23", "gsm8k", "logiqa", "gpqa", "mmlu_pro"]
ABLATION_MODELS = ["grpo", "grpo500", "refl_decoupled", "refl_dg519", "refl_dg520"]
ABLATION_DATASETS = ["math500", "aime24", "aime25", "aime26", "gpqa"]
TRAIN_LOG_GLOB = "trajectory-grpo-qwen2.5-math-1.5b-dg-5-20_*.log"


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_samples(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    out: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        out.extend(len(ids) for ids in encoded["input_ids"])
    return out


def pct(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def fmt(x, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def fmt_pct(x, digits: int = 1) -> str:
    if x is None:
        return "-"
    try:
        return f"{100.0 * float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def collect_artifacts(results_dir: Path):
    summaries: dict[tuple[str, str], dict] = {}
    samples: dict[tuple[str, str], list[dict]] = {}
    for path in results_dir.glob("*.summary.json"):
        s = read_json(path)
        summaries[(s["model"], s["dataset"])] = s
    for path in results_dir.glob("*.samples.jsonl"):
        model, dataset = path.name.removesuffix(".samples.jsonl").split("__", 1)
        samples[(model, dataset)] = read_samples(path)
    return summaries, samples


def length_and_compute_analysis(
    summaries: dict[tuple[str, str], dict],
    samples: dict[tuple[str, str], list[dict]],
    tokenizer_path: str,
    batch_size: int,
) -> tuple[dict, str]:
    tokenizer = load_tokenizer(tokenizer_path)
    sample_stats: dict[str, dict] = {}

    for key, rows in sorted(samples.items()):
        model, dataset = key
        texts = [r.get("full_text", "") for r in rows]
        lengths = np.asarray(token_lengths(tokenizer, texts, batch_size), dtype=np.float64)
        correct = np.asarray([float(r.get("correct", 0)) for r in rows], dtype=np.float64)
        n_refl = np.asarray([float(r.get("num_reflections", 0)) for r in rows], dtype=np.float64)
        n_iters = np.asarray([max(1.0, float(r.get("n_iters", 1))) for r in rows], dtype=np.float64)
        summary = summaries.get(key, {})
        max_per_round = float(summary.get("max_tokens_per_round", 1024))
        per_sample_budget = max_per_round * n_iters
        budget_hit = lengths >= np.maximum(1.0, per_sample_budget - 4.0)

        depth = {}
        for label, mask in [
            ("0", n_refl == 0),
            ("1", n_refl == 1),
            ("2", n_refl == 2),
            ("3+", n_refl >= 3),
        ]:
            if not np.any(mask):
                continue
            depth[label] = {
                "n": int(mask.sum()),
                "rate": float(mask.mean()),
                "acc": float(correct[mask].mean()),
                "mean_tokens": float(lengths[mask].mean()),
                "median_tokens": float(np.median(lengths[mask])),
                "p90_tokens": pct(lengths[mask], 90),
                "budget_hit_rate": float(budget_hit[mask].mean()),
            }

        sample_stats[f"{model}__{dataset}"] = {
            "n_samples": int(len(rows)),
            "mean_tokens": float(lengths.mean()) if lengths.size else float("nan"),
            "median_tokens": float(np.median(lengths)) if lengths.size else float("nan"),
            "p90_tokens": pct(lengths, 90),
            "p95_tokens": pct(lengths, 95),
            "max_tokens": int(lengths.max()) if lengths.size else 0,
            "budget_hit_rate": float(budget_hit.mean()) if lengths.size else float("nan"),
            "mean_accuracy": float(correct.mean()) if correct.size else float("nan"),
            "depth": depth,
        }

    rows = []
    for dataset in MAIN_DATASETS:
        grpo = summaries.get(("grpo", dataset))
        refl = summaries.get(("refl_dg520", dataset))
        gstat = sample_stats.get(f"grpo__{dataset}")
        rstat = sample_stats.get(f"refl_dg520__{dataset}")
        if not grpo or not refl or not gstat or not rstat:
            continue
        g_tok = gstat["mean_tokens"]
        r_tok = rstat["mean_tokens"]
        g_acc = grpo["mean_at_n"]
        r_acc = refl["mean_at_n"]
        rows.append(
            [
                dataset,
                fmt_pct(g_acc),
                f"{g_tok:.0f}",
                fmt_pct(r_acc),
                f"{r_tok:.0f}",
                fmt_pct((r_tok - g_tok) / g_tok),
                fmt(1000.0 * g_acc / g_tok, 3),
                fmt(1000.0 * r_acc / r_tok, 3),
                fmt_pct(refl.get("reflection_trigger_rate")),
            ]
        )

    md = [
        "## Token and compute summary",
        "",
        "Mean tokens count the complete decoded answer text. For REFLECT+, this is the sum over all clean-slate rounds, not a single round.",
        "",
        markdown_table(
            [
                "Dataset",
                "GRPO acc",
                "GRPO tok",
                "REFLECT+ acc",
                "REFLECT+ tok",
                "Tok delta",
                "GRPO acc/1k tok",
                "REFLECT+ acc/1k tok",
                "REFLECT+ trig",
            ],
            rows,
        ),
    ]

    depth_rows = []
    for dataset in MAIN_DATASETS:
        stat = sample_stats.get(f"refl_dg520__{dataset}", {})
        for label in ["0", "1", "2", "3+"]:
            d = stat.get("depth", {}).get(label)
            if not d:
                continue
            depth_rows.append(
                [
                    dataset,
                    label,
                    d["n"],
                    fmt_pct(d["rate"]),
                    fmt_pct(d["acc"]),
                    f"{d['mean_tokens']:.0f}",
                    f"{d['median_tokens']:.0f}",
                    fmt_pct(d["budget_hit_rate"]),
                ]
            )
    md.extend(
        [
            "",
            "## REFLECT+ by reflection depth",
            "",
            markdown_table(
                [
                    "Dataset",
                    "# refl",
                    "Samples",
                    "Rate",
                    "Acc",
                    "Mean tok",
                    "Median tok",
                    "Budget-hit",
                ],
                depth_rows,
            ),
        ]
    )
    return sample_stats, "\n".join(md) + "\n"


def per_question_metric(rows: list[dict], metric: str) -> dict[int, float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["qi"])].append(r)
    out = {}
    for qi, recs in grouped.items():
        vals = [float(r.get("correct", 0)) for r in recs]
        if metric == "mean":
            out[qi] = float(np.mean(vals))
        elif metric == "pass":
            out[qi] = float(any(v > 0.5 for v in vals))
        else:
            raise ValueError(metric)
    return out


def paired_bootstrap_delta(
    a: dict[int, float],
    b: dict[int, float],
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    diffs = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    n = len(diffs)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(diffs[idx].mean())
    return {
        "n": n,
        "delta": float(diffs.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
    }


def stratified_overall_bootstrap(
    deltas_by_dataset: list[np.ndarray],
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    deltas_by_dataset = [d for d in deltas_by_dataset if d.size]
    if not deltas_by_dataset:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(np.mean([d.mean() for d in deltas_by_dataset]))
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        vals = []
        for diffs in deltas_by_dataset:
            idx = rng.integers(0, len(diffs), size=len(diffs))
            vals.append(float(diffs[idx].mean()))
        boot[i] = float(np.mean(vals))
    return {
        "delta": point,
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
    }


def bootstrap_analysis(samples: dict[tuple[str, str], list[dict]], n_boot: int, seed: int) -> tuple[dict, str]:
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    md = ["## Paired bootstrap confidence intervals", ""]
    for baseline in ["grpo", "grpo500"]:
        rows = []
        overall_by_metric: dict[str, list[np.ndarray]] = {"mean": [], "pass": []}
        for dataset in MAIN_DATASETS:
            refl_rows = samples.get(("refl_dg520", dataset))
            base_rows = samples.get((baseline, dataset))
            if not refl_rows or not base_rows:
                continue
            for metric in ["mean", "pass"]:
                a = per_question_metric(refl_rows, metric)
                b = per_question_metric(base_rows, metric)
                keys = sorted(set(a) & set(b))
                diffs = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
                overall_by_metric[metric].append(diffs)
                res = paired_bootstrap_delta(a, b, rng, n_boot)
                results[f"refl_dg520_vs_{baseline}__{dataset}__{metric}"] = res
                if metric == "mean":
                    rows.append(
                        [
                            dataset,
                            res["n"],
                            fmt_pct(res["delta"]),
                            f"[{fmt_pct(res['lo'])}, {fmt_pct(res['hi'])}]",
                        ]
                    )
        overall_mean = stratified_overall_bootstrap(overall_by_metric["mean"], rng, n_boot)
        overall_pass = stratified_overall_bootstrap(overall_by_metric["pass"], rng, n_boot)
        results[f"refl_dg520_vs_{baseline}__overall__mean"] = overall_mean
        results[f"refl_dg520_vs_{baseline}__overall__pass"] = overall_pass
        rows.append(
            [
                "overall avg",
                "-",
                fmt_pct(overall_mean["delta"]),
                f"[{fmt_pct(overall_mean['lo'])}, {fmt_pct(overall_mean['hi'])}]",
            ]
        )
        md.extend(
            [
                f"### REFLECT+ minus {baseline}",
                "",
                markdown_table(["Dataset", "Questions", "Mean@N delta", "95% CI"], rows),
                "",
                f"Overall pass@N delta: {fmt_pct(overall_pass['delta'])} "
                f"[{fmt_pct(overall_pass['lo'])}, {fmt_pct(overall_pass['hi'])}]",
                "",
            ]
        )
    return results, "\n".join(md).rstrip() + "\n"


def ablation_summary(summaries: dict[tuple[str, str], dict]) -> str:
    rows = []
    for model in ABLATION_MODELS:
        vals = []
        trig = []
        for dataset in ABLATION_DATASETS:
            s = summaries.get((model, dataset))
            if s:
                vals.append(float(s["mean_at_n"]))
                trig.append(float(s.get("reflection_trigger_rate", 0.0)))
        if vals:
            rows.append(
                [
                    model,
                    len(vals),
                    fmt_pct(np.mean(vals)),
                    fmt_pct(np.mean(trig)),
                    " / ".join(f"{d}:{fmt_pct(summaries[(model, d)]['mean_at_n'])}" for d in ABLATION_DATASETS if (model, d) in summaries),
                ]
            )
    return "\n".join(
        [
            "## Existing checkpoint ablations",
            "",
            "These are already-completed checkpoints on the common five-dataset subset; no new training is implied.",
            "",
            markdown_table(["Model", "# datasets", "Avg mean@N", "Avg trigger", "Per-dataset mean@N"], rows),
            "",
        ]
    )


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+):(-?(?:\d+(?:\.\d*)?|\.\d+))")


def parse_train_logs(log_dir: Path) -> tuple[dict[int, dict], str]:
    records: dict[int, dict] = {}
    for path in sorted(log_dir.glob(TRAIN_LOG_GLOB)):
        with path.open(errors="ignore") as f:
            for line in f:
                line = ANSI_RE.sub("", line)
                m = STEP_LINE_RE.search(line)
                if not m:
                    continue
                step = int(m.group(1))
                metrics = {k: float(v) for k, v in METRIC_RE.findall(m.group(2))}
                if metrics:
                    metrics["_source"] = path.name
                    records[step] = metrics
    wanted = [0, 1, 10, 20, 50, 100, 150, 200, 240, 300, 350, 400, 450]
    rows = []
    for step in wanted:
        if step not in records:
            continue
        r = records[step]
        rows.append(
            [
                step,
                fmt_pct(r.get("val/math500/accuracy")),
                fmt_pct(r.get("val/math500/has_reflection")),
                fmt_pct(r.get("reflection_grpo/trigger_rate")),
                fmt_pct(r.get("reflection_grpo/format/valid_rate")),
                fmt_pct(r.get("reflection_grpo/accuracy/direct")),
                fmt_pct(r.get("reflection_grpo/accuracy/reflection")),
                fmt(r.get("response_length/mean"), 1),
            ]
        )
    md = [
        "## Training dynamics from existing logs",
        "",
        "Parsed from local DG run logs. Blank validation cells indicate steps without validation.",
        "",
        markdown_table(
            [
                "Step",
                "Val acc",
                "Val trig",
                "Train trig",
                "Format valid",
                "Direct acc",
                "Refl acc",
                "Resp len",
            ],
            rows,
        ),
        "",
    ]
    return records, "\n".join(md)


def write_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="test/results")
    ap.add_argument("--out-dir", default="rebuttal/generated")
    ap.add_argument(
        "--tokenizer-path",
        default="/ssd/data/wangruxin/R_grpo_sft/sft_training/outputs/phase2/final_merged_hf",
    )
    ap.add_argument("--token-batch-size", type=int, default=256)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--train-log-dir", default="/ssd/data/wangruxin/outputs/train_logs")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, samples = collect_artifacts(results_dir)
    sample_stats, length_md = length_and_compute_analysis(
        summaries,
        samples,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.token_batch_size,
    )
    boot_stats, boot_md = bootstrap_analysis(samples, args.bootstrap_iters, args.seed)
    train_records, train_md = parse_train_logs(Path(args.train_log_dir))
    ablation_md = ablation_summary(summaries)

    write_json(out_dir / "sample_token_stats.json", sample_stats)
    write_json(out_dir / "bootstrap_ci.json", boot_stats)
    write_json(out_dir / "training_dynamics.json", train_records)

    report = "\n".join(
        [
            "# REFLECT+ Rebuttal Offline Analyses",
            "",
            length_md,
            boot_md,
            ablation_md,
            train_md,
        ]
    )
    (out_dir / "rebuttal_analysis.md").write_text(report)
    print(f"wrote {out_dir / 'rebuttal_analysis.md'}")


if __name__ == "__main__":
    main()
