#!/usr/bin/env python3
"""Account for decoded and processed tokens in existing eval artifacts.

Processed tokens are counted as:

    sum_over_rounds(prompt_tokens_for_that_round + decoded_tokens_for_that_round)

For REFLECT+ samples, the continuation prompt is the clean-slate prompt used by
the eval harness: ``header + latest_reflection_tag.rstrip() + "\n"``. This
script reconstructs that prompt from the saved full text and dataset problem.
It does not run inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TEST_DIR = REPO / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from datasets_loader import load_dataset  # noqa: E402
from rgrpo.reflection import REFLECTION_SYSTEM_PROMPT  # noqa: E402


QWEN_MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
REASONING_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
REFLECTION_REASONING_SYSTEM_PROMPT = (
    "You are a reasoning assistant. When you make an error during reasoning, "
    "use <reflection>...</reflection> to diagnose the mistake, then continue "
    "reasoning based on the reflection. Put your final answer within \\boxed{}."
)
REFL_PATTERN = re.compile(r"<reflection>[\s\S]+?</reflection>", re.DOTALL)


def system_prompt_for(kind: str) -> str:
    if kind == "reflection":
        return REFLECTION_SYSTEM_PROMPT
    if kind == "qwen_math":
        return QWEN_MATH_SYSTEM_PROMPT
    if kind == "reasoning":
        return REASONING_SYSTEM_PROMPT
    if kind == "reflection_reasoning":
        return REFLECTION_REASONING_SYSTEM_PROMPT
    raise ValueError(f"unknown prompt kind: {kind!r}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    out: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


def apply_token_truncation(
    ids: list[int],
    max_len: int,
    mode: str,
) -> list[int]:
    if len(ids) <= max_len:
        return ids
    if mode == "left":
        return ids[-max_len:]
    if mode == "right":
        return ids[:max_len]
    if mode == "middle":
        left_half = max_len // 2
        right_half = max_len - left_half
        return ids[:left_half] + ids[-right_half:]
    if mode == "error":
        raise RuntimeError(f"prompt length {len(ids)} exceeds max_len={max_len}")
    raise ValueError(f"unknown truncation mode: {mode!r}")


class HeaderCache:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._cache: dict[tuple[str, str, int, str], tuple[list[str], list[int]]] = {}

    def dataset_rows(self, dataset: str) -> list[dict[str, Any]]:
        if dataset not in self._rows:
            try:
                self._rows[dataset] = load_dataset(dataset)
            except FileNotFoundError:
                if dataset not in {"aime24", "aime25", "aime26"}:
                    raise
                self._rows[dataset] = load_aime_fallback(dataset)
        return self._rows[dataset]

    def headers(
        self,
        dataset: str,
        prompt_kind: str,
        max_prompt_len: int,
        truncation: str,
    ) -> tuple[list[str], list[int]]:
        key = (dataset, prompt_kind, max_prompt_len, truncation)
        if key in self._cache:
            return self._cache[key]

        sys_prompt = system_prompt_for(prompt_kind)
        headers: list[str] = []
        lengths: list[int] = []
        for row in self.dataset_rows(dataset):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": row["problem"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = apply_token_truncation(ids, max_prompt_len, truncation)
            headers.append(self.tokenizer.decode(ids, skip_special_tokens=False))
            lengths.append(len(ids))
        self._cache[key] = (headers, lengths)
        return self._cache[key]


def load_aime_fallback(dataset: str) -> list[dict[str, Any]]:
    """Fallback for AIME when the old JSONL path in datasets_loader is absent."""
    from datasets import load_dataset as hf_load_dataset

    if dataset == "aime24":
        ds = hf_load_dataset("HuggingFaceH4/aime_2024", split="train")
        year = 2024
    elif dataset == "aime25":
        ds = hf_load_dataset("math-ai/aime25", split="test")
        year = 2025
    elif dataset == "aime26":
        ds = hf_load_dataset("math-ai/aime26", split="test")
        year = 2026
    else:
        raise KeyError(dataset)

    out = []
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        out.append(
            {
                "problem": str(row["problem"]),
                "gold": str(row["answer"]).strip(),
                "kind": "math",
                "uid": f"aime{year}-{rid}",
            }
        )
    return out


def seed_sources(root: Path) -> list[Path]:
    seeds = sorted(p for p in root.glob("seed_*") if p.is_dir())
    if root.name == "results_seeded":
        # The existing 5-seed summary treats test/results as seed_1234 and
        # test/results_seeded/seed_* as the other four seeds.
        main_seed = REPO / "test/results"
        return [main_seed] + seeds
    return seeds if seeds else [root]


def discover_pairs(root: Path, models: set[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    sources = sorted(root.glob("seed_*")) if root.name == "results_seeded" else seed_sources(root)
    for src in sources:
        for path in src.glob("*.summary.json"):
            model, dataset = path.name.removesuffix(".summary.json").split("__", 1)
            if model in models and (src / f"{model}__{dataset}.samples.jsonl").is_file():
                pairs.add((model, dataset))
    return pairs


def process_samples(
    sample_path: Path,
    summary: dict[str, Any],
    tokenizer,
    header_cache: HeaderCache,
    batch_size: int,
) -> dict[str, Any]:
    dataset = summary["dataset"]
    prompt_kind = summary["prompt_kind"]
    max_prompt_len = int(summary.get("max_prompt_len", 1024))
    truncation = str(summary.get("prompt_truncation", "left"))
    max_tokens = int(summary.get("max_tokens_per_round", 1024))
    max_cont_tokens = max_prompt_len + max_tokens // 2
    headers, header_lens = header_cache.headers(
        dataset, prompt_kind, max_prompt_len, truncation
    )

    totals = {
        "n_samples": 0,
        "decoded_sum": 0.0,
        "processed_sum": 0.0,
        "first_prompt_sum": 0.0,
        "extra_prefill_sum": 0.0,
        "correct_sum": 0.0,
        "num_reflections_sum": 0.0,
        "n_iters_sum": 0.0,
        "missing_reflection_tags": 0,
    }

    def flush(chunk: list[dict[str, Any]]) -> None:
        if not chunk:
            return
        decoded_texts = [
            row["full_text"]
            for row in chunk
            if row.get("decoded_tokens") is None
        ]
        decoded_lens_iter = iter(token_lengths(tokenizer, decoded_texts, batch_size))

        cont_prompts: list[str] = []
        row_base_values: list[tuple[float, float, float, float, int]] = []
        for row in chunk:
            qi = int(row["qi"])
            prompt_len = int(row.get("prompt_tokens") or header_lens[qi])
            if row.get("decoded_tokens") is None:
                decoded_len = float(next(decoded_lens_iter))
            else:
                decoded_len = float(row["decoded_tokens"])

            n_refl = int(row.get("num_reflections", 0) or 0)
            tags = [m.group(0) for m in REFL_PATTERN.finditer(row.get("full_text", ""))]
            if len(tags) < n_refl:
                totals["missing_reflection_tags"] += n_refl - len(tags)
            for tag in tags[:n_refl]:
                raw = headers[qi] + tag.rstrip() + "\n"
                ids = tokenizer.encode(raw, add_special_tokens=False)
                if len(ids) > max_cont_tokens:
                    raw = tokenizer.decode(ids[-max_cont_tokens:], skip_special_tokens=False)
                cont_prompts.append(raw)

            row_base_values.append(
                (
                    float(prompt_len),
                    decoded_len,
                    float(row.get("correct", 0) or 0),
                    float(n_refl),
                    int(row.get("n_iters", 1) or 1),
                )
            )

        cont_sum = float(sum(token_lengths(tokenizer, cont_prompts, batch_size)))
        totals["extra_prefill_sum"] += cont_sum
        for prompt_len, decoded_len, correct, n_refl, n_iters in row_base_values:
            totals["n_samples"] += 1
            totals["first_prompt_sum"] += prompt_len
            totals["decoded_sum"] += decoded_len
            totals["processed_sum"] += prompt_len + decoded_len
            totals["correct_sum"] += correct
            totals["num_reflections_sum"] += n_refl
            totals["n_iters_sum"] += n_iters
        totals["processed_sum"] += cont_sum

    chunk: list[dict[str, Any]] = []
    with sample_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            if len(chunk) >= batch_size:
                flush(chunk)
                chunk = []
    flush(chunk)

    n = max(1, int(totals["n_samples"]))
    return {
        **totals,
        "mean_decoded": totals["decoded_sum"] / n,
        "mean_processed": totals["processed_sum"] / n,
        "mean_first_prompt": totals["first_prompt_sum"] / n,
        "mean_extra_prefill": totals["extra_prefill_sum"] / n,
        "sample_accuracy": totals["correct_sum"] / n,
        "sample_trigger_rate": totals["num_reflections_sum"] / n,
        "sample_avg_n_iters": totals["n_iters_sum"] / n,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate_seed_stats(seed_stats: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mean_decoded",
        "mean_processed",
        "mean_first_prompt",
        "mean_extra_prefill",
        "sample_accuracy",
        "sample_trigger_rate",
        "sample_avg_n_iters",
    ]
    out: dict[str, Any] = {
        "n_seeds": len(seed_stats),
        "n_samples_total": int(sum(s["n_samples"] for s in seed_stats)),
        "missing_reflection_tags": int(sum(s["missing_reflection_tags"] for s in seed_stats)),
    }
    for key in keys:
        vals = [float(s[key]) for s in seed_stats]
        mean, std = mean_std(vals)
        out[key] = mean
        out[f"{key}_std"] = std
    for metric in ["mean_at_n", "pass_at_n", "maj_at_n", "reflection_trigger_rate", "avg_n_iters"]:
        vals = [float(s["summary"][metric]) for s in seed_stats if s["summary"].get(metric) is not None]
        if vals:
            mean, std = mean_std(vals)
            out[metric] = mean
            out[f"{metric}_std"] = std
    first = seed_stats[0]["summary"]
    out.update(
        {
            "model": first["model"],
            "dataset": first["dataset"],
            "n_questions": first.get("n_questions"),
            "n_samples_per_q": first.get("n_samples_per_q"),
            "prompt_kind": first.get("prompt_kind"),
            "max_tokens_per_round": first.get("max_tokens_per_round"),
            "max_prompt_len": first.get("max_prompt_len", 1024),
            "prompt_truncation": first.get("prompt_truncation", "left"),
        }
    )
    return out


def fmt_num(x: float, digits: int = 1) -> str:
    if x is None or math.isnan(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or math.isnan(float(x)):
        return "NA"
    return f"{100.0 * float(x):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            str(REPO / "test/results_seeded"),
            str(REPO / "test/results"),
            str(REPO / "test/results_bbh_all_pass8_t1_boxed_5seed"),
            str(REPO / "test/results_clutrr_drop_pass8_t1_boxed_5seed"),
        ],
    )
    ap.add_argument("--models", nargs="+", default=["grpo", "refl_dg520"])
    ap.add_argument(
        "--tokenizer-path",
        default="/ssd/data/wangruxin/R_grpo_sft/sft_training/outputs/phase2/final_merged_hf",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out-dir", default=str(REPO / "rebuttal/generated"))
    ap.add_argument("--cache-file", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    header_cache = HeaderCache(tokenizer)
    wanted_models = set(args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_file) if args.cache_file else out_dir / "context_token_accounting.cache.json"
    cached: dict[str, Any] = {}
    if cache_path.is_file():
        cached = read_json(cache_path)

    # Prefer multi-seed roots over the single aggregate root when the same
    # model/dataset pair appears in both.
    all_items: dict[tuple[str, str], dict[str, Any]] = {}
    pair_sources: dict[tuple[str, str], str] = {}
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            continue
        pairs = discover_pairs(root, wanted_models)
        for model, dataset in sorted(pairs):
            key = (model, dataset)
            if key in all_items:
                continue
            cache_key = f"{model}__{dataset}"
            cached_item = cached.get("items", {}).get(cache_key)
            cached_source = cached.get("sources", {}).get(cache_key)
            if cached_item is not None and cached_source == str(root):
                all_items[key] = cached_item
                pair_sources[key] = cached_source
                print(f"loaded cached {model}__{dataset} from {root}", flush=True)
                continue
            seed_stats = []
            for src in seed_sources(root):
                summary_path = src / f"{model}__{dataset}.summary.json"
                sample_path = src / f"{model}__{dataset}.samples.jsonl"
                if not summary_path.is_file() or not sample_path.is_file():
                    continue
                summary = read_json(summary_path)
                stats = process_samples(sample_path, summary, tokenizer, header_cache, args.batch_size)
                stats["summary"] = summary
                stats["source"] = str(src)
                seed_stats.append(stats)
            if seed_stats:
                all_items[key] = aggregate_seed_stats(seed_stats)
                pair_sources[key] = str(root)
                cache_obj = {
                    "items": {f"{m}__{d}": v for (m, d), v in all_items.items()},
                    "sources": {f"{m}__{d}": v for (m, d), v in pair_sources.items()},
                }
                with cache_path.open("w") as f:
                    json.dump(cache_obj, f, indent=2, ensure_ascii=False)
                print(f"processed {model}__{dataset} from {root} ({len(seed_stats)} seed(s))", flush=True)

    datasets = sorted({dataset for _, dataset in all_items})
    compare_rows: list[list[Any]] = []
    compare_json: dict[str, Any] = {}
    for dataset in datasets:
        grpo = all_items.get(("grpo", dataset))
        refl = all_items.get(("refl_dg520", dataset))
        if not grpo or not refl:
            continue
        dec_delta = (refl["mean_decoded"] - grpo["mean_decoded"]) / grpo["mean_decoded"]
        proc_delta = (refl["mean_processed"] - grpo["mean_processed"]) / grpo["mean_processed"]
        acc_delta = refl.get("mean_at_n", refl["sample_accuracy"]) - grpo.get("mean_at_n", grpo["sample_accuracy"])
        src = pair_sources[("refl_dg520", dataset)]
        compare_rows.append(
            [
                dataset,
                refl["n_seeds"],
                f"@{refl['n_samples_per_q']}",
                fmt_pct(grpo.get("mean_at_n", grpo["sample_accuracy"])),
                fmt_pct(refl.get("mean_at_n", refl["sample_accuracy"])),
                fmt_pct(acc_delta),
                fmt_pct(refl.get("reflection_trigger_rate", refl["sample_trigger_rate"])),
                fmt_num(grpo["mean_decoded"]),
                fmt_num(refl["mean_decoded"]),
                fmt_pct(dec_delta),
                fmt_num(grpo["mean_processed"]),
                fmt_num(refl["mean_processed"]),
                fmt_pct(proc_delta),
                Path(src).name,
            ]
        )
        compare_json[dataset] = {
            "source": src,
            "grpo": grpo,
            "refl_dg520": refl,
            "decoded_delta_frac": dec_delta,
            "processed_delta_frac": proc_delta,
            "mean_at_n_delta": acc_delta,
        }

    with (out_dir / "context_token_accounting.json").open("w") as f:
        json.dump(
            {
                "items": {f"{m}__{d}": v for (m, d), v in all_items.items()},
                "comparison": compare_json,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    md = [
        "# Context Token Accounting",
        "",
        "Processed tokens are counted as first-round prompt + all decoded tokens + clean-slate continuation prompts.",
        "For REFLECT+, each continuation prompt is reconstructed as `header + latest <reflection>...</reflection>`.",
        "",
        markdown_table(
            [
                "Dataset",
                "Seeds",
                "N",
                "GRPO acc",
                "REFLECT+ acc",
                "Acc delta",
                "REF trig",
                "GRPO decoded",
                "REF decoded",
                "Decoded delta",
                "GRPO processed",
                "REF processed",
                "Processed delta",
                "Source",
            ],
            compare_rows,
        ),
        "",
    ]
    (out_dir / "context_token_accounting.md").write_text("\n".join(md))
    print(out_dir / "context_token_accounting.md")
    print(out_dir / "context_token_accounting.json")


if __name__ == "__main__":
    main()
