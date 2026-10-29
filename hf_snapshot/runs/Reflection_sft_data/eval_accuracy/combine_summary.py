"""
combine_summary.py — Merge per-model result files into a single summary table.

Usage:
    cd /home/xiaofeng/wrx_project/reasoning_paths_directional/eval_accuracy
    python combine_summary.py                    # default: limo dataset
    python combine_summary.py --dataset math500
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODEL_ORDER = ["baseline", "phase1_ckpt1600", "phase_merged", "phase2", "phase2_v2"]


def build_summary(all_results, dataset_name):
    lines = [f"Dataset: {dataset_name}  |  Models: {len(all_results)}", ""]

    col_heads = ["Overall", "TotRefl", "AvgRefl", "ReflRate"]
    header    = f"{'Model':<18} | " + " | ".join(f"{h:>8}" for h in col_heads)
    sep       = "-" * len(header)
    lines    += [sep, header, sep]

    for tag in MODEL_ORDER:
        if tag not in all_results:
            continue
        results   = all_results[tag]
        n         = len(results)
        overall   = sum(r["correct"] for r in results) / n
        tot_refl  = sum(r["num_reflections"] for r in results)
        avg_refl  = tot_refl / n
        refl_rate = sum(1 for r in results if r["num_reflections"] > 0) / n
        lines.append(
            f"{tag:<18} | {overall:>7.1%}  | {tot_refl:>7d} | {avg_refl:>7.3f} | {refl_rate:>7.1%}"
        )

    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="limo", choices=["limo", "math500"])
    args = parser.parse_args()

    output_dir = Path(".")
    all_results = {}

    for tag in MODEL_ORDER:
        path = output_dir / f"results_{args.dataset}_{tag}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        results = []
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
        all_results[tag] = results
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"  [loaded] {tag}: {len(results)} samples, acc={acc:.2%}")

    if not all_results:
        print("No result files found.")
        return

    summary = build_summary(all_results, args.dataset)
    print("\n" + summary + "\n")

    out = output_dir / f"summary_{args.dataset}.txt"
    with open(out, "w") as f:
        f.write(summary + "\n")
    print(f"[Summary] -> {out}")


if __name__ == "__main__":
    main()
