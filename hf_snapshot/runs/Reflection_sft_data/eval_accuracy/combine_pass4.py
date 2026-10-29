"""
combine_pass4.py — Merge pass@4 result files into a summary table.

Usage:
    python combine_pass4.py [--dataset limo|math500]
"""

import argparse
import json
from pathlib import Path

MODEL_ORDER = ["baseline", "phase2", "phase2_v2"]
N_SAMPLES = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="limo", choices=["limo", "math500"])
    args = parser.parse_args()

    output_dir = Path(".")
    all_results = {}

    for tag in MODEL_ORDER:
        path = output_dir / f"pass4_{args.dataset}_{tag}.jsonl"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        results = []
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))
        all_results[tag] = results
        pass4 = sum(r["pass4"] for r in results) / len(results)
        avg4  = sum(r["avg4"]  for r in results) / len(results)
        print(f"  [loaded] {tag}: {len(results)} problems, pass@4={pass4:.2%}, avg@4={avg4:.2%}")

    if not all_results:
        print("No result files found.")
        return

    # Also load greedy acc from eval_accuracy results for comparison
    greedy = {}
    for tag in MODEL_ORDER:
        path = output_dir / f"results_{args.dataset}_{tag}.jsonl"
        if path.exists():
            rows = [json.loads(l) for l in open(path)]
            greedy[tag] = sum(r["correct"] for r in rows) / len(rows)

    lines = [f"Dataset: {args.dataset}  |  {N_SAMPLES} samples per problem", ""]
    col_heads = ["greedy@1", "avg@4", "pass@4", "TotRefl", "AvgRefl"]
    header = f"{'Model':<14} | " + " | ".join(f"{h:>8}" for h in col_heads)
    sep    = "-" * len(header)
    lines += [sep, header, sep]

    for tag in MODEL_ORDER:
        if tag not in all_results:
            continue
        results   = all_results[tag]
        n         = len(results)
        pass4     = sum(r["pass4"] for r in results) / n
        avg4      = sum(r["avg4"]  for r in results) / n
        tot_refl  = sum(s["num_reflections"] for r in results for s in r["samples"])
        avg_refl  = tot_refl / (n * N_SAMPLES)
        g1        = greedy.get(tag, float("nan"))
        lines.append(
            f"{tag:<14} | {g1:>7.1%}  | {avg4:>7.1%}  | {pass4:>7.1%}  "
            f"| {tot_refl:>6d} | {avg_refl:>7.3f}"
        )

    lines.append(sep)
    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    out = output_dir / f"summary_pass4_{args.dataset}.txt"
    with open(out, "w") as f:
        f.write(summary + "\n")
    print(f"[Summary] -> {out}")


if __name__ == "__main__":
    main()
