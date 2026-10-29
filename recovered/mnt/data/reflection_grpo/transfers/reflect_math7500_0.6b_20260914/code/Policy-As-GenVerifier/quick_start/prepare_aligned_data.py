"""Convert reference GRPO JSONL to PAG/SCoRe parquet without changing rows/prompts."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert(source, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new output directory")
    rows = []
    with source.open() as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            prompt = item["question"]
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"{source}:{index + 1}: expected reference GRPO chat messages")
            rows.append({
                "prompt": prompt,
                "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                "data_source": item.get("data_source", "math"),
                "ability": "math",
                "extra_info": {"index": index},
            })
    if not rows:
        raise ValueError(f"Empty input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    print(f"{source} -> {destination}: {len(rows)} rows, original order and prompts preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("/root/reflection_grpo_clean/data/rl/limo_grpo.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("/root/reflection_grpo_clean/data/rl/math500_grpo.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    args = parser.parse_args()
    pairs = [(args.train, args.output_dir / "limo_aligned.parquet"),
             (args.val, args.output_dir / "math500_aligned.parquet")]
    for source, destination in pairs:
        if not source.is_file():
            parser.error(f"Missing input: {source}")
        if destination.exists():
            parser.error(f"Output already exists: {destination}; choose a new --output-dir")
    for source, destination in pairs:
        convert(source, destination)


if __name__ == "__main__":
    main()
