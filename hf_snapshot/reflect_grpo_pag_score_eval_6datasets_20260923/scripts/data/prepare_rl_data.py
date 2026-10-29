#!/usr/bin/env python3
# Copyright 2025 R-GRPO Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare training and validation data for GRPO and R-GRPO training.

This script converts data to verl-compatible format where the prompt field
is a list of messages: [{"role": "user", "content": "..."}, ...]

Usage:
    python prepare_data.py --output_dir /path/to/data

This script generates:
    - math500_grpo.jsonl: GRPO validation set with system prompt
    - math500_rgrpo.jsonl: R-GRPO validation set (user message only)
    - limo_grpo.jsonl: Training data with GRPO system prompt
    - limo_rgrpo.jsonl: Training data for R-GRPO (user message only)
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


# Standard GRPO system prompt for mathematical reasoning
GRPO_SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. Please solve the following problem step by step and put your final answer in \\boxed{}."
)

# R-GRPO system prompt with reflection capability
RGRPO_SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."
)


def to_messages_grpo(question: str) -> list:
    """Convert question string to GRPO messages format with system prompt."""
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def to_messages_rgrpo(question: str) -> list:
    """Convert question string to R-GRPO messages format with reflection system prompt."""
    return [
        {"role": "system", "content": RGRPO_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def create_grpo_example_from_math500(example: dict) -> dict:
    """Create a GRPO example from MATH-500 dataset."""
    return {
        "question": to_messages_grpo(example["problem"]),
        "answer": example["answer"],
        "solution": example.get("solution", ""),
        "subject": example.get("subject", ""),
        "level": example.get("level", ""),
        "unique_id": example.get("unique_id", ""),
        "data_source": "math500",
    }


def create_rgrpo_example_from_math500(example: dict) -> dict:
    """Create an R-GRPO example from MATH-500 dataset."""
    return {
        "question": to_messages_rgrpo(example["problem"]),
        "answer": example["answer"],
        "solution": example.get("solution", ""),
        "subject": example.get("subject", ""),
        "level": example.get("level", ""),
        "unique_id": example.get("unique_id", ""),
        "data_source": "math500",
    }


def convert_limo_to_messages(input_path: str, output_grpo: str, output_rgrpo: str):
    """Convert limo_merged.jsonl to messages format for both GRPO and R-GRPO."""
    print(f"Converting {input_path} to messages format...")
    
    grpo_count = 0
    rgrpo_count = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_grpo, "w", encoding="utf-8") as f_grpo, \
         open(output_rgrpo, "w", encoding="utf-8") as f_rgrpo:
        
        for line in fin:
            example = json.loads(line.strip())
            question = example.get("question", "")
            
            # GRPO format: with system prompt
            grpo_example = {
                "question": to_messages_grpo(question),
                "answer": example.get("answer", ""),
                "solution": example.get("solution", ""),
                "data_source": "limo",
            }
            f_grpo.write(json.dumps(grpo_example, ensure_ascii=False) + "\n")
            grpo_count += 1
            
            # R-GRPO format: user message only
            rgrpo_example = {
                "question": to_messages_rgrpo(question),
                "answer": example.get("answer", ""),
                "solution": example.get("solution", ""),
                "data_source": "limo",
            }
            f_rgrpo.write(json.dumps(rgrpo_example, ensure_ascii=False) + "\n")
            rgrpo_count += 1
    
    print(f"  Written {grpo_count} GRPO training examples to {output_grpo}")
    print(f"  Written {rgrpo_count} R-GRPO training examples to {output_rgrpo}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare training and validation data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "data" / "rl"),
        help="Output directory for the prepared data",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset name for validation",
    )
    parser.add_argument(
        "--limo_path",
        type=str,
        default=str(project_root / "data" / "source" / "limo_merged.jsonl"),
        help="Path to limo_merged.jsonl training data",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training data conversion (only generate validation sets)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== VALIDATION DATA =====
    print(f"\n=== Loading validation dataset: {args.dataset_name} ===")
    try:
        dataset = load_dataset(args.dataset_name, split="test")
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        print("Trying alternative loading method...")
        dataset = load_dataset(args.dataset_name, split="test", trust_remote_code=True)

    print(f"Loaded {len(dataset)} validation examples")

    # Generate GRPO validation set
    grpo_val_path = output_dir / "math500_grpo.jsonl"
    print(f"Writing GRPO validation data to: {grpo_val_path}")
    with open(grpo_val_path, "w", encoding="utf-8") as f:
        for example in dataset:
            grpo_example = create_grpo_example_from_math500(example)
            f.write(json.dumps(grpo_example, ensure_ascii=False) + "\n")
    print(f"  Written {len(dataset)} GRPO validation examples")

    # Generate R-GRPO validation set
    rgrpo_val_path = output_dir / "math500_rgrpo.jsonl"
    print(f"Writing R-GRPO validation data to: {rgrpo_val_path}")
    with open(rgrpo_val_path, "w", encoding="utf-8") as f:
        for example in dataset:
            rgrpo_example = create_rgrpo_example_from_math500(example)
            f.write(json.dumps(rgrpo_example, ensure_ascii=False) + "\n")
    print(f"  Written {len(dataset)} R-GRPO validation examples")

    # ===== TRAINING DATA =====
    if not args.skip_train and os.path.exists(args.limo_path):
        print(f"\n=== Converting training data: {args.limo_path} ===")
        limo_grpo_path = output_dir / "limo_grpo.jsonl"
        limo_rgrpo_path = output_dir / "limo_rgrpo.jsonl"
        convert_limo_to_messages(args.limo_path, str(limo_grpo_path), str(limo_rgrpo_path))
    elif not args.skip_train:
        print(f"\nWarning: Training data not found at {args.limo_path}")
        print("Skipping training data conversion. Run with --skip_train to suppress this warning.")

    print("\n=== Data preparation complete! ===")
    print(f"  GRPO validation: {grpo_val_path}")
    print(f"  R-GRPO validation: {rgrpo_val_path}")
    if not args.skip_train and os.path.exists(args.limo_path):
        print(f"  GRPO training: {output_dir / 'limo_grpo.jsonl'}")
        print(f"  R-GRPO training: {output_dir / 'limo_rgrpo.jsonl'}")


if __name__ == "__main__":
    main()
