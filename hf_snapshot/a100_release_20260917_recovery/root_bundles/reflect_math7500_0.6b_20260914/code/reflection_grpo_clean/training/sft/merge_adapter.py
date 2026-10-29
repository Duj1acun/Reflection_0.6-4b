"""
Merge SFT LoRA adapter into base model.

Defaults are repository-relative and may be overridden with command-line arguments.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_model_path: str, adapter_path: str, output_path: str):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading base model from {base_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"[2/4] Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("[3/4] Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"[4/4] Saving merged model to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer from adapter dir (it has the updated special tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Done. Merged model saved.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=str(project_root / "models" / "Qwen2.5-Math-1.5B"))
    parser.add_argument(
        "--adapter",
        default=str(project_root / "artifacts" / "sft" / "phase2" / "final"),
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "artifacts" / "sft" / "phase2_merged_hf"),
    )
    args = parser.parse_args()
    merge(args.base_model, args.adapter, args.output)
