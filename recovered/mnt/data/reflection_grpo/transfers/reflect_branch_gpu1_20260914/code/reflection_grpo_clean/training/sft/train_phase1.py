"""
Phase-1 LoRA SFT for structured reflection generation.

The defaults implement the paper recipe for Qwen2.5-Math-1.5B: LoRA
r=16/alpha=32/dropout=0.05, three epochs, and a 2e-4 learning rate. Paths and
hyperparameters can be overridden with environment variables; ``--dry-run``
validates data/model setup without starting training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_IMPORT_ROOT))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from rgrpo.chat_template import apply_chat_template
from rgrpo.sft_training import (encode_supervised, causal_token_metrics, preprocess_token_logits,
    validate_split_groups, audit_datasets, load_single_device_model,
    EvaluateLastStep, save_checkpoint_selection)

# Paper recipe defaults. Paths may be overridden with environment variables.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "artifacts"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "models"))
MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(MODEL_ROOT / "Qwen2.5-Math-1.5B"),
)
DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(DATA_ROOT / "sft" / "reasoning_outputs.jsonl"),
)
PERFECT_PATH = os.environ.get(
    "PERFECT_PATH",
    str(DATA_ROOT / "sft" / "reflection_data_prepared_perfect.jsonl"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(OUTPUT_ROOT / "sft" / "phase1"))
LOG_DIR = os.environ.get("LOG_DIR", str(OUTPUT_ROOT / "logs" / "sft_phase1"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
SEED = int(os.environ.get("SEED", "42"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1").lower() in {"1", "true", "yes"}
COMPLIANCE_SAMPLES = int(os.environ.get("COMPLIANCE_SAMPLES", "20"))
COMPLIANCE_MAX_NEW_TOKENS = int(os.environ.get("COMPLIANCE_MAX_NEW_TOKENS", "768"))
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "3"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "100"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.05"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
# LoRA / data-mix knobs (env-overridable; defaults reproduce the original run)
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_TARGETS = os.environ.get("LORA_TARGETS", "all")  # "all" | "attn"
PERFECT_OVERSAMPLE = int(os.environ.get("PERFECT_OVERSAMPLE", "1"))

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."
)


# ── Dataset（与 train_phase1.py 完全一致）──────────────────────────
class Phase1Dataset(Dataset):
    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        d = self.records[idx]
        tok = self.tokenizer

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": d["problem"]},
        ]
        header_str = apply_chat_template(
            tok,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if d["type"] == "reflection":
            prefix_str = header_str + d["cut_point"].rstrip() + "\n"
        else:
            prefix_str = header_str

        target = d["reflection_text"] if d["type"] == "reflection" else d["full_reasoning"]
        return encode_supervised(tok, prefix_str, target, self.max_len)


# ── 格式合规率评测（与 train_phase1.py 完全一致）──────────────────
def eval_format_compliance(model, tokenizer, eval_records, n=100, max_new_tokens=300):
    model.eval()
    pool = [d for d in eval_records if d.get("type") == "reflection"]
    samples = random.Random(SEED).sample(pool, min(n, len(pool)))

    reflection_samples = [d for d in samples if d.get("type") == "reflection"]
    if not reflection_samples:
        print("\n[Eval] No reflection samples in eval set, skipping compliance check.")
        model.train()
        return 0.0

    compliant = 0
    for d in reflection_samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": d["problem"]},
        ]
        input_str = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_str += d["cut_point"].rstrip() + "\n"

        inputs = tokenizer(input_str, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)

        if re.search(r"<reflection>[\s\S]+?</reflection>", generated):
            compliant += 1

    rate = compliant / len(reflection_samples)
    print(f"\n[Eval] Format compliance: {compliant}/{len(reflection_samples)} = {rate:.2%}")
    model.train()
    return rate


# ── 数据加载（与 train_phase1.py 完全一致）────────────────────────
def load_data(reflection_path, perfect_path, seed=SEED):
    rng = random.Random(seed)
    reflection = []
    with open(reflection_path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("cut_point", "").strip(): continue
            if not d.get("reflection_text", "").strip(): continue
            if not d.get("reasoning", "").strip(): continue
            d["type"] = "reflection"
            reflection.append(d)

    perfect = []
    with open(perfect_path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("full_reasoning", "").strip(): continue
            d["type"] = "perfect"
            perfect.append(d)

    explicit_split = all(
        d.get("split") in {"train", "eval"} for d in reflection + perfect
    )
    if explicit_split:
        train_refl = [d for d in reflection if d["split"] == "train"]
        eval_refl = [d for d in reflection if d["split"] == "eval"]
        train_perf_raw = [d for d in perfect if d["split"] == "train"]
        eval_perf = [d for d in perfect if d["split"] == "eval"]
    else:
        rng.seed(seed)
        rng.shuffle(reflection)
        rng.shuffle(perfect)
        r_split = int(len(reflection) * 0.9)
        p_split = int(len(perfect) * 0.9)
        train_refl, eval_refl = reflection[:r_split], reflection[r_split:]
        train_perf_raw, eval_perf = perfect[:p_split], perfect[p_split:]

    validate_split_groups(train_refl + train_perf_raw, eval_refl + eval_perf)
    train_perf = train_perf_raw * PERFECT_OVERSAMPLE
    train = train_refl + train_perf
    eval_ = eval_refl + eval_perf

    rng.shuffle(train)
    rng.shuffle(eval_)

    print(f"[Data] Reflection: {len(reflection)} | Perfect: {len(perfect)} "
          f"(oversample x{PERFECT_OVERSAMPLE})")
    print(f"[Data] Split source: {'record field' if explicit_split else 'seeded 90/10 fallback'}")
    print(f"[Data] Train: {len(train)} (refl {len(train_refl)} : perf {len(train_perf)}) "
          f"| Eval: {len(eval_)}")
    return train, eval_


def load_base_model():
    return load_single_device_model(MODEL_ID, ATTN_IMPLEMENTATION)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate every train/eval sample without loading model weights.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Trainer checkpoint to resume from.")
    return parser.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    cli_args = parse_args()
    from transformers import set_seed
    set_seed(SEED)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[Config] MODEL_ID={MODEL_ID}")
    print(f"[Config] OUTPUT_DIR={OUTPUT_DIR}  LOG_DIR={LOG_DIR}")
    print(f"[Config] EPOCHS={NUM_TRAIN_EPOCHS} SAVE_STEPS={SAVE_STEPS} "
          f"EVAL_STEPS={EVAL_STEPS} LR={LEARNING_RATE} (save_total_limit=3, load_best=True)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_raw, eval_raw = load_data(DATA_PATH, PERFECT_PATH)
    train_ds = Phase1Dataset(train_raw, tokenizer)
    eval_ds = Phase1Dataset(eval_raw, tokenizer)

    audit_datasets({"train": train_ds, "eval": eval_ds}, Path(LOG_DIR) / "data_preflight.json")
    if cli_args.dry_run:
        return

    model = load_base_model()
    model.config.use_cache = False
    model.enable_input_require_grads()

    attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_targets = ["gate_proj", "up_proj", "down_proj"]
    lora_targets = attn_targets if LORA_TARGETS == "attn" else attn_targets + mlp_targets
    print(f"[LoRA] r={LORA_R} alpha={LORA_ALPHA} targets={LORA_TARGETS} ({lora_targets})")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        logging_dir=LOG_DIR,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=causal_token_metrics,
        preprocess_logits_for_metrics=preprocess_token_logits,
        callbacks=[EvaluateLastStep()],
    )

    trainer.train(resume_from_checkpoint=cli_args.resume_from_checkpoint)
    save_checkpoint_selection(trainer, OUTPUT_DIR)

    eval_format_compliance(model, tokenizer, eval_raw, n=COMPLIANCE_SAMPLES, max_new_tokens=COMPLIANCE_MAX_NEW_TOKENS)

    save_path = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[Done] Phase 1 saved → {save_path}  "
          f"(up to three checkpoints under {OUTPUT_DIR})")


if __name__ == "__main__":
    main()
