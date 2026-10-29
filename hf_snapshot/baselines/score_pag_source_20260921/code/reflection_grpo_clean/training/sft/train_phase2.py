"""
Phase 2 SFT: 在 Phase 1 基础上，训练模型基于反思文本续推得出正确答案。

混合三类样本防止灾难性遗忘：
  1. phase2_reflection : reflection_text（prefix）→ reasoning（loss）  ← 主目标
  2. phase1_replay     : 首错前缀（cut_point）→ reflection_text（loss）  ← 回放 Phase 1
  3. perfect           : 无 prefix → full_reasoning（loss）            ← 保持正常推理

Assistant 输出格式:
    {reflection_text}  <- prefix，不计 loss
    {reasoning}        <- 计 loss

运行:
    python train_phase2.py
"""

import argparse
import json
import os
import random
import re
import sys
import torch
from pathlib import Path

_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_IMPORT_ROOT))

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import PeftModel

from rgrpo.chat_template import apply_chat_template
from rgrpo.sft_training import (encode_supervised, causal_token_metrics, preprocess_token_logits,
    validate_split_groups, audit_datasets, load_single_device_model,
    EvaluateLastStep, save_checkpoint_selection)
from rgrpo.sft_data import build_paper_phase2_mix

# Paper recipe defaults. The resulting mix is approximately 64:19:17 for
# reflection-to-solution : Phase-1 replay : perfect samples.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "artifacts"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "models"))
MODEL_ID = os.environ.get("MODEL_ID", str(MODEL_ROOT / "Qwen2.5-Math-1.5B"))
PHASE1_CKPT = os.environ.get("PHASE1_CKPT", str(OUTPUT_ROOT / "sft" / "phase1" / "final"))
DATA_PATH = os.environ.get("DATA_PATH", str(DATA_ROOT / "sft" / "reasoning_outputs.jsonl"))
PERFECT_PATH = os.environ.get("PERFECT_PATH", str(DATA_ROOT / "sft" / "reflection_data_prepared_perfect.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(OUTPUT_ROOT / "sft" / "phase2"))
LOG_DIR = os.environ.get("LOG_DIR", str(OUTPUT_ROOT / "logs" / "sft_phase2"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
SEED = int(os.environ.get("SEED", "42"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1").lower() in {"1", "true", "yes"}
COMPLIANCE_SAMPLES = int(os.environ.get("COMPLIANCE_SAMPLES", "20"))
COMPLIANCE_MAX_NEW_TOKENS = int(os.environ.get("COMPLIANCE_MAX_NEW_TOKENS", "768"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
REPLAY_RATIO = float(os.environ.get("REPLAY_RATIO", "0.3"))
ENFORCE_PAPER_MIX = os.environ.get("ENFORCE_PAPER_MIX", "0").lower() in {
    "1", "true", "yes", "on"
}
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "2"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."
)


# ── Dataset ───────────────────────────────────────────────────────
class Phase2Dataset(Dataset):
    """
    三类样本：
      reflection      : prefix = header + reflection_text，target = reasoning
      phase1_replay   : prefix = header + 首错前缀，   target = reflection_text
      perfect         : prefix = header，                  target = full_reasoning
    """

    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        d   = self.records[idx]
        tok = self.tokenizer

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": d["problem"]},
        ]
        header_str = apply_chat_template(
            tok,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if d["type"] == "reflection":
            # Phase 2 主目标：给定反思文本，续写正确推理
            prefix_str = header_str + d["reflection_text"].rstrip() + "\n"
        elif d["type"] == "phase1_replay":
            # Phase 1 回放：给定cut_point中的错误前缀，生成反思文本（防灾难性遗忘）。
            # 本批数据的 cut_point 截到首错位置，包含该错误步骤。
            prefix_str = header_str + d["cut_point"].rstrip() + "\n"
        else:
            # perfect 样本：无 prefix mask，全部计 loss
            prefix_str = header_str

        target = (d["reasoning"] if d["type"] == "reflection" else
                  d["reflection_text"] if d["type"] == "phase1_replay" else d["full_reasoning"])
        return encode_supervised(tok, prefix_str, target, self.max_len)


# ── 数据加载（与 Phase 1 相同 seed，保证 train/eval 划分一致）───
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
        train_perf = [d for d in perfect if d["split"] == "train"]
        eval_perf = [d for d in perfect if d["split"] == "eval"]
    else:
        rng.seed(seed)
        rng.shuffle(reflection)
        rng.shuffle(perfect)
        r_split = int(len(reflection) * 0.9)
        p_split = int(len(perfect) * 0.9)
        train_refl, eval_refl = reflection[:r_split], reflection[r_split:]
        train_perf, eval_perf = perfect[:p_split], perfect[p_split:]

    validate_split_groups(train_refl + train_perf, eval_refl + eval_perf)

    if ENFORCE_PAPER_MIX:
        train = build_paper_phase2_mix(train_refl, train_perf, seed=seed)
        eval_ = build_paper_phase2_mix(eval_refl, eval_perf, seed=seed + 1)
        print("[Data] Enforcing paper Phase-2 mix 64:19:17 by deterministic resampling.")
        print(f"[Data] Split source: {'record field' if explicit_split else 'seeded 90/10 fallback'}")
        print(f"[Data] Train mix: {dict((t, sum(d['type'] == t for d in train)) for t in ('reflection', 'phase1_replay', 'perfect'))}")
        print(f"[Data] Eval mix:  {dict((t, sum(d['type'] == t for d in eval_)) for t in ('reflection', 'phase1_replay', 'perfect'))}")
        return train, eval_

    # Phase 1 回放样本：复用同一批 records，仅改 type，从 train/eval 各自采样
    n_replay_train = int(len(train_refl) * REPLAY_RATIO)
    n_replay_eval  = int(len(eval_refl)  * REPLAY_RATIO)

    train_replay_pool = [dict(d, type="phase1_replay") for d in train_refl]
    eval_replay_pool  = [dict(d, type="phase1_replay") for d in eval_refl]
    rng.shuffle(train_replay_pool)
    rng.shuffle(eval_replay_pool)
    train_replay = train_replay_pool[:n_replay_train]
    eval_replay  = eval_replay_pool[:n_replay_eval]

    train = train_refl + train_perf + train_replay
    eval_ = eval_refl  + eval_perf  + eval_replay

    rng.shuffle(train)
    rng.shuffle(eval_)

    print(f"[Data] Reflection: {len(reflection)} | Perfect: {len(perfect)}")
    print(f"[Data] Train: {len(train)} "
          f"(p2_refl={len(train_refl)}, perfect={len(train_perf)}, p1_replay={len(train_replay)})")
    print(f"[Data] Eval:  {len(eval_)} "
          f"(p2_refl={len(eval_refl)}, perfect={len(eval_perf)}, p1_replay={len(eval_replay)})")
    return train, eval_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate every train/eval sample without loading model weights.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


# ── Phase 1 合规率回调 ────────────────────────────────────────────
class ComplianceCallback(TrainerCallback):
    """
    每隔 EVAL_STEPS 步在 eval_refl 上抽样检查 Phase 1 合规率：
    给定cut_point中的错误前缀（存于 cut_point 兼容字段），模型能否仍然生成完整
    <reflection>...</reflection>。
    结果写入 tensorboard（log_history）和日志文件。
    """

    def __init__(self, eval_refl, tokenizer, log_path, n=50, every_steps=200):
        self.eval_refl  = eval_refl
        self.tokenizer  = tokenizer
        self.log_path   = log_path
        self.n          = n
        self.every_steps = every_steps

    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.global_step % self.every_steps != 0:
            return

        model.eval()
        samples = random.Random(SEED).sample(self.eval_refl, min(self.n, len(self.eval_refl)))
        if not samples:
            model.train()
            return

        compliant = 0
        for d in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": d["problem"]},
            ]
            input_str = apply_chat_template(
                self.tokenizer,
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_str += d["cut_point"].rstrip() + "\n"

            inputs = self.tokenizer(
                input_str, return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=COMPLIANCE_MAX_NEW_TOKENS,
                    do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if re.search(r"<reflection>[\s\S]+?</reflection>", generated):
                compliant += 1

        rate = compliant / len(samples)
        msg = f"[Step {state.global_step}] p1_compliance: {compliant}/{len(samples)} = {rate:.2%}"
        print(f"\n{msg}")
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

        # 写入 trainer log_history 供 tensorboard 读取
        state.log_history.append({
            "step": state.global_step,
            "p1_compliance": rate,
        })
        model.train()


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    cli_args = parse_args()
    from transformers import set_seed
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 数据
    train_raw, eval_raw = load_data(DATA_PATH, PERFECT_PATH)
    train_ds = Phase2Dataset(train_raw, tokenizer)
    eval_ds  = Phase2Dataset(eval_raw,  tokenizer)

    audit_datasets({"train": train_ds, "eval": eval_ds}, Path(LOG_DIR) / "data_preflight.json")
    if cli_args.dry_run:
        return

    # 3. 加载 Phase 1 的 LoRA 权重继续训练
    from peft import PeftConfig
    adapter_cfg = PeftConfig.from_pretrained(PHASE1_CKPT)
    if Path(adapter_cfg.base_model_name_or_path).resolve() != Path(MODEL_ID).resolve():
        raise ValueError("Phase 1 adapter belongs to a different base model")
    base_model = load_single_device_model(MODEL_ID, ATTN_IMPLEMENTATION)
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model,
        PHASE1_CKPT,
        is_trainable=True,   # 继续训练，不 freeze
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 4. 训练参数（lr 比 Phase 1 更小，防止覆盖已学到的反思格式）
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,     # Phase 1 的 1/4
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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

    # 5. Trainer（含 token accuracy + Phase 1 合规率回调）

    eval_refl = [d for d in eval_raw if d["type"] == "reflection"]
    compliance_cb = ComplianceCallback(
        eval_refl   = eval_refl,
        tokenizer   = tokenizer,
        log_path    = Path(LOG_DIR) / "compliance.log",
        n           = COMPLIANCE_SAMPLES,
        every_steps = EVAL_STEPS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=causal_token_metrics,
        preprocess_logits_for_metrics=preprocess_token_logits,
        callbacks=[compliance_cb, EvaluateLastStep()],
    )

    trainer.train(resume_from_checkpoint=cli_args.resume_from_checkpoint)
    save_checkpoint_selection(trainer, OUTPUT_DIR)

    # 6. 保存
    save_path = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[Done] Phase 2 saved → {save_path}")


if __name__ == "__main__":
    main()
