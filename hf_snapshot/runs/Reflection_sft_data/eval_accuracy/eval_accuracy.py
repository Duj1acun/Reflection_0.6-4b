"""
eval_accuracy.py — Reflective Inference Accuracy Evaluation (vLLM backend)

Evaluates 5 models on LIMO-clean (757) or MATH500 (500):
  1. baseline          — Qwen2.5-Math-1.5B, no LoRA, one-shot
  2. phase1_ckpt1600   — LoRA, reflective loop ≤3 rounds
  3. phase_merged      — LoRA, reflective loop ≤3 rounds
  4. phase2            — LoRA, reflective loop ≤3 rounds
  5. phase2_v2         — LoRA, reflective loop ≤3 rounds

Speed: vLLM engine loaded ONCE per process; LoRA adapters hot-swapped.
       Two processes on GPU1/GPU2 share the work.

Reflective loop:
  - Generate; detect complete <reflection>...</reflection>
  - On reflection: discard bad CoT, rebuild context =
        header + <reflection>...</reflection>   (clean slate)
  - Up to MAX_REFLECTIONS rounds; every round saved for inspection

Usage:
    cd /home/xiaofeng/wrx_project/reasoning_paths_directional/eval_accuracy

    # GPU1: baseline + phase1_ckpt1600 + phase_merged
    CUDA_VISIBLE_DEVICES=1 python -u eval_accuracy.py \\
        --models baseline,phase1_ckpt1600,phase_merged > run_gpu1.log 2>&1 &

    # GPU2: phase2 + phase2_v2
    CUDA_VISIBLE_DEVICES=2 python -u eval_accuracy.py \\
        --models phase2,phase2_v2 > run_gpu2.log 2>&1 &

    # After both finish:
    python combine_summary.py

    # Smoke test (5 samples, one model):
    CUDA_VISIBLE_DEVICES=1 python -u eval_accuracy.py --models phase_merged --test_n 5
"""

import argparse
import json
import re
import torch
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "/hd1/models/Qwen2.5-Math-1.5B"

DATASETS = {
    "limo":    "limo_clean.jsonl",
    "math500": "/hd1/datasets/math500/test.jsonl",
}
DEFAULT_DATASET = "limo"

OUTPUT_DIR = Path(".")

CHECKPOINTS = {
    "baseline":        None,
    "phase1_ckpt1600": "../reasoning_paths/outputs/phase1/checkpoint-1600",
    "phase_merged":    "../reasoning_paths/outputs/phase_merged/final",
    "phase2":          "../reasoning_paths/outputs/phase2/final",
    "phase2_v2":       "../reasoning_paths/outputs/phase2_v2/final",
}
MODEL_ORDER = list(CHECKPOINTS.keys())

MAX_REFLECTIONS   = 3
MAX_NEW_PER_ROUND = 2048
MAX_NEW_BASELINE  = 2048
MAX_MODEL_LEN     = 4096   # vLLM context window

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection. "
    "Always present your final answer in \\boxed{}."
)


# ── Math answer scoring (hendrycks_math normalization) ────────────────────────
def _fix_fracs(s):
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p and p[0] == "{":
            out += p
        elif len(p) >= 2:
            a, b, rest = p[0], p[1], p[2:]
            out += ("{" + a + "}{" + b + "}" + rest) if b != "{" else ("{" + a + "}" + b + rest)
        else:
            out += p
    return out


def _fix_sqrt(s):
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        out += ("\\sqrt{" + p[0] + "}" + p[1:]) if (p and p[0] != "{") else ("\\sqrt" + p)
    return out


def _fix_slash(s):
    sp = s.split("/")
    if len(sp) != 2:
        return s
    try:
        a, b = int(sp[0]), int(sp[1])
        assert s == f"{a}/{b}"
        return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        return s


def _strip(s):
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    for t in ("tfrac", "dfrac"):
        s = s.replace(t, "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "")
    if "\\text{ " in s:
        s = s.split("\\text{ ")[0]
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    return _fix_slash(s)


def is_equiv(s1, s2):
    if not s1 or not s2:
        return False
    try:
        return _strip(s1) == _strip(s2)
    except Exception:
        return s1.strip() == s2.strip()


def extract_boxed(text):
    """Extract content of the LAST \\boxed{...} in text. Returns None if not found."""
    if "\\boxed " in text:
        val = text.split("\\boxed ")[-1].split("$")[0].strip()
        if val:
            return val
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth, right = idx + len("\\boxed{") - 1, 0, None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return text[idx + len("\\boxed{"):right] if right is not None else None


def compute_score(generated, ground_truth):
    try:
        pred = extract_boxed(generated)
        if pred is not None and is_equiv(pred, ground_truth):
            return 1.0
    except Exception:
        pass
    return 0.0


# ── vLLM engine ───────────────────────────────────────────────────────────────
def build_engine():
    """Load vLLM engine once; LoRA adapters are swapped per model."""
    print("[vLLM] Loading engine...")
    engine = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_lora_rank=8,
        max_cpu_loras=5,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,           # skip torch.compile to avoid memory fragmentation
        num_gpu_blocks_override=80000, # workaround: vLLM V1 auto-detection returns 0 blocks
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    print("[vLLM] Engine ready.\n")
    return engine, tokenizer


def vllm_generate(engine, prompts, max_new_tokens, lora_request=None):
    """Batch generate with vLLM; returns list of generated strings."""
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
    )
    outputs = engine.generate(prompts, params, lora_request=lora_request)
    # vLLM returns results in request order
    return [o.outputs[0].text for o in outputs]


# ── Prompt building ───────────────────────────────────────────────────────────
def build_header(tokenizer, problem):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Reflective inference (batched, vLLM) ─────────────────────────────────────
def reflective_evaluate(engine, tokenizer, data, ckpt_path):
    """
    Evaluate all samples with batched reflective inference.

    is_sft = (ckpt_path is not None)
    LoRA request is built from ckpt_path for SFT models; None for baseline.

    After each reflection:
        context[i] = header[i] + <reflection>...</reflection>
        (bad CoT before the reflection tag is DISCARDED)
    """
    is_sft      = (ckpt_path is not None)
    lora_request = LoRARequest("adapter", 1, ckpt_path) if is_sft else None
    max_new      = MAX_NEW_PER_ROUND if is_sft else MAX_NEW_BASELINE

    n             = len(data)
    headers       = [build_header(tokenizer, rec["problem"]) for rec in data]
    contexts      = list(headers)
    num_refls     = [0] * n
    all_rounds    = [[] for _ in range(n)]
    final_outputs = [""] * n
    active        = list(range(n))

    for round_idx in range(MAX_REFLECTIONS + 1):
        if not active:
            break

        prompts = [contexts[i] for i in active]
        outputs = vllm_generate(engine, prompts, max_new, lora_request)

        next_active = []
        for j, i in enumerate(active):
            output           = outputs[j]
            final_outputs[i] = output
            refl_match       = re.search(r"<reflection>[\s\S]+?</reflection>", output)
            has_refl         = refl_match is not None

            all_rounds[i].append({
                "round":          round_idx,
                "context_type":   "initial" if round_idx == 0 else "post_reflection",
                "output":         output,
                "has_reflection": has_refl,
            })

            if is_sft and has_refl and num_refls[i] < MAX_REFLECTIONS:
                reflection_tag = refl_match.group(0)
                num_refls[i]  += 1
                # ── Discard bad CoT; only keep header + reflection tag ──
                contexts[i]    = headers[i] + reflection_tag.rstrip() + "\n"
                next_active.append(i)

        active = next_active
        print(f"  round {round_idx} done | remaining={len(active)} | "
              f"total_reflections={sum(num_refls)}", flush=True)

    results = []
    for i, rec in enumerate(data):
        correct = compute_score(final_outputs[i], rec["answer"]) == 1.0
        results.append({
            "subject":         rec.get("subject", ""),
            "level":           str(rec.get("level", "")),
            "ground_truth":    rec["answer"],
            "predicted":       extract_boxed(final_outputs[i]),
            "correct":         correct,
            "num_reflections": num_refls[i],
            "rounds":          all_rounds[i],
        })
    return results


# ── Summary ───────────────────────────────────────────────────────────────────
def print_model_summary(tag, results):
    n         = len(results)
    acc       = sum(r["correct"] for r in results) / n
    tot_refl  = sum(r["num_reflections"] for r in results)
    avg_refl  = tot_refl / n
    refl_rate = sum(1 for r in results if r["num_reflections"] > 0) / n
    print(f"  acc={acc:.2%}  total_reflections={tot_refl}  "
          f"avg_refl={avg_refl:.3f}  refl_rate={refl_rate:.1%}")
    return acc, tot_refl, avg_refl, refl_rate


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(MODEL_ORDER),
                        help="Comma-separated model tags to evaluate in order")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASETS.keys()))
    parser.add_argument("--test_n", type=int, default=0,
                        help="Evaluate only first N samples (0 = all)")
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip() in CHECKPOINTS]
    if not models_to_run:
        print(f"No valid models. Choose from: {list(CHECKPOINTS.keys())}")
        return

    # Load dataset
    data = []
    with open(DATASETS[args.dataset]) as f:
        for line in f:
            data.append(json.loads(line))
    if args.test_n > 0:
        data = data[:args.test_n]
    print(f"[Dataset] {args.dataset} — {len(data)} samples")
    print(f"[Models]  {models_to_run}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load vLLM engine ONCE — all models share it
    engine, tokenizer = build_engine()

    for tag in models_to_run:
        ckpt_path = CHECKPOINTS[tag]
        print("=" * 60)
        print(f"[Model] {tag}  {'(baseline)' if not ckpt_path else ckpt_path}")

        results = reflective_evaluate(engine, tokenizer, data, ckpt_path)
        print_model_summary(tag, results)

        out_file = OUTPUT_DIR / f"results_{args.dataset}_{tag}.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_file}\n")

    print("All done. Run `python combine_summary.py` to merge results.")


if __name__ == "__main__":
    main()
