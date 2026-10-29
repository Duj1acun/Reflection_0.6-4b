"""
eval_pass4.py — Pass@4 / Avg@4 evaluation (vLLM backend)

Evaluates baseline / phase2 / phase2_v2 with 4 independent samples per problem.
Each sample goes through the same reflective inference loop as eval_accuracy.py.

Metrics:
  pass@4   — fraction of problems where AT LEAST ONE sample is correct
  avg@4    — average accuracy across all 4 samples (= expected accuracy of a random pick)

Usage:
    # GPU1: baseline + phase2
    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python -u eval_pass4.py \\
        --models baseline,phase2 > run_pass4_gpu1.log 2>&1 &

    # GPU2: phase2_v2
    CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID python -u eval_pass4.py \\
        --models phase2_v2 > run_pass4_gpu2.log 2>&1 &

    # After both finish:
    python combine_pass4.py
"""

import argparse
import json
import re
import math
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
    "baseline":  None,
    "phase2":    "../reasoning_paths/outputs/phase2/final",
    "phase2_v2": "../reasoning_paths/outputs/phase2_v2/final",
}
MODEL_ORDER = list(CHECKPOINTS.keys())

N_SAMPLES        = 4
TEMPERATURE      = 0.7
MAX_REFLECTIONS  = 3
MAX_NEW_PER_ROUND = 2048
MAX_NEW_BASELINE  = 2048
MAX_MODEL_LEN     = 4096

SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection. "
    "Always present your final answer in \\boxed{}."
)


# ── Math answer scoring ───────────────────────────────────────────────────────
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
    print("[vLLM] Loading engine...")
    engine = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_lora_rank=8,
        max_cpu_loras=3,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        num_gpu_blocks_override=80000,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    print("[vLLM] Engine ready.\n")
    return engine, tokenizer


def vllm_generate(engine, prompts, max_new_tokens, temperature, lora_request=None):
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = engine.generate(prompts, params, lora_request=lora_request)
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


# ── Reflective inference — batched, handles N_SAMPLES copies per problem ──────
def reflective_evaluate_pass4(engine, tokenizer, data, ckpt_path):
    """
    Each problem is evaluated N_SAMPLES times independently.
    Internally, the data is expanded to len(data) * N_SAMPLES entries.
    Returns a list of dicts, one per original problem, with per-sample results.
    """
    is_sft        = (ckpt_path is not None)
    lora_request  = LoRARequest("adapter", 1, ckpt_path) if is_sft else None
    max_new       = MAX_NEW_PER_ROUND if is_sft else MAX_NEW_BASELINE
    temperature   = TEMPERATURE

    # Expand: each problem → N_SAMPLES independent trajectories
    # flat_idx -> (orig_idx, sample_idx)
    flat_data = []
    for orig_idx, rec in enumerate(data):
        for s in range(N_SAMPLES):
            flat_data.append((orig_idx, s, rec))

    n             = len(flat_data)
    headers       = [build_header(tokenizer, rec["problem"]) for _, _, rec in flat_data]
    contexts      = list(headers)
    num_refls     = [0] * n
    all_rounds    = [[] for _ in range(n)]
    final_outputs = [""] * n
    active        = list(range(n))

    for round_idx in range(MAX_REFLECTIONS + 1):
        if not active:
            break

        prompts = [contexts[i] for i in active]
        outputs = vllm_generate(engine, prompts, max_new, temperature, lora_request)

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
                contexts[i]    = headers[i] + reflection_tag.rstrip() + "\n"
                next_active.append(i)

        active = next_active
        print(f"  round {round_idx} done | remaining={len(active)} | "
              f"total_reflections={sum(num_refls)}", flush=True)

    # Aggregate by original problem index
    # per_problem[orig_idx] = list of N_SAMPLES sample results
    per_problem = defaultdict(list)
    for flat_i, (orig_idx, sample_idx, rec) in enumerate(flat_data):
        correct = compute_score(final_outputs[flat_i], rec["answer"]) == 1.0
        per_problem[orig_idx].append({
            "sample_idx":      sample_idx,
            "predicted":       extract_boxed(final_outputs[flat_i]),
            "correct":         correct,
            "num_reflections": num_refls[flat_i],
            "rounds":          all_rounds[flat_i],
        })

    results = []
    for orig_idx, rec in enumerate(data):
        samples    = per_problem[orig_idx]
        any_correct = any(s["correct"] for s in samples)
        avg_correct = sum(s["correct"] for s in samples) / len(samples)
        results.append({
            "subject":      rec.get("subject", ""),
            "level":        str(rec.get("level", "")),
            "ground_truth": rec["answer"],
            "pass4":        any_correct,
            "avg4":         avg_correct,
            "samples":      samples,
        })
    return results


# ── Summary ───────────────────────────────────────────────────────────────────
def print_model_summary(tag, results):
    n         = len(results)
    pass4     = sum(r["pass4"] for r in results) / n
    avg4      = sum(r["avg4"]  for r in results) / n
    tot_refl  = sum(s["num_reflections"] for r in results for s in r["samples"])
    avg_refl  = tot_refl / (n * N_SAMPLES)
    print(f"  pass@4={pass4:.2%}  avg@4={avg4:.2%}  "
          f"total_reflections={tot_refl}  avg_refl={avg_refl:.3f}")
    return pass4, avg4, tot_refl, avg_refl


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(MODEL_ORDER))
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASETS.keys()))
    parser.add_argument("--test_n", type=int, default=0)
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip() in CHECKPOINTS]
    if not models_to_run:
        print(f"No valid models. Choose from: {list(CHECKPOINTS.keys())}")
        return

    data = []
    with open(DATASETS[args.dataset]) as f:
        for line in f:
            data.append(json.loads(line))
    if args.test_n > 0:
        data = data[:args.test_n]
    print(f"[Dataset] {args.dataset} — {len(data)} problems × {N_SAMPLES} samples "
          f"= {len(data) * N_SAMPLES} total")
    print(f"[Models]  {models_to_run}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine, tokenizer = build_engine()

    for tag in models_to_run:
        ckpt_path = CHECKPOINTS[tag]
        print("=" * 60)
        print(f"[Model] {tag}  {'(baseline)' if not ckpt_path else ckpt_path}")

        results = reflective_evaluate_pass4(engine, tokenizer, data, ckpt_path)
        print_model_summary(tag, results)

        out_file = OUTPUT_DIR / f"pass4_{args.dataset}_{tag}.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_file}\n")

    print("All done. Run `python combine_pass4.py` to merge results.")


if __name__ == "__main__":
    main()
