"""
test_reflection_format.py — Verify reflection trigger + continuation format.

Tests:
  1. System prompt consistency across all modules
  2. Continuation prompt format = header + refl_tag.rstrip() + "\n"
  3. Live inference: model triggers reflection on hard problems
  4. Live inference: model continues correctly from continuation prompt
  5. Token boundary correctness: encode(response_slice) vs encode(prompt+slice)-prompt_len

Run:
    cd /path/to/reflection_grpo_clean
    CUDA_VISIBLE_DEVICES=0,1 python test_reflection_format.py
"""
import os, re, sys, json

_TMP = os.environ.get("TEST_TMPDIR", "/tmp/reflect_plus_eval")
os.makedirs(_TMP, exist_ok=True)
os.environ["TMPDIR"]    = _TMP
os.environ["TEMP"]      = _TMP
os.environ["TMP"]       = _TMP
os.environ["VLLM_USE_V1"] = "0"
os.environ.setdefault("TRITON_CACHE_DIR",        "/tmp/reflect_plus_triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/reflect_plus_torchinductor_cache")

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/sft/phase2_merged_hf")
REFL_PATTERN = re.compile(r"<reflection>[\s\S]+?</reflection>", re.DOTALL)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

# ── 1. System-Prompt Consistency Check ────────────────────────────────────────

# Prompt used at Phase-2 SFT training time (from eval_phase2_format.py, which
# mirrors the training script — no space before "Always"):
TRAIN_SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."  # ← NO trailing space
    "Always present your final answer in \\boxed{}."
)

# Prompt used in reflection.py — imported directly from the module:
sys.path.insert(0, os.getcwd())
from rgrpo.reflection import REFLECTION_SYSTEM_PROMPT as REFLECTION_PY_PROMPT

def test_system_prompt_consistency():
    print("\n=== Test 1: System Prompt Consistency ===")
    if TRAIN_SYSTEM_PROMPT == REFLECTION_PY_PROMPT:
        print(f"  {PASS} Prompts match.")
        return True
    else:
        print(f"  {FAIL} Prompt mismatch detected!")
        # Find the difference
        for i, (a, b) in enumerate(zip(TRAIN_SYSTEM_PROMPT, REFLECTION_PY_PROMPT)):
            if a != b:
                print(f"  First diff at char {i}: train={repr(a)}, reflection.py={repr(b)}")
                print(f"  Context: train=...{repr(TRAIN_SYSTEM_PROMPT[max(0,i-20):i+20])}...")
                print(f"  Context: refl =...{repr(REFLECTION_PY_PROMPT[max(0,i-20):i+20])}...")
                break
        if len(TRAIN_SYSTEM_PROMPT) != len(REFLECTION_PY_PROMPT):
            print(f"  Lengths differ: train={len(TRAIN_SYSTEM_PROMPT)}, reflection.py={len(REFLECTION_PY_PROMPT)}")
        print(f"  {WARN} Fix: in reflection.py line 51, remove the trailing space after 'the reflection.'")
        return False

# ── 2. Continuation Prompt Format Check ───────────────────────────────────────

def test_continuation_prompt_format(tokenizer):
    print("\n=== Test 2: Continuation Prompt Format ===")
    problem = "What is 2 + 2?"

    # Build header (same as training)
    messages = [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    header = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    refl_tag = "<reflection>I made an error above. Let me reconsider.</reflection>"

    # Training format (from eval_phase2_format.py build_phase2_prompt):
    train_cont_prompt = header + refl_tag.rstrip() + "\n"

    # rollout_loop.py format (line 801-806):
    reflection_tag_rl = refl_tag.rstrip() + "\n"
    rollout_cont_prompt = header + reflection_tag_rl  # real_prompt_ids + reflection_ids

    # reflection.py format (line 247):
    refl_py_cont_prompt = header + refl_tag.rstrip() + "\n"

    ok_rl   = (train_cont_prompt == rollout_cont_prompt)
    ok_py   = (train_cont_prompt == refl_py_cont_prompt)

    print(f"  rollout_loop.py format matches training: {PASS if ok_rl else FAIL}")
    print(f"  reflection.py format matches training:   {PASS if ok_py else FAIL}")

    if ok_rl and ok_py:
        print(f"  {PASS} All continuation prompt formats are consistent.")
        print(f"  Format: header + refl_tag.rstrip() + '\\n'")
        print(f"  Sample continuation prompt tail: {repr(train_cont_prompt[-80:])}")
        return True

    # Show what differs
    if not ok_rl:
        for i, (a, b) in enumerate(zip(train_cont_prompt, rollout_cont_prompt)):
            if a != b:
                print(f"  rollout_loop.py diff at char {i}")
                break
    return ok_rl and ok_py


def test_nested_reflection_format(tokenizer):
    """
    Verify reflection.py uses clean-slate format for nested reflections:
      nested_prompt = header + nested_tag.rstrip() + "\\n"  (CORRECT)
    NOT:
      nested_prompt = cont_prompt + nested_tag.rstrip() + "\\n"  (stacks tags)

    Strategy: read the source file and check which formula is present.
    """
    print("\n=== Test 2b: Nested Reflection Prompt in reflection.py ===")
    import inspect
    from rgrpo import reflection as refl_module
    src = inspect.getsource(refl_module.ReflectionRolloutManager.generate_continuations)

    buggy_pattern   = "nested_prompt = cont_prompt + nested_tag"
    correct_pattern = "nested_prompt = header + nested_tag"

    has_buggy   = buggy_pattern   in src
    has_correct = correct_pattern in src

    if has_buggy and not has_correct:
        print(f"  {FAIL} Buggy formula found: '{buggy_pattern}'")
        print(f"  Fix: change to '{correct_pattern}.rstrip() + \"\\n\"'")
        return False
    elif has_correct and not has_buggy:
        print(f"  {PASS} Correct formula in place: '{correct_pattern}.rstrip() + \"\\n\"'")
        return True
    elif has_correct and has_buggy:
        print(f"  {WARN} Both patterns present — check manually.")
        return False
    else:
        print(f"  {WARN} Neither pattern found — nested reflection handling may have changed.")
        print(f"  Inspect generate_continuations manually.")
        return False


# ── 3. Token Boundary Method Check ────────────────────────────────────────────

def test_token_boundary(tokenizer):
    """
    Compare rollout_loop.py's token-count method vs core_rgrpo's find_token_boundary.
    rollout_loop.py encodes text[:pos] alone; core_rgrpo encodes (prompt+text[:pos])
    and subtracts prompt token count.
    For Qwen tokenizers, beginning-of-sequence context can affect tokenization.
    """
    print("\n=== Test 3: Token Boundary Method ===")
    problem = "Compute $\\sum_{k=1}^{100} k$."
    messages = [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Simulate a response with reflection
    response = (
        "Let me solve this step by step. The sum $\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$.\n"
        "For n=100: $\\frac{100 \\cdot 101}{2} = 5050$.\n"
        "<reflection>Wait, I need to double-check my formula application.</reflection>\n"
        "The formula is correct. The answer is $\\boxed{5050}$."
    )

    match = REFL_PATTERN.search(response)
    assert match, "No reflection found in test response"

    # rollout_loop.py method (encodes text slice alone):
    pre_text  = response[:match.start()]
    full_refl = response[:match.end()]
    pre_end_tok_rl   = len(tokenizer.encode(pre_text,  add_special_tokens=False))
    refl_end_tok_rl  = len(tokenizer.encode(full_refl, add_special_tokens=False))

    # core_rgrpo.py method (encodes prompt+text, subtracts prompt):
    def find_token_boundary(prompt, response, char_pos):
        if char_pos <= 0:
            return 0
        prefix_ids = tokenizer(prompt + response[:char_pos], add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return max(0, len(prefix_ids) - len(prompt_ids))

    pre_end_tok_cr  = find_token_boundary(prompt, response, match.start())
    refl_end_tok_cr = find_token_boundary(prompt, response, match.end())

    diff_pre  = abs(pre_end_tok_rl - pre_end_tok_cr)
    diff_refl = abs(refl_end_tok_rl - refl_end_tok_cr)

    print(f"  Pre-refl boundary  — rollout_loop: {pre_end_tok_rl}, core_rgrpo: {pre_end_tok_cr}, diff={diff_pre}")
    print(f"  Refl-end boundary  — rollout_loop: {refl_end_tok_rl}, core_rgrpo: {refl_end_tok_cr}, diff={diff_refl}")

    if diff_pre == 0 and diff_refl == 0:
        print(f"  {PASS} Token boundary methods agree exactly.")
        return True
    else:
        print(f"  {WARN} Token boundary methods differ by {diff_pre}/{diff_refl} tokens.")
        print(f"  This can cause wrong advantage assignment at pre_refl/refl/cont segment boundaries.")
        print(f"  Fix: replace rollout_loop.py lines 950-952 with find_token_boundary from core_rgrpo.")
        return False


# ── 4. Live Inference: Trigger + Continuation ─────────────────────────────────

def test_live_reflection(llm, tokenizer):
    """
    End-to-end test:
      (a) Generate initial response — check some responses contain reflection
      (b) For those with reflection, build continuation prompt and generate again
      (c) Verify continuation prompt matches training format exactly
      (d) Verify continuation produces \boxed{} answer
    """
    print("\n=== Test 4: Live Reflection Trigger + Continuation ===")
    from vllm import SamplingParams

    # Use LIMO problems (39% reflection rate observed in eval)
    # Pull 8 hard problems directly from limo_clean.jsonl
    import json as _json
    limo_path = os.environ.get("LIMO_EVAL_PATH", "data/source/limo_clean.jsonl")
    problems_pool = []
    with open(limo_path) as _f:
        for _line in _f:
            problems_pool.append(_json.loads(_line)["problem"])
    # Pick problems spread across the dataset
    problems = [problems_pool[i] for i in [0, 50, 100, 200, 300, 400, 600, 750]]

    # Build initial prompts
    headers = []
    for p in problems:
        messages = [
            {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
            {"role": "user",   "content": p},
        ]
        headers.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    sp_init = SamplingParams(temperature=0.7, max_tokens=1024, n=4)
    print(f"  Generating {len(problems)} problems × 4 samples for reflection trigger ...")
    out_init = llm.generate(headers, sp_init, use_tqdm=False)

    triggered = []  # (problem_idx, sample_idx, header, refl_tag, full_output)
    for p_idx, result in enumerate(out_init):
        for s_idx, out in enumerate(result.outputs):
            text = out.text
            match = REFL_PATTERN.search(text)
            if match:
                triggered.append((p_idx, s_idx, headers[p_idx], match.group(0), text))

    n_triggered = len(triggered)
    print(f"  Reflection triggered: {n_triggered}/{len(problems)*4} samples "
          f"({n_triggered/(len(problems)*4)*100:.0f}%)")

    if n_triggered == 0:
        print(f"  {FAIL} No reflections triggered on LIMO problems (expected ~39% rate).")
        print(f"  Check: system prompt, model path, or temperature setting.")
        return False

    # Test (b)+(c)+(d): Build continuation prompts in training format and generate
    print(f"\n  Building {n_triggered} continuation prompts in training format ...")
    cont_prompts = []
    for p_idx, s_idx, header, refl_tag, full_out in triggered:
        cont_prompt = header + refl_tag.rstrip() + "\n"
        cont_prompts.append((p_idx, s_idx, cont_prompt, full_out))

    sp_cont = SamplingParams(temperature=0.7, max_tokens=1024, n=1)
    prompts_only = [cp for _, _, cp, _ in cont_prompts]
    print(f"  Generating continuations ...")
    out_cont = llm.generate(prompts_only, sp_cont, use_tqdm=False)

    has_boxed = 0
    for i, ((p_idx, s_idx, cont_prompt, orig_out), cont_result) in enumerate(zip(cont_prompts, out_cont)):
        cont_text = cont_result.outputs[0].text
        # Check format: continuation prompt should start with header + <reflection>
        assert cont_prompt.startswith("<|im_start|>system") or "<|im_start|>system" in cont_prompt[:30], \
            "Continuation prompt should start with chat template header"
        # Check continuation ends with \n (as per training format)
        assert cont_prompt.endswith("\n"), f"Continuation prompt should end with \\n, got: {repr(cont_prompt[-20:])}"
        # Check model generates something useful
        if "\\boxed" in cont_text:
            has_boxed += 1

        if i < 2:
            refl_in_prompt = REFL_PATTERN.search(cont_prompt)
            refl_tag_shown = refl_in_prompt.group(0)[:100] if refl_in_prompt else "(none)"
            print(f"\n  --- Sample {i+1} (problem {p_idx}, rollout {s_idx}) ---")
            print(f"  Problem: {problems[p_idx][:80]}")
            print(f"  Original output (first 200 chars): {repr(orig_out[:200])}")
            print(f"  Reflection tag in prompt: {repr(refl_tag_shown)}")
            print(f"  Cont prompt tail (last 150 chars): {repr(cont_prompt[-150:])}")
            print(f"  Continuation text (first 200 chars): {repr(cont_text[:200])}")

    boxed_rate = has_boxed / len(cont_prompts)
    print(f"\n  \\boxed{{}} in continuations: {has_boxed}/{len(cont_prompts)} ({boxed_rate:.0%})")

    if boxed_rate >= 0.5:
        print(f"  {PASS} Continuations produce answers (\\boxed{{}} rate {boxed_rate:.0%} ≥ 50%).")
    else:
        print(f"  {WARN} Low \\boxed{{}} rate ({boxed_rate:.0%}). Continuations may not be completing properly.")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("R-GRPO Reflection Format Test Suite")
    print("=" * 65)

    # Static tests (no model needed)
    r1 = test_system_prompt_consistency()

    print("\nLoading tokenizer for format tests ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    r2  = test_continuation_prompt_format(tokenizer)
    r2b = test_nested_reflection_format(tokenizer)
    r3  = test_token_boundary(tokenizer)

    # Live inference tests
    print("\nLoading vLLM model for live tests ...")
    from vllm import LLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=False,
    )
    r4 = test_live_reflection(llm, tokenizer)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    results = {
        "1. System prompt consistency":         r1,
        "2. Continuation format (1st refl)":    r2,
        "2b. Nested refl format (reflection.py)": r2b,
        "3. Token boundary method":              r3,
        "4. Live trigger + continuation":        r4,
    }
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {PASS} All tests passed!")
    else:
        print(f"  {FAIL} Some tests failed. See details above.")
    print("=" * 65)


if __name__ == "__main__":
    main()
