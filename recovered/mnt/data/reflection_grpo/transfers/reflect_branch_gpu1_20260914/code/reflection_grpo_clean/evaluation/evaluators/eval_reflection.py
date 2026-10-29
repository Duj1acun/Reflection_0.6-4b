"""Reflection-continuation eval harness (single model, single process).

Replicates the TRAINING rollout's reflection-continuation sampling with a
standalone vLLM, and judges with the TRAINING answer extraction / judge:

  * direct  -> normal sampling (one round).
  * reflection -> on `<reflection>...</reflection>`, drop the pre-reflection
    wrong prefix and continue from `chat_header + reflection_tag.rstrip()+"\n"`
    (clean-slate, exactly rgrpo/reflection.py:175 & rollout_loop.py:951);
    repeat up to (max_iters-1) continuations.

Judging: math datasets -> rgrpo.core_hybrid_grpo_v2.compute_math_score
(training-identical, timeout+process-pool, last \\boxed{} extraction).
gpqa -> upstream-compatible Answer: LETTER regex (regex "Answer: $LETTER").

Metrics per question over n samples: mean@n, pass@n, maj@n; plus reflection
trigger rate and mean n_iters.

Usage:
  python eval_reflection.py --model-path <hf_dir> \
      --prompt-kind {qwen_math|reflection} --model-name <tag> \
      --datasets math500,aime24,aime25,aime26,gpqa \
      --n 32 --temp 1.0 --max-iters 3 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = str(Path(__file__).resolve().parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.datasets import MAX_TOKENS, load_dataset  # noqa: E402

# Training-identical judge + extraction (do NOT reimplement).
from rgrpo.core_hybrid_grpo_v2 import compute_math_score  # noqa: E402
from rgrpo.math_score_light import extract_answer, math_equal  # noqa: E402
from rgrpo.core_rgrpo import find_first_reflection  # noqa: E402
from rgrpo.reflection import REFLECTION_SYSTEM_PROMPT  # noqa: E402

QWEN_MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
REASONING_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
REFLECTION_REASONING_SYSTEM_PROMPT = (
    "You are a reasoning assistant. When you make an error during reasoning, "
    "use <reflection>...</reflection> to diagnose the mistake, then continue "
    "reasoning based on the reflection. Put your final answer within \\boxed{}."
)

# Exact pattern used by the upstream veRL GPQA judge.
GPQA_ANSWER_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")

# The repo judge uses re.search (FIRST "Answer: X", else 0). On reflection-
# continuation / math-RL outputs the final choice is often the LAST "Answer:"
# or a \boxed{LETTER} (RL pushes the policy toward \boxed{} math form). That
# strict first-only regex therefore scores 0 on answers the model did emit,
# under-reporting math-specialized models. Mirror the math side (extract_answer
# takes the LAST \boxed{}): take the LAST of all Answer:/\boxed{LETTER} signals.
GPQA_ANS_ALL = re.compile(r"(?i)Answer[ \t]*:[ \t]*\*{0,2}\$?\(?([A-D])\)?[.,)!]?\*{0,2}\$?")
GPQA_BOXED_LETTER = re.compile(r"\\boxed\{\s*\\?(?:text\{)?\s*\(?([A-D])\)?\s*\}?\s*\}")
# MMLU-Pro: 3-10 options (letters A-J).
MMLU_ANS_ALL = re.compile(r"(?i)Answer[ \t]*:[ \t]*\*{0,2}\$?\(?([A-J])\)?[.,)!]?\*{0,2}\$?")
MMLU_BOXED_LETTER = re.compile(r"\\boxed\{\s*\\?(?:text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$")
INLINE_ANSWER_RE = re.compile(r"(?i)(?:final\s+)?answer\s*:\s*(.+)")


def gpqa_extract_letter(full_text: str) -> str | None:
    """Final MCQ letter = last-occurring Answer:/\\boxed{LETTER} signal."""
    cands = [(m.start(), m.group(1).upper()) for m in GPQA_ANS_ALL.finditer(full_text)]
    cands += [(m.start(), m.group(1).upper()) for m in GPQA_BOXED_LETTER.finditer(full_text)]
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])[1]


def mmlu_pro_extract_letter(full_text: str) -> str | None:
    """MMLU-Pro letter (A-J), last-occurring."""
    cands = [(m.start(), m.group(1).upper()) for m in MMLU_ANS_ALL.finditer(full_text)]
    cands += [(m.start(), m.group(1).upper()) for m in MMLU_BOXED_LETTER.finditer(full_text)]
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])[1]


def _extract_last_boxed(text: str) -> str | None:
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth = 1
        out = []
        for ch in text[start:]:
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out).strip()
                out.append(ch)
            else:
                out.append(ch)
    bare = list(re.finditer(r"\\boxed\s+([^$\n]+)", text))
    if bare:
        return bare[-1].group(1).strip()
    return None


def _clean_extracted_answer(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r"</?answer>", "", ans, flags=re.I).strip()
    ans = re.sub(r"^\\(?:mathrm|mathbf|text)\{(.+)\}$", r"\1", ans).strip()
    ans = ans.strip("`*_ ")
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    if "\n" in ans:
        ans = next((line.strip() for line in ans.splitlines() if line.strip()), ans)
    return ans.strip()


def extract_freeform_answer(full_text: str) -> str | None:
    boxed = _extract_last_boxed(full_text)
    if boxed:
        return _clean_extracted_answer(boxed)
    matches = list(ANSWER_LINE_RE.finditer(full_text))
    if not matches:
        matches = list(INLINE_ANSWER_RE.finditer(full_text))
    if matches:
        return _clean_extracted_answer(matches[-1].group(1))
    lines = [
        ln.strip()
        for ln in full_text.strip().splitlines()
        if ln.strip() and not ln.strip().startswith("```")
    ]
    if not lines:
        return None
    return _clean_extracted_answer(lines[-1])


def _normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\\(?:mathrm|mathbf|text)\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^0-9a-z%.\-]+", " ", text)
    text = text.strip(" .,:;!?")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_dyck(text: Any) -> str:
    return re.sub(r"\s+", "", str(text).strip())


def _as_number(text: Any) -> float | None:
    s = str(text).strip().replace(",", "")
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _choice_letter(text: Any) -> str | None:
    s = str(text).strip()
    exact = re.fullmatch(r"\(?([A-Z])\)?", s)
    if exact:
        return exact.group(1)
    starts = re.match(r"^\s*\(([A-Z])\)\b", s)
    if starts:
        return starts.group(1)
    short_option = re.fullmatch(r"(?i)(?:option|choice)\s*\(?([A-Z])\)?\.?", s)
    if short_option:
        return short_option.group(1).upper()
    answerish = re.findall(
        r"(?i)\b(?:answer|correct\s+answer)\s*(?:is|:|-)?\s*\(?([A-Z])\)?\b",
        s,
    )
    if answerish:
        return answerish[-1].upper()
    return None


def _freeform_equal(pred: str | None, gold: Any, task: str | None = None) -> bool:
    if pred is None:
        return False
    golds = gold if isinstance(gold, list) else [gold]
    for g in golds:
        p = pred.strip()
        gg = str(g).strip()
        gm = re.fullmatch(r"\(([A-Z])\)", gg)
        if gm:
            return _choice_letter(p) == gm.group(1)
        if task == "dyck_languages":
            if _normalize_dyck(p) == _normalize_dyck(gg):
                return True
            continue
        pn, gn = _as_number(p), _as_number(gg)
        if pn is not None and gn is not None and abs(pn - gn) < 1e-9:
            return True
        if _normalize_text(p) == _normalize_text(gg):
            return True
    return False


def _yesno_extract(full_text: str) -> str | None:
    ans = extract_freeform_answer(full_text)
    if not ans:
        return None
    m = re.search(r"\b(yes|no|true|false)\b", ans, flags=re.I)
    if not m:
        return None
    val = m.group(1).lower()
    if val == "true":
        return "yes"
    if val == "false":
        return "no"
    return val


def system_prompt_for(kind: str) -> str:
    if kind == "reflection":
        return REFLECTION_SYSTEM_PROMPT
    if kind == "qwen_math":
        return QWEN_MATH_SYSTEM_PROMPT
    if kind == "reasoning":
        return REASONING_SYSTEM_PROMPT
    if kind == "reflection_reasoning":
        return REFLECTION_REASONING_SYSTEM_PROMPT
    raise ValueError(f"unknown prompt-kind {kind!r}")


def judge_row(row: dict[str, Any], full_text: str) -> int:
    """Training-identical correctness, 0/1."""
    kind = row["kind"]
    gold = row["gold"]
    try:
        if kind == "gpqa":
            return int(gpqa_extract_letter(full_text) == gold)
        if kind == "mmlu_pro":
            return int(mmlu_pro_extract_letter(full_text) == gold)
        if kind == "bbh":
            return int(_freeform_equal(extract_freeform_answer(full_text), gold, row.get("task")))
        if kind == "strategyqa":
            return int(_yesno_extract(full_text) == gold)
        if kind in {"drop", "clutrr"}:
            return int(_freeform_equal(extract_freeform_answer(full_text), gold))
        return int(
            compute_math_score(generated=full_text, ground_truth=gold) > 0.5
        )
    except Exception:  # noqa: BLE001  -- never let one sample crash the run
        return 0


def extracted_key_row(row: dict[str, Any], full_text: str) -> str | None:
    """Per-sample extracted answer used for majority voting."""
    kind = row["kind"]
    try:
        if kind == "gpqa":
            return gpqa_extract_letter(full_text)
        if kind == "mmlu_pro":
            return mmlu_pro_extract_letter(full_text)
        if kind == "strategyqa":
            return _yesno_extract(full_text)
        if kind in {"bbh", "drop", "clutrr"}:
            return extract_freeform_answer(full_text)
        a = extract_answer(full_text)
        return a.strip() if a else None
    except Exception:  # noqa: BLE001
        return None


def maj_correct_row(row: dict[str, Any], keys: list[str | None]) -> int:
    kind = row["kind"]
    gold = row["gold"]
    valid = [k for k in keys if k]
    if not valid:
        return 0
    winner = Counter(valid).most_common(1)[0][0]
    try:
        if kind in ("gpqa", "mmlu_pro"):
            return int(winner == gold)
        if kind == "bbh":
            return int(_freeform_equal(winner, gold, row.get("task")))
        if kind == "strategyqa":
            return int(winner == gold)
        if kind in {"drop", "clutrr"}:
            return int(_freeform_equal(winner, gold))
        return int(bool(math_equal(winner, gold)))
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True, help="tag for output files")
    ap.add_argument(
        "--prompt-kind",
        required=True,
        choices=["qwen_math", "reflection", "reasoning", "reflection_reasoning"],
    )
    ap.add_argument("--datasets", default=",".join(MAX_TOKENS.keys()))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Override per-round max tokens for every dataset "
                         "(use 3072 for single-turn GRPO models trained with "
                         "max_response_length=3072).")
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument(
        "--prompt-truncation",
        choices=["left", "right", "middle", "error"],
        default="left",
    )
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix KV caching for repeated question/header prefixes.",
    )
    ap.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable vLLM chunked prefill. Off by default for strict old-config parity.",
    )
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--out", default=os.path.join(REPO, "artifacts", "evaluation")
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    sys_prompt = system_prompt_for(args.prompt_kind)

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model_path,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "enforce_eager": False,
        "trust_remote_code": True,
        "seed": args.seed,
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    def header_for(problem: str) -> tuple[str, int, int, bool]:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": problem},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        raw_len = len(prompt_ids)
        if len(prompt_ids) > args.max_prompt_len:
            if args.prompt_truncation == "left":
                prompt_ids = prompt_ids[-args.max_prompt_len :]
            elif args.prompt_truncation == "right":
                prompt_ids = prompt_ids[: args.max_prompt_len]
            elif args.prompt_truncation == "middle":
                left_half = args.max_prompt_len // 2
                right_half = args.max_prompt_len - left_half
                prompt_ids = prompt_ids[:left_half] + prompt_ids[-right_half:]
            elif args.prompt_truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(prompt_ids)} exceeds max-prompt-len={args.max_prompt_len}"
                )
        return (
            tok.decode(prompt_ids, skip_special_tokens=False),
            raw_len,
            len(prompt_ids),
            raw_len > args.max_prompt_len,
        )

    for ds_name in datasets:
        rows = load_dataset(ds_name)
        if args.limit is not None:
            rows = rows[: args.limit]
        max_tok = args.max_tokens if args.max_tokens is not None else MAX_TOKENS[ds_name]
        header_infos = [header_for(r["problem"]) for r in rows]
        headers = [h[0] for h in header_infos]
        raw_prompt_tokens = [h[1] for h in header_infos]
        prompt_tokens = [h[2] for h in header_infos]
        prompt_truncated = [h[3] for h in header_infos]

        # ---- PHASE 1: round-0 generation, n samples per question ----
        sp0 = SamplingParams(
            n=args.n,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=-1,
            max_tokens=max_tok,
            seed=args.seed,
        )
        outs0 = llm.generate(headers, sp0, use_tqdm=True)

        # trajectory state, flat index = qi * n + g
        trajs = []  # each: {qi, header, texts:[...], n_refl:int, active:bool, last_tag}
        for qi, ro in enumerate(outs0):
            for comp in ro.outputs:
                txt = comp.text
                m = find_first_reflection(txt)
                trajs.append(
                    {
                        "qi": qi,
                        "header": headers[qi],
                        "texts": [txt],
                        "token_counts": [len(getattr(comp, "token_ids", []) or [])],
                        "n_refl": 0,
                        "active": m is not None,
                        "last_tag": m.group(0) if m else None,
                    }
                )

        # ---- reflection continuations, up to max_iters-1 extra rounds ----
        for _ in range(max(0, args.max_iters - 1)):
            act = [t for t in trajs if t["active"]]
            if not act:
                break
            # Mirror training's _truncate_cont_prompt: left-truncate to
            # max_prompt_len + max_tok//2 tokens so continuation prompts
            # never exceed the budget the rollout_loop enforces.
            max_cont_tokens = args.max_prompt_len + max_tok // 2
            cont_prompts = []
            for t in act:
                raw = t["header"] + t["last_tag"].rstrip() + "\n"
                ids = tok.encode(raw, add_special_tokens=False)
                if len(ids) > max_cont_tokens:
                    raw = tok.decode(ids[-max_cont_tokens:], skip_special_tokens=False)
                cont_prompts.append(raw)
            sp = SamplingParams(
                n=1,
                temperature=args.temp,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=max_tok,
                seed=args.seed,
            )
            cont_outs = llm.generate(cont_prompts, sp, use_tqdm=True)
            for t, co in zip(act, cont_outs):
                comp = co.outputs[0]
                txt = comp.text
                t["texts"].append(txt)
                t["token_counts"].append(len(getattr(comp, "token_ids", []) or []))
                t["n_refl"] += 1
                m = find_first_reflection(txt)
                if m is not None:
                    t["last_tag"] = m.group(0)
                    t["active"] = True
                else:
                    t["active"] = False

        # ---- PHASE 2: judge (training-identical) ----
        kind = rows[0]["kind"]
        per_q: dict[int, list[dict]] = {}
        samples = []
        for t in trajs:
            row = rows[t["qi"]]
            full = "".join(t["texts"])
            gold = row["gold"]
            ok = judge_row(row, full)
            key = extracted_key_row(row, full)
            decoded_tokens = sum(t["token_counts"])
            qi = t["qi"]
            rec = {
                "uid": row["uid"],
                "qi": qi,
                "gold": gold,
                "extracted": key,
                "correct": ok,
                "num_reflections": t["n_refl"],
                "n_iters": 1 + t["n_refl"],
                "prompt_tokens_raw": raw_prompt_tokens[qi],
                "prompt_tokens": prompt_tokens[qi],
                "prompt_truncated": prompt_truncated[qi],
                "decoded_tokens": decoded_tokens,
                "total_tokens": prompt_tokens[qi] + decoded_tokens,
                "decoded_chars": len(full),
                "full_text": full,
            }
            if "task" in row:
                rec["task"] = row["task"]
            samples.append(rec)
            per_q.setdefault(qi, []).append(rec)

        # ---- aggregate: mean@n / pass@n / maj@n + reflection stats ----
        q_mean, q_pass, q_maj, q_reflrate, q_niters = [], [], [], [], []
        # direct / reflection accuracy computed globally over all samples
        direct_correct = direct_total = 0
        refl_correct   = refl_total   = 0
        for qi, recs in per_q.items():
            corr = [r["correct"] for r in recs]
            q_mean.append(sum(corr) / len(corr))
            q_pass.append(1 if any(corr) else 0)
            q_maj.append(
                maj_correct_row(rows[qi], [r["extracted"] for r in recs])
            )
            q_reflrate.append(
                sum(1 for r in recs if r["num_reflections"] > 0) / len(recs)
            )
            q_niters.append(sum(r["n_iters"] for r in recs) / len(recs))
            for r in recs:
                if r["num_reflections"] == 0:
                    direct_total   += 1
                    direct_correct += r["correct"]
                else:
                    refl_total   += 1
                    refl_correct += r["correct"]

        nq = len(per_q)
        summary = {
            "model": args.model_name,
            "model_path": args.model_path,
            "prompt_kind": args.prompt_kind,
            "dataset": ds_name,
            "kind": kind,
            "n_questions": nq,
            "n_samples_per_q": args.n,
            "max_tokens_per_round": max_tok,
            "max_prompt_len": args.max_prompt_len,
            "prompt_truncation": args.prompt_truncation,
            "max_iters": args.max_iters,
            "mean_at_n": sum(q_mean) / nq,
            "pass_at_n": sum(q_pass) / nq,
            "maj_at_n": sum(q_maj) / nq,
            "reflection_trigger_rate": sum(q_reflrate) / nq,
            "avg_n_iters": sum(q_niters) / nq,
            "accuracy_direct":     direct_correct / direct_total if direct_total else None,
            "accuracy_reflection": refl_correct   / refl_total   if refl_total   else None,
            "n_direct_samples":    direct_total,
            "n_reflection_samples": refl_total,
            "avg_prompt_tokens_raw": (
                sum(raw_prompt_tokens) / len(raw_prompt_tokens) if raw_prompt_tokens else 0.0
            ),
            "avg_prompt_tokens": (
                sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0
            ),
            "max_prompt_tokens_raw": max(raw_prompt_tokens) if raw_prompt_tokens else 0,
            "n_prompt_truncated": sum(1 for x in prompt_truncated if x),
            "prompt_truncated_rate": (
                sum(1 for x in prompt_truncated if x) / len(prompt_truncated)
                if prompt_truncated else 0.0
            ),
            "avg_decoded_tokens": (
                sum(r["decoded_tokens"] for r in samples) / len(samples)
                if samples else 0.0
            ),
            "avg_total_tokens": (
                sum(r["total_tokens"] for r in samples) / len(samples)
                if samples else 0.0
            ),
            "avg_decoded_tokens_reflection": (
                sum(r["decoded_tokens"] for r in samples if r["num_reflections"] > 0)
                / sum(1 for r in samples if r["num_reflections"] > 0)
                if any(r["num_reflections"] > 0 for r in samples)
                else None
            ),
        }

        tag = f"{args.model_name}__{ds_name}"
        with open(os.path.join(args.out, f"{tag}.summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out, f"{tag}.samples.jsonl"), "w") as f:
            for rec in samples:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        da = summary["accuracy_direct"]
        ra = summary["accuracy_reflection"]
        _line = (
            f"[{tag}] mean@{args.n}={summary['mean_at_n']:.4f} "
            f"pass@{args.n}={summary['pass_at_n']:.4f} "
            f"maj@{args.n}={summary['maj_at_n']:.4f} "
            f"refl_rate={summary['reflection_trigger_rate']:.3f} "
            f"avg_iters={summary['avg_n_iters']:.3f}"
        )
        if da is not None:
            _line += f" acc_direct={da:.3f}"
        if ra is not None:
            _line += f" acc_refl={ra:.3f}"
        print(_line, flush=True)


if __name__ == "__main__":
    main()
