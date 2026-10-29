# Copyright 2025 R-GRPO Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Hybrid-GRPO v2: direct/event 联合归一化 + 续推局部归一化

设计要点
========

奖励原始值
----------
direct sample raw     = +1 if is_correct else -DIRECT_WRONG_PENALTY  ∈ {-0.0625, 1}
                        # 错答按 (1/PENALTY_BASE)^2 量级惩罚，与 n_refl=1 的
                        # 反思惩罚同量级 → "错的 direct" ≡ "深度 1 全错的 event"
event reflection raw  = format_ok * (
                          1[any continuation correct]
                          - penalty(n_reflections)
                          + EVENT_RECOVERY_COEF * max(0, cont_acc - dir_acc_uid)
                        )
                        # 最后一项：K 条续推平均成功率超过 direct 基线时的 recovery
                        # bonus；uid 内无 direct 样本（无可比基线）时该项置 0，
                        # 避免与 fallback 基线 0 相减误判为强恢复。
cont local raw        = is_correct                              ∈ {0, 1}

归一化（per-uid 联合池）
------------------------
direct_adv/event_adv  = z-score over [direct raws, event raws] on same uid
                        with a std floor to avoid tiny-std batches turning
                        small raw gaps into very large negative event_adv.
                        联合池保留：模型需要在同一道题里看到"直接答 vs 反思"
                        的相对排序，才能学到"易题直接答、难题反思"的策略。
local_cont_adv        = z-score among K cont samples within event     ← cont 局部池
                        no-signal event (K 全对/全错) 自然得到全 0 advantage，
                        不再注入人造信号；K 越大该退化越罕见。

Token-level 优势分配
--------------------
direct sample:        adv = direct_adv 整条（使用 response_mask）
event_reflection sample:
  若 hybrid_prefix_len 落在 (0, valid_len) 区间：
    tokens [0:prefix_len)  ← 0           （不动反思前推理链）
    tokens [prefix_len:)   ← event_adv   （只训练反思段及以后）
  否则（prefix_len=0 / 无效 / 缺字段）：
    整条 ← event_adv * response_mask     （兼容当前 rollout 默认行为）
cont sample:
  all tokens            ← local_cont_adv（v2 无 prefix_len 切分，续推整条受训）

模块加载副作用
--------------
import 本模块时会把 rgrpo.core_hybrid_grpo.compute_hybrid_outcome_advantage
和 .compute_hybrid_math_score monkey-patch 成 v2 版本，从而 verl 的
adv_estimator=hybrid_grpo dispatch 透明落到 v2，无需修改 verl 任何代码。

启用方式
--------
env: RGRPO_HYBRID_V2=1   （由 rgrpo/__init__.py 触发本模块导入）
hydra: custom_reward_function.path=rgrpo.core_hybrid_grpo_v2
       custom_reward_function.name=compute_hybrid_v2_math_score
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import defaultdict
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sympy import parse_expr, simplify

from . import core_rgrpo as _core_rgrpo
from . import math_score_light as _math_score_light
from .core_rgrpo import REFL_PATTERN

try:
    from latex2sympy2_extended import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    try:
        from latex2sympy2 import latex2sympy
        LATEX2SYMPY_AVAILABLE = True
    except ImportError:
        LATEX2SYMPY_AVAILABLE = False

try:
    from math_verify import parse as math_verify_parse
    from math_verify import verify as math_verify_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# ── 超参 ──────────────────────────────────────────────────────────────────
EPS = 1e-8
MAX_REFLECTIONS = 3
PENALTY_BASE = 4      # penalty(n) = min((n / PENALTY_BASE)^2, 1.0)
# 直接样本错答惩罚：对答 +1，错答 -DIRECT_WRONG_PENALTY。
# 默认与 n_refl=1 的反思惩罚同量级 ((1/PENALTY_BASE)^2 = 0.0625)，
# 可通过 HYBRID_V2_DIRECT_WRONG_PENALTY 环境变量覆盖。
DIRECT_WRONG_PENALTY = float(
    os.environ.get(
        "HYBRID_V2_DIRECT_WRONG_PENALTY",
        str(min((1 / PENALTY_BASE) ** 2, 1.0)),
    )
)
GLOBAL_ADV_STD_FLOOR = float(os.environ.get("HYBRID_V2_GLOBAL_STD_FLOOR", "0.25"))
EVENT_ANY_CORRECT_COEF = float(os.environ.get("HYBRID_V2_EVENT_ANY_CORRECT_COEF", "1.0"))
EVENT_CONT_ACC_COEF = float(os.environ.get("HYBRID_V2_EVENT_CONT_ACC_COEF", "1.0"))
EVENT_DIR_ACC_BETA = float(os.environ.get("HYBRID_V2_EVENT_DIR_ACC_BETA", "1.0"))
# Recovery bonus 系数：event_raw 上叠加 EVENT_RECOVERY_COEF * max(0, cont_acc - dir_acc_uid)。
# 设为 0 即关闭该项，回退到 v2 原始 event_raw 公式。
EVENT_RECOVERY_COEF = float(os.environ.get("HYBRID_V2_EVENT_RECOVERY_COEF", "1.0"))
PREFIX_CONTRAST_ENABLE = os.environ.get("HYBRID_V2_PREFIX_CONTRAST", "0").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
PREFIX_CONTRAST_PENALTY_COEF = float(
    os.environ.get("HYBRID_V2_PREFIX_CONTRAST_PENALTY_COEF", "1.0")
)
# 格式不合格反思的兜底 raw：与 DIRECT_WRONG_PENALTY 同量纲（=-0.0625），语义上
# 把"格式失败"等价于"答错 direct"——这样 z-score 后两类样本拿到相同的 adv，
# 任何 uid 池形状下都不会出现"格式不合格反思反而比错的 direct 优势更高"的情况。
# 实测 v5 用 -0.25 时 trigger_rate 在 step 20 就被吓到 0.337（v3 是 0.45+），
# 模型把"反思能力"丢了，val 反而退步 6pp/100steps。
EVENT_INVALID_FORMAT_REWARD = float(
    os.environ.get("HYBRID_V2_EVENT_INVALID_FORMAT_REWARD", str(-DIRECT_WRONG_PENALTY))
)
_EVENT_ADV_MIN_RAW = os.environ.get("HYBRID_V2_EVENT_ADV_MIN", "")
EVENT_ADV_MIN = float(_EVENT_ADV_MIN_RAW) if _EVENT_ADV_MIN_RAW != "" else None
_EVENT_ADV_MAX_RAW = os.environ.get("HYBRID_V2_EVENT_ADV_MAX", "")
EVENT_ADV_MAX = float(_EVENT_ADV_MAX_RAW) if _EVENT_ADV_MAX_RAW != "" else None


_MATH_SCORE_TIMEOUT_SEC = float(os.environ.get("HYBRID_V2_MATH_SCORE_TIMEOUT", "5"))

_REFLECTION_TAG_RE = re.compile(
    r"<reflection\b[^>]*>(.*?)</reflection>",
    re.IGNORECASE | re.DOTALL,
)
_REFLECTION_OPEN_RE = re.compile(r"<reflection\b[^>]*>", re.IGNORECASE)
_REFLECTION_CLOSE_RE = re.compile(r"</reflection>", re.IGNORECASE)
_REFLECTION_FORMAT_RE = re.compile(
    r"\bso\s+far\s*,?\s+i\s+have\b.*?\bhowever\b",
    re.IGNORECASE | re.DOTALL,
)

# Thread-based timeout. SIGALRM-based timeouts only work on the main thread of
# the main interpreter, but compute_math_score is invoked from a Ray actor
# worker thread (verl's RayPPOTrainer dispatches the rollout to a TaskRunner
# actor), so the previous _SignalTimeout silently no-op'd and sympy could hang
# the driver indefinitely on adversarial LaTeX. The thread executor below works
# from any thread; on timeout the inner thread is leaked (CPython has no safe
# thread-interrupt) but the caller continues. The pool is replaced after each
# leak so leaked threads don't starve subsequent calls.
import concurrent.futures as _cf
import multiprocessing as _mp
import threading as _threading

try:
    from concurrent.futures.process import BrokenProcessPool as _BrokenProcessPool
except Exception:  # pragma: no cover - very old/edge Python
    _BrokenProcessPool = RuntimeError

# Bound on how many times one scoring call may be recomputed after the shared
# process pool is torn down (by *another* sample's timeout) before falling
# back to the default score. Prevents an unbounded retry loop while still
# letting collaterally-broken valid samples be rescored instead of mislabeled.
_SCORE_MAX_ATTEMPTS = max(1, int(os.environ.get("HYBRID_V2_MATH_SCORE_MAX_ATTEMPTS", "3")))

_process_score_pool_lock = _threading.Lock()
_process_score_pool: Optional[_cf.ProcessPoolExecutor] = None
_PROCESS_SCORE_POOL_WORKERS = int(os.environ.get("HYBRID_V2_MATH_SCORE_PROCESS_WORKERS", "16"))
_SCORE_PROCESS_START_METHOD = os.environ.get("HYBRID_V2_MATH_SCORE_START_METHOD", "spawn").lower()
_FAST_REJECT_PATHOLOGICAL_ANSWERS = os.environ.get(
    "HYBRID_V2_FAST_REJECT_PATHOLOGICAL_ANSWERS", "1"
).lower() in {"1", "true", "yes", "y"}
_MAX_EXTRACTED_ANSWER_CHARS = int(os.environ.get("HYBRID_V2_MAX_EXTRACTED_ANSWER_CHARS", "384"))
_MAX_EXTRACTED_ANSWER_WORDS = int(os.environ.get("HYBRID_V2_MAX_EXTRACTED_ANSWER_WORDS", "8"))
_MAX_EXTRACTED_ANSWER_PARTS = int(os.environ.get("HYBRID_V2_MAX_EXTRACTED_ANSWER_PARTS", "16"))
_SCORE_SLOW_LOG_SEC = float(os.environ.get("HYBRID_V2_SCORE_SLOW_LOG_SEC", "2.0"))

_timeout_stats_lock = _threading.Lock()
_TIMEOUT_HIT_COUNT = 0
_SCORE_CALL_COUNT = 0
_FAST_REJECT_COUNT = 0
_SLOW_SCORE_COUNT = 0
_SCORE_LOG_EVERY = int(os.environ.get("HYBRID_V2_TIMEOUT_LOG_EVERY", "50"))


def _record_timeout_hit() -> None:
    global _TIMEOUT_HIT_COUNT
    with _timeout_stats_lock:
        _TIMEOUT_HIT_COUNT += 1


def _record_fast_reject_hit() -> None:
    global _FAST_REJECT_COUNT
    with _timeout_stats_lock:
        _FAST_REJECT_COUNT += 1


def _record_slow_score_hit() -> None:
    global _SLOW_SCORE_COUNT
    with _timeout_stats_lock:
        _SLOW_SCORE_COUNT += 1


def _maybe_log_timeout_window() -> None:
    """Periodically print timeout hit count to stderr so we can see how often
    sympy paths are getting cut off in practice."""
    global _SCORE_CALL_COUNT, _TIMEOUT_HIT_COUNT, _FAST_REJECT_COUNT, _SLOW_SCORE_COUNT
    with _timeout_stats_lock:
        _SCORE_CALL_COUNT += 1
        if _SCORE_CALL_COUNT < _SCORE_LOG_EVERY:
            return
        hits = _TIMEOUT_HIT_COUNT
        fast_rejects = _FAST_REJECT_COUNT
        slow_scores = _SLOW_SCORE_COUNT
        calls = _SCORE_CALL_COUNT
        _TIMEOUT_HIT_COUNT = 0
        _FAST_REJECT_COUNT = 0
        _SLOW_SCORE_COUNT = 0
        _SCORE_CALL_COUNT = 0
    if hits > 0 or fast_rejects > 0 or slow_scores > 0:
        print(
            (
                f"[hybrid_v2] math_score window: calls={calls} "
                f"timeouts={hits} fast_rejects={fast_rejects} slow_scores={slow_scores}"
            ),
            file=sys.stderr,
            flush=True,
        )


def _get_process_score_pool() -> _cf.ProcessPoolExecutor:
    global _process_score_pool
    with _process_score_pool_lock:
        if _process_score_pool is None:
            os.environ["RGRPO_LIGHT_IMPORT"] = "1"
            start_method = _SCORE_PROCESS_START_METHOD
            if start_method not in {"spawn", "forkserver"}:
                print(
                    f"[hybrid_v2] unsupported math score start method {start_method!r}; using spawn",
                    file=sys.stderr,
                    flush=True,
                )
                start_method = "spawn"
            try:
                ctx = _mp.get_context(start_method)
            except ValueError:
                print(
                    f"[hybrid_v2] math score start method {start_method!r} unavailable; using spawn",
                    file=sys.stderr,
                    flush=True,
                )
                ctx = _mp.get_context("spawn")
            _process_score_pool = _cf.ProcessPoolExecutor(
                max_workers=max(1, _PROCESS_SCORE_POOL_WORKERS),
                mp_context=ctx,
            )
        return _process_score_pool


_REPLACE_ANY = object()


def _replace_process_score_pool(only_if=_REPLACE_ANY) -> None:
    """Terminate the score process pool so a wedged parser cannot block forever.

    ``only_if`` lets a caller request the teardown *only* if the live pool is
    still the exact object it was using. Concurrent timeouts in the same wave
    all pass the same stale pool, so the first one rebuilds and the rest become
    no-ops -- this collapses the thundering-herd replacement storm.
    """
    global _process_score_pool
    with _process_score_pool_lock:
        old = _process_score_pool
        if old is None:
            return
        if only_if is not _REPLACE_ANY and old is not only_if:
            # Already replaced by another sample's timeout in this wave; do
            # not tear down the fresh pool other scorers are now using.
            return
        _process_score_pool = None
        processes = list((getattr(old, "_processes", None) or {}).values())
        for proc in processes:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            old.shutdown(wait=False)
        except Exception:
            pass
        for proc in processes:
            try:
                proc.join(timeout=0.2)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass


def _run_score_with_timeout(fn, args=(), timeout: float = _MATH_SCORE_TIMEOUT_SEC, default=None):
    """Run one answer-judging call in the shared killable process pool.

    Concurrency-safe by design:

    * A genuinely slow sample triggers exactly one pool rebuild (compare-and-
      swap on the pool object), then returns ``default`` -- it cannot be
      scored within the wall-clock budget anyway.
    * A *valid* sample whose worker is collaterally killed because some other
      sample timed out (cf.ProcessPoolExecutor breaks wholesale on any worker
      death) raises ``BrokenProcessPool``. It is **recomputed** on a fresh
      pool instead of being silently mislabeled as ``default`` (0.0). This is
      the bug fix: previously every in-flight valid sample in the wave was
      scored wrong.
    * Retries are bounded by ``_SCORE_MAX_ATTEMPTS``; a genuine error inside
      ``fn`` is not retried.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            pool = _get_process_score_pool()
        except Exception:
            return default
        try:
            fut = pool.submit(fn, *args)
        except (_BrokenProcessPool, RuntimeError):
            # Pool already broken/shutting down (collateral). Rebuild it (only
            # if still the one we grabbed) and recompute rather than scoring a
            # valid answer as wrong.
            if attempts < _SCORE_MAX_ATTEMPTS:
                _replace_process_score_pool(only_if=pool)
                continue
            return default
        except Exception:
            return default
        try:
            return fut.result(timeout=timeout)
        except _cf.TimeoutError:
            # This sample is genuinely too slow. Kill the wedged worker by
            # replacing the pool (cf pools break wholesale on worker death),
            # but only once per wave via the compare-and-swap.
            _record_timeout_hit()
            _replace_process_score_pool(only_if=pool)
            return default
        except _BrokenProcessPool:
            # Collateral: the pool was torn down by another sample's timeout
            # while this valid sample was running. Recompute it.
            if attempts < _SCORE_MAX_ATTEMPTS:
                _replace_process_score_pool(only_if=pool)
                continue
            return default
        except Exception:
            # Genuine error raised inside fn -- not a pool problem; do not
            # retry, do not tear down the pool (avoids needless churn).
            return default


# α schedule 参数（当前版本未接入主流程，预留给后续实现）
_DEFAULT_ALPHA_BASE = 0.0
_DEFAULT_ALPHA_FINAL = 0.05
_DEFAULT_ALPHA_WARMUP_STEPS = 500

# step 计数器（当前未使用，_alpha_for_step 尚未接入 compute_hybrid_v2_outcome_advantage）
_step_counter: Dict[str, int] = {"step": 0}


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _alpha_for_step(step: int) -> float:
    base = float(os.environ.get("HYBRID_V2_ALPHA_BASE", _DEFAULT_ALPHA_BASE))
    final = float(os.environ.get("HYBRID_V2_ALPHA_FINAL", _DEFAULT_ALPHA_FINAL))
    warmup = int(os.environ.get("HYBRID_V2_ALPHA_WARMUP_STEPS", _DEFAULT_ALPHA_WARMUP_STEPS))
    if warmup <= 0:
        return final
    if step >= warmup:
        return final
    frac = step / warmup
    return base + (final - base) * frac

def _reflection_penalty(n_refl: int) -> float:
    n = max(0, min(int(n_refl), MAX_REFLECTIONS))
    return min((n / PENALTY_BASE) ** 2, 1.0)



def _grpo_norm(rewards: list, std_floor: float = 0.0) -> list:
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    s = max(math.sqrt(var), float(std_floor))
    return [(r - m) / (s + EPS) for r in rewards]


def _safe_mean(lst: list) -> float:
    return float(sum(lst) / len(lst)) if lst else float("nan")


def _is_no_signal(rewards: list) -> bool:
    if not rewards or len(rewards) == 1:
        return True
    first = rewards[0]
    return all(abs(r - first) <= EPS for r in rewards)


def _normalize_prefix_len(prefix_len: Any, valid_len: int) -> Optional[int]:
    if valid_len <= 0:
        return None
    try:
        prefix_len = int(prefix_len)
    except (TypeError, ValueError):
        return None
    if prefix_len < 0 or prefix_len >= valid_len:
        return None
    return prefix_len


def _reflection_format_valid(text: Any) -> bool:
    """Require each complete reflection tag to follow:
    'So far I have ... However ...'. Malformed/unclosed tags are invalid.
    """
    if text is None:
        return False
    text = str(text)
    tags = list(_REFLECTION_TAG_RE.finditer(text))
    if not tags:
        return False
    if len(_REFLECTION_OPEN_RE.findall(text)) != len(tags):
        return False
    if len(_REFLECTION_CLOSE_RE.findall(text)) != len(tags):
        return False
    return all(_REFLECTION_FORMAT_RE.search(match.group(1)) for match in tags)


def _array_flag(arr: Any, idx: int, default: float) -> float:
    if arr is None:
        return float(default)
    try:
        val = arr[idx]
    except (IndexError, TypeError, KeyError):
        return float(default)
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if bool(val) else 0.0
    if isinstance(val, str):
        return 1.0 if val.strip().lower() in {"1", "true", "yes", "valid"} else 0.0
    try:
        return 1.0 if float(val) > 0.5 else 0.0
    except (TypeError, ValueError):
        return float(default)


# ── Robust math answer extraction / verification ───────────────────────────
# Keep this self-contained so run_hybrid_grpo_v2_math.sh only needs
# Imported from the repository root. The implementation mirrors the
# stronger math_rewardflow checker: reliable boxed extraction, math_verify,
# numeric/symbolic fallbacks, and ordered-vs-unordered part comparison.

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_math_string(string: Any) -> str:
    string = str(string).strip()
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{(.*?)\}", r"\1", string)
    string = string.replace("^{\\circ}", "")
    string = string.replace("\\circ", "")
    string = string.replace("°", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = re.sub(r"\s*degrees", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*inches\^2", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*square inches", "", string, flags=re.IGNORECASE)
    string = string.replace("\n", "")
    if string.endswith("."):
        string = string[:-1]
    string = string.replace("\\!", "")
    string = string.replace("\\ ", " ")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\n", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("$", "")
    string = re.sub(r" ", "", string)
    string = re.sub("\u200b", "", string)
    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    return string.strip()


def _find_braced_command_content(string: str, command: str, start: int) -> Optional[str]:
    m = re.match(re.escape(command) + r"\s*\{", string[start:])
    if not m:
        return None
    brace_start = start + m.end() - 1
    depth = 0
    pos = brace_start
    while pos < len(string):
        ch = string[pos]
        if ch == "\\":
            pos += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return string[brace_start + 1:pos].strip()
        pos += 1
    return string[brace_start + 1:].strip()


def extract_answer(generated: str) -> str:
    if generated is None:
        return ""

    candidates = []
    for command in (r"\boxed", r"\fbox"):
        for match in re.finditer(re.escape(command) + r"\s*\{", generated):
            content = _find_braced_command_content(generated, command, match.start())
            if content is not None:
                candidates.append((match.start(), content))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]

    bare_boxed = list(re.finditer(r"\\boxed\s+([^$\n]+)", generated))
    if bare_boxed:
        return bare_boxed[-1].group(1).strip()

    answer_tags = list(re.finditer(r"<answer>(.*?)</answer>", generated, re.DOTALL))
    if answer_tags:
        tagged = answer_tags[-1].group(1).strip()
        nested = extract_answer(tagged)
        return nested if nested else tagged

    gsm8k_answers = re.findall(r"####\s*(-?[0-9\.\,]+)", generated)
    if gsm8k_answers:
        return gsm8k_answers[-1].replace(",", "").replace("$", "")

    answer_pattern = re.findall(
        r"(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)",
        generated,
    )
    if answer_pattern:
        return answer_pattern[-1]

    lines = [line.strip() for line in generated.splitlines() if line.strip()]
    return lines[-1] if lines else generated.strip()


_PROSE_HINT_RE = re.compile(
    r"\b(because|therefore|thus|hence|since|answer|final|problem|solution|"
    r"calculate|computed?|we|our|i|think|is|are|equals?)\b",
    re.IGNORECASE,
)
_MATH_SIGNAL_RE = re.compile(r"[\d=+\-*/^_{}()[\],.;]|\\[A-Za-z]+")
_ALLOWED_MATH_WORDS = {
    "frac",
    "sqrt",
    "cdot",
    "times",
    "div",
    "left",
    "right",
    "pi",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "lambda",
    "mu",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "log",
    "ln",
    "exp",
    "min",
    "max",
    "mod",
    "infty",
    "infinity",
    "pm",
    "le",
    "ge",
    "neq",
    "text",
    "begin",
    "end",
    "matrix",
    "pmatrix",
    "bmatrix",
    "cases",
    "no",
    "none",
    "solution",
    "undefined",
}


def _pathological_answer(prediction: Any, reference: Any) -> bool:
    if not _FAST_REJECT_PATHOLOGICAL_ANSWERS:
        return False

    raw = str(prediction or "").strip()
    if not raw:
        return True

    pred = strip_math_string(raw)
    ref = strip_math_string(reference)
    if pred and ref and pred.lower() == ref.lower():
        return False

    if len(raw) > _MAX_EXTRACTED_ANSWER_CHARS or len(pred) > _MAX_EXTRACTED_ANSWER_CHARS:
        return True

    parts = _split_top_level(pred)
    if len(parts) > _MAX_EXTRACTED_ANSWER_PARTS:
        return True

    text_without_latex_commands = re.sub(r"\\[A-Za-z]+", " ", raw)
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z]{2,}", text_without_latex_commands)
        if word.lower() not in _ALLOWED_MATH_WORDS
    ]
    if len(words) > _MAX_EXTRACTED_ANSWER_WORDS:
        return True

    has_math_signal = bool(_MATH_SIGNAL_RE.search(raw))
    if not has_math_signal and len(words) >= 2:
        return True

    if _PROSE_HINT_RE.search(text_without_latex_commands) and len(words) >= 3:
        return True

    return False


def _parse_symbolic_expr(expr: Any):
    expr = strip_math_string(expr)
    if not expr:
        return None
    if LATEX2SYMPY_AVAILABLE:
        try:
            return latex2sympy(expr)
        except Exception:
            pass
    try:
        return parse_expr(expr)
    except Exception:
        return None


def _parse_numeric(num: Any) -> Optional[float]:
    num_str = re.sub(",", "", strip_math_string(num))
    try:
        return float(num_str)
    except Exception:
        pass
    unitless = re.sub(r"^(-?(?:\d+(?:\.\d*)?|\.\d+))(?:[A-Za-z]+(?:\^\d+)?)$", r"\1", num_str)
    if unitless != num_str:
        try:
            return float(unitless)
        except Exception:
            pass
    try:
        return float(Fraction(num_str))
    except Exception:
        pass
    expr = _parse_symbolic_expr(num_str)
    if expr is not None:
        try:
            if getattr(expr, "free_symbols", set()):
                return None
            return float(expr.evalf())
        except Exception:
            pass
    return None


_BASE_NOTATION_RE = re.compile(r"(?<![A-Za-z\\])\d+\s*_\s*\{?\d+\}?")
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)=(.+)$")


def _has_base_notation(text: str) -> bool:
    return bool(_BASE_NOTATION_RE.search(text))


def _normalize_base_notation(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return re.sub(r"(\d+)_\{?(\d+)\}?", r"\1_\2", text)


def _assignment_rhs(text: str) -> Optional[tuple[str, str]]:
    match = _ASSIGNMENT_RE.match(text)
    if not match:
        return None
    return match.group(1), match.group(2)


def _split_top_level(text: str) -> list[str]:
    parts = []
    buf = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            buf.append(text[i:i + 2])
            i += 2
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch in ",;" and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def _unwrap_container(text: Any) -> tuple[str, str]:
    s = strip_math_string(text)
    if "\\begin{pmatrix}" in s or "\\begin{bmatrix}" in s or "\\begin{matrix}" in s:
        return s, "ordered"
    if s.startswith(r"\{") and s.endswith(r"\}"):
        return s[2:-2], "set"
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1], "set"
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        return s[1:-1], "ordered"
    return s, "bare"


def compute_math_score(
    solution_str: str = None,
    ground_truth: Any = None,
    generated: str = None,
    data_source: str = None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """Robust 0/1 math reward used by Hybrid-GRPO v2."""
    del data_source, extra_info, kwargs
    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""
    if ground_truth is None:
        ground_truth = ""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", ground_truth.get("ground_truth", ""))

    if os.environ.get("FOURWAY_CONFIG"):
        from fourway.scoring import judge
        return float(judge(str(generated), str(ground_truth))["correct"])

    start = time.monotonic()
    try:
        pred_answer = _math_score_light.extract_answer(str(generated))
        if _math_score_light.pathological_answer(pred_answer, ground_truth):
            _record_fast_reject_hit()
            score = 0.0
        else:
            # Outer wall-clock cap for sympy/math_verify. The default backend
            # is a subprocess because a worker thread cannot interrupt C/regex
            # code that holds the GIL.
            args = (str(pred_answer), str(ground_truth))
            # The sympy/math_verify judge is always run in a killable worker
            # process. The old thread/inline backends could not interrupt
            # C/regex code holding the GIL and would hang the driver.
            score = float(
                _run_score_with_timeout(
                    _math_score_light.compute_math_score_inner,
                    args=args,
                    timeout=_MATH_SCORE_TIMEOUT_SEC * 2,
                    default=0.0,
                )
            )
    except Exception:
        score = 0.0
    finally:
        if time.monotonic() - start >= _SCORE_SLOW_LOG_SEC:
            _record_slow_score_hit()
        _maybe_log_timeout_window()
    return score


# ── 主函数：Hybrid v2 优势计算 ────────────────────────────────────────────

def compute_hybrid_v2_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    hybrid_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 1e-6,
    metrics_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    batch_size = token_level_rewards.shape[0]
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)

    if hybrid_info is None:
        return _fallback_grpo(token_level_rewards, response_mask, index, epsilon)

    refl_group_ids = hybrid_info.get("hybrid_refl_group_id", None)
    n_refl_arr     = hybrid_info.get("hybrid_n_refl",       np.zeros(batch_size, dtype=np.float32))
    roles          = hybrid_info.get("hybrid_sample_role",  None)
    prefix_len_arr = hybrid_info.get("hybrid_prefix_len",   None)
    format_valid_arr = hybrid_info.get("reflection_format_valid", None)
    cf_continue_arr = hybrid_info.get("hybrid_cf_continue_acc", None)

    # ── 累积器 ────────────────────────────────────────────────────────
    stat_direct_correct: list = []
    stat_cont_correct:   list = []
    stat_event_raw:      list = []
    stat_event_raw_ungated: list = []
    stat_event_adv:      list = []
    stat_event_penalty:  list = []
    stat_event_recovery_term: list = []   # max(0, cont_acc - dir_acc_uid)
    stat_event_cf_continue_acc: list = []
    stat_event_prefix_delta: list = []
    stat_event_prefix_contrast_used: list = []
    stat_event_format_valid: list = []
    stat_dir_acc_uid:    list = []
    stat_cont_local_advs: list = []
    stat_event_reflection_adv: list = []
    stat_zero_local:     list = []
    stat_n_refl:         list = []
    stat_event_recovery: list = []   # 1[dir_acc_uid<0.4 and any_correct=1]
    stat_event_regression: list = [] # 1[dir_acc_uid>0.6 and any_correct=0]
    stat_event_all_correct: list = []  # 1[K 条 cont 全对]
    stat_event_all_wrong:   list = []  # 1[K 条 cont 全错]
    n_event_reflection = 0
    n_cont_only_mode = 0

    uid_to_indices: Dict[Any, list] = defaultdict(list)
    for i, uid in enumerate(index):
        uid_to_indices[uid].append(i)

    def _safe_prefix_len(idx: int):
        if prefix_len_arr is None:
            return None
        try:
            return prefix_len_arr[idx]
        except (IndexError, TypeError, KeyError):
            return None

    for uid, sample_indices in uid_to_indices.items():
        # ── 1. 分类 direct / event_reflection / continuation ────────
        direct_indices: list = []
        event_reflection_indices: list = []  # (sample_idx, rg_id, n_refl, raw_prefix_len)
        cont_indices:   list = []  # (sample_idx, rg_id, n_refl, raw_prefix_len)

        for i in sample_indices:
            role = str(roles[i]) if roles is not None else (
                "direct" if (refl_group_ids is None or refl_group_ids[i] == -1) else "continuation"
            )
            if role == "direct":
                direct_indices.append(i)
            elif role == "event_reflection":
                rg_id = int(refl_group_ids[i]) if refl_group_ids is not None else -1
                n_refl = int(n_refl_arr[i])
                event_reflection_indices.append((i, rg_id, n_refl, _safe_prefix_len(i)))
            elif role == "continuation":
                rg_id = int(refl_group_ids[i]) if refl_group_ids is not None else -1
                n_refl = int(n_refl_arr[i])
                cont_indices.append((i, rg_id, n_refl, _safe_prefix_len(i)))

        # ── 2. 计算 dir_acc_uid（条件基线）────────────────────────────
        direct_correct = [float(scores[i].item() > 0.5) for i in direct_indices]
        # direct 进入联合池的 raw：对答 +1，错答 -DIRECT_WRONG_PENALTY，与
        # event_raw = any_correct - penalty(n_refl) 保持同量纲。
        direct_raw = [
            1.0 if c > 0.5 else -DIRECT_WRONG_PENALTY for c in direct_correct
        ]
        has_direct_sample = bool(direct_correct)
        if has_direct_sample:
            dir_acc_uid = sum(direct_correct) / len(direct_correct)
        else:
            # 无 direct 样本（罕见 corner case）：基线置 0；下游 recovery 项也会
            # 因 has_direct_sample=False 整体置 0，避免把 cont_acc 误读成"强恢复"。
            dir_acc_uid = 0.0
        stat_dir_acc_uid.append(dir_acc_uid)
        stat_direct_correct.extend(direct_correct)

        # ── 3. 按 event 分组 cont 样本 ────────────────────────────────
        event_to_samples: Dict[int, list] = defaultdict(list)
        event_n_refl_map: Dict[int, int]  = {}
        event_format_valid_map: Dict[int, float] = {}
        event_cf_continue_map: Dict[int, float] = {}
        default_event_format_valid = 1.0 if format_valid_arr is None else 0.0
        for (i, rg_id, _, _) in event_reflection_indices:
            is_valid = _array_flag(format_valid_arr, i, default=default_event_format_valid)
            if rg_id in event_format_valid_map:
                event_format_valid_map[rg_id] = min(event_format_valid_map[rg_id], is_valid)
            else:
                event_format_valid_map[rg_id] = is_valid
            if cf_continue_arr is not None:
                try:
                    cf_val = float(cf_continue_arr[i])
                except (IndexError, TypeError, ValueError):
                    cf_val = float("nan")
                if not math.isnan(cf_val):
                    event_cf_continue_map[rg_id] = cf_val
        for (i, rg_id, n_refl, _) in cont_indices:
            event_to_samples[rg_id].append(i)
            event_n_refl_map[rg_id] = n_refl
            if rg_id not in event_cf_continue_map and cf_continue_arr is not None:
                try:
                    cf_val = float(cf_continue_arr[i])
                except (IndexError, TypeError, ValueError):
                    cf_val = float("nan")
                if not math.isnan(cf_val):
                    event_cf_continue_map[rg_id] = cf_val

        # ── 4. 每个 event：raw + 局部 cont z-score ────────────────────
        event_raw_map: Dict[int, float]  = {}
        local_cont_adv: Dict[int, float] = {}

        for rg_id, sids in event_to_samples.items():
            cont_correct_list = [float(scores[i].item() > 0.5) for i in sids]
            cont_acc = sum(cont_correct_list) / len(cont_correct_list) if cont_correct_list else 0.0
            n_refl = event_n_refl_map[rg_id]

            any_correct = 1.0 if any(cont_correct_list) else 0.0
            refl_penalty = _reflection_penalty(n_refl)
            # Recovery bonus：cont 平均成功率超过 direct 基线时奖励，仅在该 uid
            # 内有 direct 样本（基线可比）时启用；ReLU 保证不会变成额外负惩罚，
            # "恶化"由 any_correct=0 与 refl_penalty 共同兜底。
            if has_direct_sample:
                recovery_term = max(0.0, cont_acc - dir_acc_uid)
            else:
                recovery_term = 0.0
            cf_continue_acc = event_cf_continue_map.get(rg_id, float("nan"))
            use_prefix_contrast = PREFIX_CONTRAST_ENABLE and not math.isnan(cf_continue_acc)
            if use_prefix_contrast:
                prefix_delta = cont_acc - cf_continue_acc
                event_raw_ungated = max(-1.0, min(1.0, (
                    prefix_delta - PREFIX_CONTRAST_PENALTY_COEF * refl_penalty
                )))
            else:
                prefix_delta = float("nan")
                event_raw_ungated = max(-1.0, min(1.0, (
                    EVENT_ANY_CORRECT_COEF * any_correct
                    - refl_penalty
                    + EVENT_RECOVERY_COEF * recovery_term
                )))
            format_valid = event_format_valid_map.get(rg_id, default_event_format_valid)
            event_raw = event_raw_ungated if format_valid > 0.5 else EVENT_INVALID_FORMAT_REWARD
            event_raw_map[rg_id] = event_raw
            stat_event_raw.append(event_raw)
            stat_event_raw_ungated.append(event_raw_ungated)
            stat_event_penalty.append(refl_penalty)
            stat_event_recovery_term.append(recovery_term)
            stat_event_cf_continue_acc.append(cf_continue_acc)
            stat_event_prefix_delta.append(prefix_delta)
            stat_event_prefix_contrast_used.append(float(use_prefix_contrast))
            stat_event_format_valid.append(format_valid)
            stat_n_refl.append(float(n_refl))
            stat_cont_correct.extend(cont_correct_list)
            # 新增：监控 recovery / regression 标记
            stat_event_recovery.append(
                float(dir_acc_uid < 0.4 and any_correct > 0.5)
            )
            stat_event_regression.append(
                float(dir_acc_uid > 0.6 and any_correct < 0.5)
            )
            # 新增：K 条 cont 全对 / 全错（zero_event_rate 的细分）
            if cont_correct_list:
                stat_event_all_correct.append(float(all(c > 0.5 for c in cont_correct_list)))
                stat_event_all_wrong.append(float(all(c < 0.5 for c in cont_correct_list)))

            # 局部 cont z-score（同 event 内 K 条续推）。K 全对/全错时
            # _grpo_norm 自然返回全 0，等于该 event 不贡献梯度——这是 GRPO
            # 数学上"应有的样子"，不再注入人造信号。zero_event_rate 仅作监控。
            stat_zero_local.append(float(_is_no_signal(cont_correct_list)))
            local_advs = _grpo_norm(cont_correct_list)
            for sid, la in zip(sids, local_advs):
                local_cont_adv[sid] = la
                stat_cont_local_advs.append(la)

        # ── 5. direct/event 联合池 z-score（同 uid 内）────────────────
        direct_adv_map: Dict[int, float] = {}
        event_adv_map: Dict[int, float] = {}
        global_keys = []
        global_raws = []
        for i, raw in zip(direct_indices, direct_raw):
            global_keys.append(("direct", i))
            global_raws.append(raw)
        for rg_id in event_to_samples.keys():
            global_keys.append(("event", rg_id))
            global_raws.append(event_raw_map[rg_id])

        for key, adv in zip(global_keys, _grpo_norm(global_raws, std_floor=GLOBAL_ADV_STD_FLOOR)):
            kind, ident = key
            if kind == "direct":
                direct_adv_map[int(ident)] = adv
            else:
                if EVENT_ADV_MIN is not None:
                    adv = max(float(EVENT_ADV_MIN), adv)
                if EVENT_ADV_MAX is not None:
                    adv = min(float(EVENT_ADV_MAX), adv)
                event_adv_map[int(ident)] = adv
                stat_event_adv.append(adv)

        # ── 6. 写入 direct 样本优势 ───────────────────────────────────
        for i in direct_indices:
            mask = response_mask[i]
            adv = direct_adv_map.get(i, 0.0)
            advantages[i] = adv * mask
            returns[i]    = scores[i] * mask

        # ── 7. 写入 event_reflection 样本优势 ───────────────────────
        # prefix_len > 0 时切分：[0:prefix_len) 不动反思前推理链，[prefix_len:)
        # 才施加 event_adv。当前 rollout 默认 hybrid_prefix_len=0（反思前推理链
        # 已经在 prompt 部分、不在 response_mask 里），等价于整条施加 event_adv。
        # 这里做防御性切分，便于后续 rollout 端语义升级时无需再改本函数。
        for (i, rg_id, _, raw_prefix_len) in event_reflection_indices:
            mask = response_mask[i]
            adv = event_adv_map.get(rg_id, 0.0)
            valid_len = int(mask.sum().item())
            prefix_len = _normalize_prefix_len(raw_prefix_len, valid_len)
            if prefix_len is not None and prefix_len > 0:
                adv_tensor = torch.zeros_like(mask, dtype=advantages.dtype)
                adv_tensor[prefix_len:] = adv
                advantages[i] = adv_tensor * mask
            else:
                advantages[i] = adv * mask
            returns[i]    = scores[i] * mask
            stat_event_reflection_adv.append(adv)
            n_event_reflection += 1

        # ── 8. 写入 cont 样本优势（只训练续推本身）───────────────────
        for (i, rg_id, _, _) in cont_indices:
            mask = response_mask[i]
            local_adv = local_cont_adv.get(i, 0.0)
            advantages[i] = local_adv * mask
            n_cont_only_mode += 1
            returns[i] = scores[i] * mask
        # cont 样本（v2）目前 rollout 端 hybrid_prefix_len 始终为 0，response 仅
        # 为续推段，前缀已在 prompt；故无需在此切分。若后续把前缀放回 response，
        # 复用 _normalize_prefix_len 即可（参考 v1 的 split_prefix_len 写法）。

    # ── 9. 写入指标 ────────────────────────────────────────────────────
    if metrics_out is not None:
        n_d = len(stat_direct_correct)
        n_c = len(stat_cont_correct)
        total = n_d + n_c

        metrics_out["hybrid_v2/n_direct_samples"] = n_d
        metrics_out["hybrid_v2/n_cont_samples"]   = n_c
        metrics_out["hybrid_v2/n_event_reflection_samples"] = n_event_reflection
        metrics_out["hybrid_v2/trigger_rate"]     = n_c / total if total > 0 else 0.0
        metrics_out["hybrid_v2/direct_acc"]       = _safe_mean(stat_direct_correct)
        metrics_out["hybrid_v2/cont_acc"]         = _safe_mean(stat_cont_correct)
        metrics_out["hybrid_v2/dir_acc_uid_mean"] = _safe_mean(stat_dir_acc_uid)

        metrics_out["hybrid_v2/event/raw_mean"]   = _safe_mean(stat_event_raw)
        metrics_out["hybrid_v2/event/raw_signed_mean"] = _safe_mean(stat_event_raw)
        metrics_out["hybrid_v2/event/raw_ungated_mean"] = _safe_mean(stat_event_raw_ungated)
        metrics_out["hybrid_v2/event/adv_mean"]   = _safe_mean(stat_event_adv)
        metrics_out["hybrid_v2/event/refl_penalty_mean"] = _safe_mean(stat_event_penalty)
        metrics_out["hybrid_v2/event/recovery_term_mean"] = _safe_mean(stat_event_recovery_term)
        metrics_out["hybrid_v2/event/recovery_term_pos_rate"] = _safe_mean(
            [float(r > EPS) for r in stat_event_recovery_term]
        )
        finite_cf = [x for x in stat_event_cf_continue_acc if not math.isnan(x)]
        finite_delta = [x for x in stat_event_prefix_delta if not math.isnan(x)]
        metrics_out["hybrid_v2/event/prefix_contrast_used_rate"] = _safe_mean(
            stat_event_prefix_contrast_used
        )
        metrics_out["hybrid_v2/event/cf_continue_acc_mean"] = _safe_mean(finite_cf)
        metrics_out["hybrid_v2/event/prefix_delta_mean"] = _safe_mean(finite_delta)
        metrics_out["hybrid_v2/event/prefix_delta_pos_rate"] = _safe_mean(
            [float(x > EPS) for x in finite_delta]
        )
        metrics_out["hybrid_v2/event/recovery_coef"] = float(EVENT_RECOVERY_COEF)
        metrics_out["hybrid_v2/event/prefix_contrast_penalty_coef"] = float(
            PREFIX_CONTRAST_PENALTY_COEF
        )
        metrics_out["hybrid_v2/event/prefix_contrast_enabled"] = float(
            PREFIX_CONTRAST_ENABLE
        )
        metrics_out["hybrid_v2/event/invalid_format_reward"] = float(
            EVENT_INVALID_FORMAT_REWARD
        )
        metrics_out["hybrid_v2/event/raw_pos_rate"] = _safe_mean(
            [float(r > EPS) for r in stat_event_raw]
        )
        metrics_out["hybrid_v2/event/reflection_format_valid_rate"] = _safe_mean(
            stat_event_format_valid
        )
        metrics_out["hybrid_v2/event/format_gate_zero_rate"] = _safe_mean(
            [float(v <= 0.5) for v in stat_event_format_valid]
        )
        metrics_out["hybrid_v2/event/recovery_rate"]   = _safe_mean(stat_event_recovery)
        metrics_out["hybrid_v2/event/regression_rate"] = _safe_mean(stat_event_regression)
        metrics_out["hybrid_v2/event/all_correct_rate"] = _safe_mean(stat_event_all_correct)
        metrics_out["hybrid_v2/event/all_wrong_rate"]   = _safe_mean(stat_event_all_wrong)
        metrics_out["hybrid_v2/event/n_all_correct"]    = float(sum(stat_event_all_correct))
        metrics_out["hybrid_v2/event/n_all_wrong"]      = float(sum(stat_event_all_wrong))

        metrics_out["hybrid_v2/refl_penalty_base"] = float(PENALTY_BASE)
        metrics_out["hybrid_v2/direct_wrong_penalty"] = float(DIRECT_WRONG_PENALTY)
        metrics_out["hybrid_v2/global_adv_std_floor"] = float(GLOBAL_ADV_STD_FLOOR)
        metrics_out["hybrid_v2/event/any_correct_coef"] = float(EVENT_ANY_CORRECT_COEF)
        metrics_out["hybrid_v2/event/cont_acc_coef"] = float(EVENT_CONT_ACC_COEF)
        metrics_out["hybrid_v2/event/dir_acc_beta"] = float(EVENT_DIR_ACC_BETA)
        metrics_out["hybrid_v2/event/adv_min"] = (
            float(EVENT_ADV_MIN) if EVENT_ADV_MIN is not None else float("nan")
        )
        metrics_out["hybrid_v2/event/adv_max"] = (
            float(EVENT_ADV_MAX) if EVENT_ADV_MAX is not None else float("nan")
        )
        metrics_out["hybrid_v2/n_refl_depth/mean"] = _safe_mean(stat_n_refl)
        metrics_out["hybrid_v2/event_reflection_abs_mean"] = _safe_mean(
            [abs(x) for x in stat_event_reflection_adv]
        )
        metrics_out["hybrid_v2/local_signal/zero_event_rate"] = _safe_mean(stat_zero_local)
        metrics_out["hybrid_v2/cont_local_abs_mean"] = _safe_mean(
            [abs(x) for x in stat_cont_local_advs]
        )
        metrics_out["hybrid_v2/cont_local_pos_rate"] = _safe_mean(
            [float(x > EPS) for x in stat_cont_local_advs]
        )
        metrics_out["hybrid_v2/cont_local_neg_rate"] = _safe_mean(
            [float(x < -EPS) for x in stat_cont_local_advs]
        )

        metrics_out["hybrid_v2/prefix_mode/cont_only_rate"] = (
            n_cont_only_mode / n_c if n_c > 0 else float("nan")
        )

    return advantages, returns


def _fallback_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """hybrid_info 缺失时的标准 GRPO 退化实现。"""
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(token_level_rewards)
    returns    = torch.zeros_like(token_level_rewards)

    uid_to_scores: Dict[Any, list] = defaultdict(list)
    for i, uid in enumerate(index):
        uid_to_scores[uid].append(scores[i].item())

    uid_mean = {uid: float(np.mean(v)) for uid, v in uid_to_scores.items()}
    uid_std  = {uid: float(np.std(v)) + epsilon for uid, v in uid_to_scores.items()}

    for i, uid in enumerate(index):
        adv = (scores[i].item() - uid_mean[uid]) / uid_std[uid]
        advantages[i] = adv * response_mask[i]
        returns[i]    = scores[i] * response_mask[i]

    return advantages, returns


# ── 兼容 NaiveRewardManager 的每样本奖励函数 ─────────────────────────────

def compute_hybrid_v2_math_score(
    solution_str: str = None,
    ground_truth: str = None,
    generated: str = None,
    data_source: str = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """与 v1 一致：返回 0/1 score + 反思元信息。

    训练时驱动 token_level_rewards；
    验证时同时为 veRL 提供 has_reflection / n_reflections / is_correct 字段。
    """
    del data_source, kwargs

    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""

    extra_info = extra_info if isinstance(extra_info, dict) else {}

    if "hybrid_reward" in extra_info:
        return {"score": float(extra_info["hybrid_reward"])}

    score = float(compute_math_score(solution_str=generated, ground_truth=ground_truth))
    is_correct = score > 0.5
    n_reflections = len(REFL_PATTERN.findall(generated))
    has_reflection = n_reflections > 0
    reflection_format_valid = _reflection_format_valid(generated)

    return {
        "score":          score,
        "is_correct":     float(is_correct),
        "has_reflection": float(has_reflection),
        "n_reflections":  float(n_reflections),
        "reflection_format_valid": float(reflection_format_valid),
    }


# ── Monkey-patch v1 dispatch（核心机关）────────────────────────────────
# 让 verl 在 adv_estimator=hybrid_grpo 时调用本模块的 v2 函数，无需改 verl。
# core_hybrid_grpo 必定先于本模块被 import（rgrpo/__init__.py 顺序）。
from . import core_hybrid_grpo as _orig  # noqa: E402

_core_rgrpo.compute_math_score = compute_math_score
_orig.compute_math_score = compute_math_score
_orig.compute_hybrid_outcome_advantage = compute_hybrid_v2_outcome_advantage
_orig.compute_hybrid_math_score = compute_hybrid_v2_math_score
