"""Lightweight math-answer scoring helpers.

This module is intentionally free of torch/ray imports so it can be used by
spawn/forkserver scoring workers without re-importing the training stack.
"""

from __future__ import annotations

import math
import os
import re
import signal
import threading
import time
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Optional

from sympy import parse_expr, simplify

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


class MathScoreTimeout(TimeoutError):
    pass


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@contextmanager
def _score_alarm(timeout: float):
    if timeout <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(signum, frame):
        del signum, frame
        raise MathScoreTimeout()

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


REFL_PATTERN = re.compile(r"<reflection>[\s\S]+?</reflection>", re.DOTALL)
_REFLECTION_OPEN_RE = re.compile(r"<reflection\b[^>]*>", re.IGNORECASE)
_REFLECTION_CLOSE_RE = re.compile(r"</reflection>", re.IGNORECASE)
_REFLECTION_TAG_RE = re.compile(
    r"<reflection\b[^>]*>(.*?)</reflection>",
    re.IGNORECASE | re.DOTALL,
)
_REFLECTION_FORMAT_RE = re.compile(
    r"\bso\s+far\s*,?\s+i\s+have\b.*?\bhowever\b",
    re.IGNORECASE | re.DOTALL,
)


def reflection_markup_stats(generated: str) -> tuple[int, bool]:
    text = generated or ""
    complete_count = len(REFL_PATTERN.findall(text))
    has_markup = bool(
        complete_count
        or _REFLECTION_OPEN_RE.search(text)
        or _REFLECTION_CLOSE_RE.search(text)
    )
    return complete_count, has_markup


def reflection_format_valid(text: Any) -> bool:
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


def should_skip_math_for_invalid_pretrigger_format(generated: str) -> bool:
    if not _env_bool("PRETRIGGER_SKIP_MATH_ON_INVALID_FORMAT", "1"):
        return False
    _count, has_markup = reflection_markup_stats(generated)
    if not has_markup:
        return False
    require_format = _env_bool(
        "PRETRIGGER_REQUIRE_REFLECTION_FORMAT",
        os.environ.get("CF_REFL_REQUIRE_REFLECTION_FORMAT", "1"),
    )
    return bool(require_format and not reflection_format_valid(generated))


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
    match = re.match(re.escape(command) + r"\s*\{", string[start:])
    if not match:
        return None
    brace_start = start + match.end() - 1
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


def pathological_answer(prediction: Any, reference: Any) -> bool:
    if not _env_bool("HYBRID_V2_FAST_REJECT_PATHOLOGICAL_ANSWERS", "1"):
        return False

    raw = str(prediction or "").strip()
    if not raw:
        return True

    pred = strip_math_string(raw)
    ref = strip_math_string(reference)
    if pred and ref and pred.lower() == ref.lower():
        return False

    max_chars = _env_int("HYBRID_V2_MAX_EXTRACTED_ANSWER_CHARS", 384)
    if len(raw) > max_chars or len(pred) > max_chars:
        return True

    parts = _split_top_level(pred)
    if len(parts) > _env_int("HYBRID_V2_MAX_EXTRACTED_ANSWER_PARTS", 16):
        return True

    text_without_latex_commands = re.sub(r"\\[A-Za-z]+", " ", raw)
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z]{2,}", text_without_latex_commands)
        if word.lower() not in _ALLOWED_MATH_WORDS
    ]
    if len(words) > _env_int("HYBRID_V2_MAX_EXTRACTED_ANSWER_WORDS", 8):
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
        except MathScoreTimeout:
            raise
        except Exception:
            pass
    try:
        return parse_expr(expr)
    except MathScoreTimeout:
        raise
    except Exception:
        return None


def _parse_numeric(num: Any) -> Optional[float]:
    num_str = re.sub(",", "", strip_math_string(num))
    try:
        return float(num_str)
    except MathScoreTimeout:
        raise
    except Exception:
        pass
    unitless = re.sub(r"^(-?(?:\d+(?:\.\d*)?|\.\d+))(?:[A-Za-z]+(?:\^\d+)?)$", r"\1", num_str)
    if unitless != num_str:
        try:
            return float(unitless)
        except MathScoreTimeout:
            raise
        except Exception:
            pass
    try:
        return float(Fraction(num_str))
    except MathScoreTimeout:
        raise
    except Exception:
        pass
    expr = _parse_symbolic_expr(num_str)
    if expr is not None:
        try:
            if getattr(expr, "free_symbols", set()):
                return None
            return float(expr.evalf())
        except MathScoreTimeout:
            raise
        except Exception:
            pass
    return None


def _symbolic_compare(a: Any, b: Any) -> Optional[bool]:
    expr_a = _parse_symbolic_expr(a)
    expr_b = _parse_symbolic_expr(b)
    if expr_a is None or expr_b is None:
        return None
    try:
        return bool(simplify(expr_a - expr_b) == 0)
    except MathScoreTimeout:
        raise
    except Exception:
        try:
            return bool(simplify(expr_a) == simplify(expr_b))
        except MathScoreTimeout:
            raise
        except Exception:
            return False


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


def _math_verify_equal(prediction: Any, reference: Any) -> bool:
    if not MATH_VERIFY_AVAILABLE:
        return False
    try:
        pred = math_verify_parse(str(prediction))
        ref = math_verify_parse(str(reference))
        if not pred or not ref:
            return False
        return bool(math_verify_verify(pred, ref))
    except MathScoreTimeout:
        raise
    except Exception:
        return False


def _basic_math_equal(prediction: Any, reference: Any) -> bool:
    prediction_str = strip_math_string(prediction)
    reference_str = strip_math_string(reference)
    if prediction_str == reference_str:
        return True
    if prediction_str.lower() == reference_str.lower():
        return True

    if _has_base_notation(prediction_str) or _has_base_notation(reference_str):
        return _normalize_base_notation(prediction_str) == _normalize_base_notation(reference_str)

    pred_assignment = _assignment_rhs(prediction_str)
    ref_assignment = _assignment_rhs(reference_str)
    if pred_assignment or ref_assignment:
        if pred_assignment and ref_assignment and pred_assignment[0] != ref_assignment[0]:
            return False
        pred_rhs = pred_assignment[1] if pred_assignment else prediction_str
        ref_rhs = ref_assignment[1] if ref_assignment else reference_str
        return _basic_math_equal(pred_rhs, ref_rhs)

    pred_num = _parse_numeric(prediction_str)
    ref_num = _parse_numeric(reference_str)
    if pred_num is not None and ref_num is not None and math.isclose(
        pred_num, ref_num, rel_tol=1e-4, abs_tol=1e-8
    ):
        return True

    symbolic_result = _symbolic_compare(prediction_str, reference_str)
    if symbolic_result is not None:
        return symbolic_result

    if _math_verify_equal(prediction, reference):
        return True

    return False


def _pm_variants(expr: Any) -> Optional[list[str]]:
    expr = str(expr).strip()
    if r"\pm" not in expr:
        return None
    match = re.match(r"^(.*?)\\pm\s*(.+)$", expr)
    if not match:
        return None
    base = match.group(1).strip()
    delta = match.group(2).strip()
    if not base or base in {"+", r"\left"}:
        return [delta, f"-{delta}"]
    return [f"{base}+{delta}", f"{base}-{delta}"]


def _compare_unordered_parts(pred_parts: list[str], ref_parts: list[str]) -> bool:
    if len(pred_parts) != len(ref_parts):
        return False
    remaining = list(ref_parts)
    for pred in pred_parts:
        match_idx = None
        for idx, ref in enumerate(remaining):
            if _basic_math_equal(pred, ref):
                match_idx = idx
                break
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return True


def _compare_part_lists(prediction: Any, reference: Any) -> Optional[bool]:
    pred_body, pred_kind = _unwrap_container(prediction)
    ref_body, ref_kind = _unwrap_container(reference)
    pred_parts = _split_top_level(pred_body)
    ref_parts = _split_top_level(ref_body)
    if len(pred_parts) <= 1 and len(ref_parts) <= 1:
        return None
    if len(pred_parts) != len(ref_parts):
        return False
    unordered = pred_kind == "set" or ref_kind == "set" or (pred_kind == ref_kind == "bare")
    if unordered:
        return _compare_unordered_parts(pred_parts, ref_parts)
    return all(_basic_math_equal(p, r) for p, r in zip(pred_parts, ref_parts))


def math_equal(prediction: Any, reference: Any) -> bool:
    if prediction is None or reference is None:
        return False

    pred_pm = _pm_variants(prediction)
    ref_pm = _pm_variants(reference)
    if pred_pm is not None or ref_pm is not None:
        pred_parts = pred_pm if pred_pm is not None else _split_top_level(_unwrap_container(prediction)[0])
        ref_parts = ref_pm if ref_pm is not None else _split_top_level(_unwrap_container(reference)[0])
        return _compare_unordered_parts(pred_parts, ref_parts)

    if strip_math_string(prediction) == strip_math_string(reference):
        return True

    parts_equal = _compare_part_lists(prediction, reference)
    if parts_equal is not None:
        return parts_equal

    return _basic_math_equal(prediction, reference)


def compute_math_score_inner(pred_answer: str, ground_truth: str) -> float:
    return 1.0 if math_equal(pred_answer, ground_truth) else 0.0


def compute_math_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    generated = payload.get("solution_str", payload.get("generated", ""))
    ground_truth = payload.get("ground_truth", "")
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", ground_truth.get("ground_truth", ""))

    timeout = float(payload.get("timeout", _env_float("HYBRID_V2_MATH_SCORE_TIMEOUT", 5.0)))
    slow_log_sec = float(payload.get("slow_log_sec", _env_float("HYBRID_V2_SCORE_SLOW_LOG_SEC", 2.0)))
    start = time.monotonic()
    out = {
        "score": 0.0,
        "math_score_timeout": 0.0,
        "math_score_exception": 0.0,
        "math_score_fast_reject": 0.0,
        "math_score_skipped_by_format": 0.0,
        "math_score_slow": 0.0,
        "math_score_elapsed_s": 0.0,
    }
    try:
        if should_skip_math_for_invalid_pretrigger_format(str(generated)):
            out["math_score_skipped_by_format"] = 1.0
            return out
        with _score_alarm(timeout):
            pred_answer = extract_answer(str(generated))
            if pathological_answer(pred_answer, ground_truth):
                out["math_score_fast_reject"] = 1.0
            else:
                out["score"] = float(compute_math_score_inner(str(pred_answer), str(ground_truth)))
    except MathScoreTimeout:
        out["math_score_timeout"] = 1.0
    except Exception:
        out["math_score_exception"] = 1.0
    finally:
        elapsed = time.monotonic() - start
        out["math_score_elapsed_s"] = float(elapsed)
        if elapsed >= slow_log_sec:
            out["math_score_slow"] = 1.0
    return out


def compute_math_score(
    solution_str: str = None,
    ground_truth: Any = None,
    generated: str = None,
    data_source: str = None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    del data_source, extra_info, kwargs
    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""
    result = compute_math_score_payload(
        {
            "solution_str": generated,
            "ground_truth": ground_truth,
            "timeout": _env_float("HYBRID_V2_MATH_SCORE_TIMEOUT", 5.0),
        }
    )
    return float(result.get("score", 0.0))
