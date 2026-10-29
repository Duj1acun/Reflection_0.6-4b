from __future__ import annotations

import math
import re
import signal
from fractions import Fraction
from typing import Any, List, Optional, Union

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

from sympy import parse_expr, simplify

EPS = 1e-8


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


def _symbolic_compare(a: Any, b: Any) -> Optional[bool]:
    expr_a = _parse_symbolic_expr(a)
    expr_b = _parse_symbolic_expr(b)
    if expr_a is None or expr_b is None:
        return None

    def _check() -> bool:
        try:
            return bool(simplify(expr_a - expr_b) == 0)
        except Exception:
            try:
                return bool(simplify(expr_a) == simplify(expr_b))
            except Exception:
                return False

    try:
        def timeout_handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        try:
            return _check()
        finally:
            signal.alarm(0)
    except (TimeoutError, ValueError):
        return None


def _symbolic_equal(a: Any, b: Any) -> bool:
    result = _symbolic_compare(a, b)
    return bool(result) if result is not None else False


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


def compute_math_score(
    solution_str: str = None,
    ground_truth: Any = None,
    generated: str = None,
    data_source: str = None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """Robust 0/1 math reward using enhanced math equality checking."""
    del data_source, extra_info, kwargs
    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""
    if ground_truth is None:
        ground_truth = ""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", ground_truth.get("ground_truth", ""))

    try:
        pred_answer = extract_answer(generated)
        return 1.0 if math_equal(pred_answer, str(ground_truth)) else 0.0
    except Exception:
        return 0.0


def compute_score_robust(solution_str: str, ground_truth: str) -> float:
    """Legacy wrapper for backward compatibility."""
    return compute_math_score(solution_str=solution_str, ground_truth=ground_truth)


def compute_math_reward(
    response: str,
    gold_answer: str,
) -> float:
    return compute_math_score(solution_str=response, ground_truth=gold_answer)


def compute_batch_rewards(
    responses: List[str],
    gold_answers: List[str],
) -> List[float]:
    assert len(responses) == len(gold_answers)
    return [
        compute_math_reward(resp, gold)
        for resp, gold in zip(responses, gold_answers)
    ]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, dict],
    extra_info: Optional[dict] = None,
) -> float:
    del data_source, extra_info
    if isinstance(ground_truth, dict):
        gold_answer = ground_truth.get("answer", "")
    else:
        gold_answer = ground_truth

    return compute_math_reward(solution_str, gold_answer)
