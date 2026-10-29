"""Versioned MATH outcome judge for new four-method experiments.

Only the final answer is parsed. Exact scalar arithmetic is cheap; structured
answers retain their shape and use Math-Verify with gold first. Run symbolic
work in the killable worker provided by fourway.scoring.
"""
import re
from fractions import Fraction
from functools import lru_cache

from evaluation.evaluators.paired_scoring import extract_final

SCORER_VERSION = "fourway-math-v5-20260914"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT = r"(?:square\s+|cubic\s+|sq\.?\s+)?(?:cm|mm|m|km|ft|in|inches|inch|feet|foot|yards?|miles?|mph|hours?|minutes?|min|seconds?|days?|pounds?|dollars?|USD|euros?|cents?|degrees?|units?|centimeters?|gallons?|acres?|values|ways|pieces|boxes|dogs|daps|right triangles|distinct integers)(?:\s+squared|\s+cubed)?"
_UNITS = _UNIT + r"(?:\s+per\s+" + _UNIT + r")?"


class InvalidAnswer(ValueError):
    pass


def normalize_answer(value):
    value = str(value).strip().replace("−", "-")
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if value.startswith(left) and value.endswith(right):
            value = value[len(left):-len(right)].strip()
            break
    # Sentence punctuation, not a decimal point in the middle of a number.
    value = value.rstrip(".").strip()
    value = re.sub(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°", "", value)
    value = re.sub(r"^\\approx\s*", "", value)
    value = re.sub(r"^\\?\$", "", value)
    value = re.sub(r"(?<!\\)\\ ", " ", value)
    value = re.sub(r"\s*\\(?:text|mathrm|mbox|textnormal)\{\s*" + _UNITS + r"\.?\s*\}(?:\^\{?[23]\}?)?$", "", value)
    value = re.sub(r"(?<=\d)\s+(?:" + _UNITS + r")(?:\^\{?[23]\}?)?$", "", value)
    value = value.replace(r"\!", "").replace(r"\,", "")
    value = re.sub(r"\\(?:dfrac|tfrac)\b", lambda m: r"\frac", value)
    value = re.sub(r"\\frac\s*\\pi\s+(\d)", lambda m: r"\frac{\pi}{" + m[1] + "}", value)
    # \phantom - is a hidden alignment sign in the supplied matrix labels.
    value = re.sub(r"\\phantom\s*-", "", value)
    def thousands(part):
        return part.replace(",", "") if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", part) else part
    value = re.sub(r"\\frac\{([\d,+-]+)\}\{([\d,+-]+)\}",
                   lambda m: r"\frac{" + thousands(m[1]) + "}{" + thousands(m[2]) + "}", value)
    value = re.sub(r"(" + _NUMBER + r")\s*\\?%", lambda m: r"\frac{" + m[1] + "}{100}", value)
    ordinal = re.fullmatch(r"(\d+)\^\{\\(?:mathrm|text)\{(?:st|nd|rd|th)\}\}\s*(?:\\text\{\s*grade\s*\})?", value)
    if ordinal:
        value = ordinal[1]
    # Some supplied labels state the same scalar twice: 0.56 = 56\%.
    # Collapse only a numerically verified identity, never a general equation.
    sides = value.split("=")
    if len(sides) == 2:
        left, right = (simple_number(s.strip()) for s in sides)
        if left is not None and right is not None and left == right:
            value = sides[-1].strip()
    return value.strip()


def simple_number(value):
    """No eval, no floats, no relative tolerance, no commas/containers erased."""
    if re.fullmatch(_NUMBER, value):
        return Fraction(value)
    frac = re.fullmatch(r"(" + _NUMBER + r")\s*/\s*(" + _NUMBER + r")", value)
    if not frac:
        frac = re.fullmatch(r"\\frac\{(" + _NUMBER + r")\}\{(" + _NUMBER + r")\}", value)
    if not frac:
        # TeX's unbraced single-digit form denotes the same exact fraction.
        frac = re.fullmatch(r"\\frac\s*(\d)\s*(\d)", value)
    if frac:
        denominator = Fraction(frac[2])
        if not denominator:
            raise InvalidAnswer("zero denominator")
        return Fraction(frac[1]) / denominator
    return None


@lru_cache(maxsize=1024)
def parse_answer(value):
    from latex2sympy2_extended import latex2sympy, NormalizationConfig
    from latex2sympy2_extended.latex2sympy2 import ConversionConfig
    from sympy import log, preorder_traversal
    try:
        parsed = latex2sympy(value, normalization_config=NormalizationConfig(
            basic_latex=True, units=False, malformed_operators=True, nits=True,
            boxed="none", equations=False),
            conversion_config=ConversionConfig(lowercase_symbols=False))
        for node in preorder_traversal(parsed):
            if getattr(node, "func", None) == log and len(node.args) == 2 and node.args[1] in (0, 1):
                raise InvalidAnswer("undefined logarithm base")
        return parsed
    except Exception as exc:
        from sympy.matrices.exceptions import MatrixError
        # Grammar rejection is a malformed answer, not an infrastructure fault.
        # Unknown library errors still propagate and stop the run after retries.
        if (isinstance(exc, (MatrixError, ZeroDivisionError))
                or str(exc) == "mismatched dimensions"
                or str(exc).startswith(("I expected", "I don't understand", "I cannot", "I can't", "missing "))):
            raise InvalidAnswer(str(exc)) from exc
        raise


def literal_choice(value):
    value = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", value).strip().rstrip(",")
    match = re.fullmatch(r"\(?([A-Z])\)?", value)
    return match[1] if match else None


def literal_text(value):
    match = re.fullmatch(r"\\(?:text|mbox|mathrm)\{([^{}]+)\}", value)
    if match:
        return " ".join(match[1].split()).casefold()
    return None


def base_notation(value):
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", value)
    match = re.fullmatch(r"([0-9A-Z]*[0-9][0-9A-Z]*)_\{?(\d+)\}?", value)
    return (match[1], match[2]) if match else None


def clock_value(value, reference=False):
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", value)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", value, re.I)
    if not match:
        return None
    hour, minute, second = int(match[1]), int(match[2]), int(match[3] or 0)
    marker = (match[4] or "").replace(".", "").lower()
    if hour > 23 or minute > 59 or second > 59 or (marker and not 1 <= hour <= 12):
        # E.g. the supplied label 19:60 is a ratio, not a clock time.
        return None
    if reference and not (marker or match[3] is not None or match[1].startswith("0") or minute == 0):
        # Without clock evidence, a label such as 9:24 remains a ratio. The
        # supplied time labels have zero padding, seconds, or an AM/PM marker.
        return None
    if marker:
        hour = hour % 12 + (12 if marker == "pm" else 0)
    return hour, minute, second


def scalar_rhs(value):
    # Only scalar targets accept the RHS of an assignment/calculation. Relations,
    # answer sets, coordinates and matrices are never flattened to their tail.
    if re.search(r"[,;<>]", value) or re.search(r"\\(?:leq?|geq?|ne|neq|in)\b|\b(?:or|and)\b", value):
        return value
    return re.split(r"(?<![<>=!])=(?!=)", value)[-1].strip()


def _bounded_integer(expr, max_bits=16384):
    """Evaluate an integer exponent without constructing unbounded integers.

    The LaTeX parser leaves powers unevaluated. An exponent such as 2**2008
    is small enough to materialize; the outer 2**(2**2008) is not.
    None means that this conservative fast path cannot handle the expression.
    """
    if expr.is_Integer:
        value = int(expr)
    elif expr.is_Add or expr.is_Mul:
        value = 0 if expr.is_Add else 1
        for arg in expr.args:
            part = _bounded_integer(arg, max_bits)
            if part is None:
                return None
            value = value + part if expr.is_Add else value * part
            if value.bit_length() > max_bits:
                return None
    elif expr.is_Pow:
        base = _bounded_integer(expr.base, max_bits)
        exponent = _bounded_integer(expr.exp, max_bits)
        if base is None or exponent is None:
            return None
        if base in (1, -1):
            return -1 if base == -1 and exponent % 2 else 1
        if exponent < 0 or (base == 0 and exponent == 0):
            return None
        if base == 0:
            return 0
        # Upper bound the result size before invoking Python's integer power.
        if base.bit_length() * exponent > max_bits:
            return None
        value = pow(base, exponent)
    else:
        return None
    return value if value.bit_length() <= max_bits else None


def _rational_residue(expr, modulus):
    """Map rational arithmetic to residues, or return None if undefined.

    Only integer powers are supported. Inverses must exist modulo the chosen
    modulus; a zero denominator residue is inconclusive, never a wrong answer.
    No symbolic reconstruction or floating point approximation is involved.
    """
    if expr.is_Rational:
        denominator = int(expr.q) % modulus
        if denominator == 0:
            return None
        return (int(expr.p) % modulus) * pow(denominator, -1, modulus) % modulus
    if expr.is_Add or expr.is_Mul:
        value = 0 if expr.is_Add else 1
        for arg in expr.args:
            part = _rational_residue(arg, modulus)
            if part is None:
                return None
            value = (value + part if expr.is_Add else value * part) % modulus
        return value
    if expr.is_Pow:
        exponent = _bounded_integer(expr.exp)
        if exponent is None:
            return None
        base = _rational_residue(expr.base, modulus)
        if base is None or (base == 0 and exponent <= 0):
            return None
        return pow(base, exponent, modulus)
    return None


def _provably_not_integer(expr, reference):
    # Different residues prove inequality over the rationals. Matching residues
    # do NOT prove equality: retain the existing exact symbolic comparison.
    for modulus in (1000000007, 1000000009, 998244353):
        residue = _rational_residue(expr, modulus)
        if residue is not None and residue != reference % modulus:
            return True
    return False


def _integer_equal_without_expanding_powers(target, reference):
    """Prove identities with large numeric powers treated as formal symbols.

    Repeated, identical nonzero powers may cancel. A nonzero symbolic result is
    inconclusive because different powers can be related; never mark it wrong.
    """
    from sympy import Dummy, Integer, simplify
    from rgrpo.math_score_light import MathScoreTimeout

    symbols = {}

    def rewrite(expr):
        if expr.is_Pow:
            base = _bounded_integer(expr.base)
            exponent = _bounded_integer(expr.exp)
            if base is not None and abs(base) > 1 and exponent is not None:
                if abs(base).bit_length() * abs(exponent) > 16384:
                    key = (abs(base), exponent)
                    symbol = symbols.setdefault(key, Dummy(positive=True))
                    return -symbol if base < 0 and exponent % 2 else symbol
        if not expr.args:
            return expr
        args = tuple(rewrite(arg) for arg in expr.args)
        # Preserve untouched unevaluated trees: reconstruction can itself expand
        # a power, even before simplify is called.
        return expr.func(*args) if args != expr.args else expr

    rewritten = rewrite(target)
    if symbols:
        difference = simplify(rewritten - Integer(reference))
        if difference == 0:
            return True
        if not difference.free_symbols:
            return False
        raise MathScoreTimeout("Large-power identity unresolved without integer expansion")
    return bool(simplify(target - Integer(reference)) == 0)


def compare(pred, gold):
    from sympy import Expr
    from math_verify import verify
    if not pred or len(str(pred)) > 2048:
        raise InvalidAnswer("empty or overlong final answer")
    try:
        gold = normalize_answer(gold)
        gnum = simple_number(gold)
    except InvalidAnswer as exc:
        raise ValueError(f"Invalid reference answer: {exc}") from exc
    pred = normalize_answer(pred)
    if not gold:
        raise ValueError("Empty reference answer")
    if not pred or len(pred) > 2048:
        raise InvalidAnswer("empty or overlong final answer")
    if pred == gold:
        return True
    clock = clock_value(gold, reference=True)
    if clock is not None:
        return clock_value(pred) == clock
    base = base_notation(gold)
    if base:
        p = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", pred)
        return base_notation(pred) == base or p == base[0]
    if base_notation(pred):
        return False
    choice = literal_choice(gold)
    if choice:
        return literal_choice(pred) == choice
    literal = literal_text(gold)
    if literal is not None:
        candidate = literal_text(pred) or " ".join(pred.split()).casefold()
        if re.fullmatch(r"[a-z](?:\s*,\s*[a-z])+", literal):
            return set(re.split(r"\s*,\s*", candidate)) == set(re.split(r"\s*,\s*", literal))
        if literal == "four":
            return candidate in {"four", "4"}
        if literal in {"empty", "empty set", "no solution", "no solutions"}:
            return candidate in {literal, "empty", "empty set", "no solution", "no solutions",
                                 r"\emptyset", r"\varnothing", r"\{\}", "{}"}
        return candidate == literal
    scalar = scalar_rhs(pred) if gnum is not None else pred
    pnum = simple_number(scalar)
    if gnum is not None and pnum is not None:
        # Decimal approximations of non-integer answers get an absolute 1e-8
        # tolerance. Integer labels and two exact fractions stay exact.
        return pnum == gnum or bool(gnum.denominator != 1 and "." in scalar
                                   and abs(pnum - gnum) <= Fraction(1, 100000000))
    target = parse_answer(scalar)
    if gnum is not None and gnum.denominator == 1:
        if not isinstance(target, Expr) or target.free_symbols:
            return False
        if _provably_not_integer(target, gnum.numerator):
            return False
        return _integer_equal_without_expanding_powers(target, gnum.numerator)
    try:
        reference = parse_answer(gold)
    except InvalidAnswer as exc:
        raise ValueError(f"Invalid reference answer: {exc}") from exc
    return bool(verify(reference, target, float_rounding=8, timeout_seconds=3, raise_on_error=True))


def score_sample(job):
    from math_verify.errors import TimeoutException
    from rgrpo.math_score_light import _score_alarm, MathScoreTimeout
    text, gold, dataset = job
    result = dict(extracted=None, extraction_method="missing", correct=0,
                  scorer_version=SCORER_VERSION, score_status="ok")
    try:
        # Unlike nested signal.alarm calls in parsers, the parent worker timeout
        # in fourway.scoring also covers normalization and symbolic parsing.
        with _score_alarm(8):
            pred, method = extract_final(str(text), dataset)
            result.update(extracted=pred, extraction_method=method)
            if pred is None:
                result["score_status"] = "extraction_failed"
                return result
            if dataset == "gpqa":
                result["correct"] = int(pred == str(gold).strip().upper())
                return result
            result["correct"] = int(compare(pred, str(gold)))
    except InvalidAnswer as exc:
        result.update(score_status="invalid_answer", score_error=str(exc)[:300])
    except (Exception, TimeoutException) as exc:
        status = "timeout" if isinstance(exc, (TimeoutException, MathScoreTimeout)) else "exception"
        result.update(score_status=status, score_error=f"{type(exc).__name__}: {exc}")
    return result
