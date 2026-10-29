"""Auditable answer-only scoring for the paired Qwen3 benchmark."""
import math
import re
from functools import lru_cache
from collections import defaultdict

from rgrpo import math_score_light as math_utils

REFLECTION = re.compile(r'<reflection\b[^>]*>.*?</reflection>', re.I | re.S)
SCORER_VERSION = 'paired-v4-structured-math-mcq-20260912'


@lru_cache(maxsize=4096)
def structured_math_equal(pred, gold):
    # Do not flatten matrices, intervals, or answer lists to a floating scalar.
    # Preserve base notation: generic LaTeX parsers may discard its subscript.
    if math_utils._has_base_notation(pred) or math_utils._has_base_notation(gold):
        p, g = math_utils._normalize_base_notation(pred), math_utils._normalize_base_notation(gold)
        expected = re.fullmatch(r'(\d+)_(\d+)', g)
        # In an explicitly requested base, plain output digits may omit its
        # redundant subscript; an explicitly different base is still wrong.
        return p == g or bool(expected and p == expected.group(1))
    def ordinal(value):
        m = re.fullmatch(r'(\d+)\^\{\\(?:mathrm|text)\{(?:st|nd|rd|th)\}\}\s*(?:\\text\{\s*grade\s*\})?', value)
        if m:
            return m.group(1)
        percentage = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%', value)
        return '\\frac{' + percentage.group(1) + '}{100}' if percentage else value
    from math_verify import parse, verify, LatexExtractionConfig
    config = [LatexExtractionConfig()]
    reference = parse('\\boxed{' + ordinal(gold) + '}', extraction_config=config,
                      parsing_timeout=2, raise_on_error=True)
    prediction = parse('\\boxed{' + ordinal(pred) + '}', extraction_config=config,
                       parsing_timeout=2, raise_on_error=True)
    return bool(reference and prediction and verify(reference, prediction, float_rounding=8,
                timeout_seconds=2, raise_on_error=True))


def answer_text(text):
    # A diagnosis may quote the wrong boxed answer; it is not a final answer.
    matches = list(REFLECTION.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    if re.search(r'<reflection\b[^>]*>', text, re.I):
        return ''
    return text


def boxed_candidates(text):
    out = []
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start, depth, pos = match.end(), 1, match.end()
        while pos < len(text):
            if text[pos] == '\\' and pos + 1 < len(text) and text[pos + 1] in '{}':
                pos += 2
                continue
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
                if depth == 0:
                    out.append((match.start(), text[start:pos].strip(), 'boxed'))
                    break
            pos += 1
    return out


def extract_final(text, dataset):
    text = answer_text(text)
    candidates = boxed_candidates(text)
    for m in re.finditer(r'''(?im)^[ \t]*(?:\#{1,6}[ \t]*)?['"`]?(?:\*\*)?(?:the[ \t]+)?(?:final[ \t]+)?answer(?:\*\*)?[ \t]*(?::|is\b)[ \t]*(.+?)[ \t]*$''', text):
        value = m.group(1).strip().strip('*').strip().strip("'\"`").strip()
        # A box on an answer line is already parsed with balanced braces.
        if not re.search(r'\\(?:boxed|fbox)', value):
            candidates.append((m.start(), value, 'answer_line'))
    if not candidates:
        return None, 'missing'
    position, value, method = max(candidates, key=lambda c: c[0])
    if any(m.start() > position for m in re.finditer(r'\\(?:boxed|fbox)\s*\{', text)):
        return None, 'incomplete_final_box'
    if dataset == 'gpqa':
        value = re.sub(r'\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}', r'\1', value)
        value = value.strip().strip("'\"`").strip()
        value = re.sub(r'(?i)^\s*answer\s*:\s*', '', value)
        # Accept A, (A), $A$, Answer: A. Reject words beginning with A-D.
        m = re.fullmatch(r'[\s$*]*(?:\()?([A-D])(?:\))?[\s$*]*[.,!]?', value, re.I)
        if not m:
            # An explicit option label remains unambiguous with its option text.
            # Require punctuation after the letter, so "Because" is never B.
            m = re.match(r'^[\s$*]*\(?([A-D])\)?[\s$*]*[).:\-–—]\s*\S', value, re.I)
        return (m.group(1).upper(), method) if m else (None, 'invalid_choice')
    return value, method


def score_sample(job):
    text, gold, dataset = job
    pred, method = extract_final(text, dataset)
    result = {'extracted': pred, 'extraction_method': method, 'correct': 0, 'scorer_version': SCORER_VERSION,
              'score_status': 'ok'}
    if pred is None:
        result['score_status'] = 'extraction_failed'
        return result
    if dataset == 'gpqa':
        result['correct'] = int(pred == str(gold).strip().upper())
        return result
    # Local AIME25 contains a gold angle written as 336^\circ.
    # Strip the complete degree suffix, including its caret.
    def units(value):
        return re.sub(r'\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°', '', str(value)).strip()
    pred, gold = units(pred), units(gold)
    try:
        with math_utils._score_alarm(5):
            if pred == gold:
                result['correct'] = 1
            elif math_utils.pathological_answer(pred, gold):
                result['score_status'] = 'invalid_answer'
            elif dataset in {'amc23', 'aime24', 'aime25'}:
                # No relative tolerance for integer-answer competition datasets.
                # Models also box a calculation ("9 + 4 = 13") or assignment.
                # The final RHS is their answer; a SymPy Equality is not a scalar.
                scalar = re.split(r'(?<![<>=!])=(?!=)', pred)[-1].strip()
                expr = math_utils._parse_symbolic_expr(scalar)
                from sympy import Expr, Integer, simplify
                if not isinstance(expr, Expr) or expr.free_symbols:
                    result['score_status'] = 'invalid_answer'
                else:
                    result['correct'] = int(simplify(expr - Integer(str(gold))) == 0)
            else:
                result['correct'] = int(structured_math_equal(pred, gold))
    except math_utils.MathScoreTimeout:
        result['score_status'] = 'timeout'
    except Exception as exc:
        result['score_status'] = 'exception'
        result['score_error'] = f'{type(exc).__name__}: {exc}'
    return result


def summarize(samples, expected_questions, n=4):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['uid']].append(sample)
    if len(groups) != expected_questions:
        raise ValueError('Missing or duplicate question identities')
    if any(len(g) != n or {s['sample_index'] for s in g} != set(range(n)) for g in groups.values()):
        raise ValueError('Every question must have exactly n distinct samples')
    if any(s['score_status'] in {'timeout', 'exception'} for s in samples):
        raise ValueError('Judge errors must be resolved before publishing scores; see samples')
    return {
        'n_questions': len(groups), 'n_samples_per_q': n, 'n_samples': len(samples),
        'pass_at_n': sum(any(s['correct'] for s in g) for g in groups.values()) / len(groups),
        'mean_at_n': sum(s['correct'] for s in samples) / len(samples),
        'reflection_trigger_rate': sum(s['has_reflection_open'] for s in samples) / len(samples),
        'complete_reflection_rate': sum(s['complete_reflections'] > 0 for s in samples) / len(samples),
        'truncated_rate': sum(s['finish_reason'] == 'length' for s in samples) / len(samples),
        'any_round_truncated_rate': sum(any(r['finish_reason'] == 'length' for r in s.get('rounds', [])) for s in samples) / len(samples),
        'avg_n_iters': sum(s.get('n_iters', 1) for s in samples) / len(samples),
        'extraction_failure_rate': sum(s['extracted'] is None for s in samples) / len(samples),
        'avg_decoded_tokens': sum(s['decoded_tokens'] for s in samples) / len(samples),
    }
