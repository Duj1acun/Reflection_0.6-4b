import pytest
from evaluation.evaluators.paired_scoring import extract_final, score_sample, summarize


@pytest.mark.parametrize('text,gold,ds,correct', [
    (r'Answer: $B$', 'B', 'gpqa', 1),
    (r'Answer: Because', 'B', 'gpqa', 0),
    (r'Answer: B) 10^-9 eV', 'B', 'gpqa', 1),
    (r'Answer: C) 11', 'C', 'gpqa', 1),
    (r'Answer: (A) 12', 'A', 'gpqa', 1),
    ("'Answer: D'", 'D', 'gpqa', 1),
    ('### Final Answer: D) 2', 'D', 'gpqa', 1),
    ("### Final Answer:\n'Answer: B) 5.0x10^-3 M'", 'B', 'gpqa', 1),
    (r'\boxed{\text{Answer: D}}', 'D', 'gpqa', 1),
    (r'Answer: A' + '\n' + r'\boxed{\text{C}}', 'C', 'gpqa', 1),
    (r'\boxed{\frac{1}{2}}', '0.5', 'math500', 1),
    (r'\boxed{100001}', '100000', 'amc23', 0),
    (r'\boxed{070}', '70', 'aime25', 1),
    (r'\boxed{\frac{140}{2}}', '70', 'aime25', 1),
    (r'<reflection>Wrong: \boxed{3}</reflection>', '3', 'math500', 0),
    (r'\boxed{3}<reflection>That was wrong</reflection>', '3', 'math500', 0),
    (r'<reflection>Wrong: \boxed{3}</reflection>\boxed{4}', '4', 'math500', 1),
    (r'\boxed{3} Later: \boxed{4', '3', 'math500', 0),
    (r'\boxed{336}', r'336^\circ', 'aime25', 1),
    (r'\boxed{(2,\infty)}', r'(2,\infty)', 'math500', 1),
    (r'\boxed{50\%}', '0.5', 'math500', 1),
    (r'\boxed{50\%}', '50', 'math500', 0),
    (r'\boxed{9 + 4 = 13}', '13', 'amc23', 1),
    (r'\boxed{1 + 324 = 325}', '265', 'amc23', 0),
    (r'\boxed{x = 9}', '9', 'aime24', 1),
    (r'\boxed{(1, 2)}', '2', 'amc23', 0),
    (r'\boxed{x < 9}', '9', 'aime25', 0),
    (r'\boxed{\begin{pmatrix}-4\\-12\\-7\end{pmatrix}}', r'\begin{pmatrix}-2\\-14\\-7\end{pmatrix}', 'math500', 0),
    (r'\boxed{[-\sqrt{3},\sqrt{3}]}', r'(-\sqrt{3},\sqrt{3})', 'math500', 0),
    (r'\boxed{(2,12)\cup(12,\infty)}', r'(2,12)\cup(12,102)', 'math500', 0),
    (r'\boxed{3,7,5}', '3,5,7', 'math500', 1),
    (r'\boxed{4210_5}', '4210_{5}', 'math500', 1),
    (r'\boxed{4210_6}', '4210_{5}', 'math500', 0),
    (r'\boxed{4210}', '4210_{5}', 'math500', 1),
    (r'\boxed{\$10}', '10', 'math500', 1),
    (r'\boxed{[-2,7]}', r'x\in[-2,7]', 'math500', 1),
    (r'\boxed{12^{\mathrm{th}} \text{ grade}}', '12', 'math500', 1),
])
def test_extraction_and_judging(text, gold, ds, correct):
    scored = score_sample((text, gold, ds))
    assert scored['score_status'] not in {'timeout', 'exception'}, scored
    assert scored['correct'] == correct, scored


def test_pass4_question_weighting_and_trigger():
    samples = [dict(uid=q, sample_index=i, correct=int(q == 'a' and i == 0),
                    score_status='ok', has_reflection_open=i == 0,
                    complete_reflections=int(i == 0), finish_reason='stop',
                    extracted='1', decoded_tokens=10) for q in ['a', 'b'] for i in range(4)]
    result = summarize(samples, 2)
    assert result['pass_at_n'] == 0.5
    assert result['mean_at_n'] == 0.125
    assert result['reflection_trigger_rate'] == 0.25
    with pytest.raises(ValueError):
        summarize(samples[:-1], 2)
    samples[0]['score_status'] = 'timeout'
    with pytest.raises(ValueError):
        summarize(samples, 2)
