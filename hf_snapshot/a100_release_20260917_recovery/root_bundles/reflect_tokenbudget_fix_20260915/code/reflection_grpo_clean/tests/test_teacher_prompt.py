import json

from scripts.data.prepare_sft_data import _teacher_prompt


def test_teacher_prompt_verifies_first_error_without_forcing_gold():
    row = {"problem": "TEST_PROBLEM", "answer": "TEST_GOLD", "completion": "TEST_STUDENT"}
    prompt = _teacher_prompt(row)
    assert "step by step, in its original order" in prompt
    assert "Do not move\n   conclusions from later steps" in prompt
    assert "first-person" in prompt
    assert 'verified_states: "I have"; error_reason: "I incorrectly"; new_direction: "I will"' in prompt
    assert "A true equation first written after that error is still excluded" in prompt
    assert "Do not silently assume distinctness" in prompt
    assert "insufficient conditions" in prompt
    assert "unquestionable authority" in prompt
    assert "GOLD_ANSWER is authoritative" not in prompt
    assert "Do not spend tokens questioning" not in prompt
    assert "PROBLEM:\nTEST_PROBLEM" in prompt
    assert "GOLD_ANSWER:\nTEST_GOLD" in prompt
    assert "STUDENT_SOLUTION:\nTEST_STUDENT" in prompt
    assert all(ord(c) >= 32 or c in "\n\r\t" for c in prompt)
    example = prompt.split('JSON source string ', 1)[1].split(' must decode', 1)[0]
    assert json.loads(example) == r"\boxed{2}"


def test_teacher_prompt_keeps_retry_feedback_and_prior_analysis():
    prompt = _teacher_prompt(
        {"problem": "P", "answer": "A", "completion": "S"},
        previous_errors=["test_validation_error"],
        previous_analysis="TEST_PRIOR",
    )
    assert "test_validation_error" in prompt
    assert "PRIOR_ANALYSIS:\nTEST_PRIOR\nEND_PRIOR_ANALYSIS" in prompt
