import json
import pytest
from evaluation.evaluators.eval_paired_qwen3 import saved_samples


def test_resume_preserves_complete_question_groups(tmp_path):
    rows = [{'uid':'a','problem':'p','gold':'1'}, {'uid':'b','problem':'q','gold':'2'}]
    records = [{**rows[0], 'sample_index':i} for i in range(4)]
    p = tmp_path / 'samples.jsonl'
    assert saved_samples(p, rows) == []
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert saved_samples(p, rows) == records
    with pytest.raises(ValueError, match='ordered dataset'):
        saved_samples(p, rows[::-1])
    p.write_text(''.join(json.dumps(r)+'\n' for r in records[:3]))
    with pytest.raises(ValueError, match='Incomplete saved question'):
        saved_samples(p, rows)
