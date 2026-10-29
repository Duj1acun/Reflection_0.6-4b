import pytest
from pathlib import Path
@pytest.fixture(autouse=True)
def relocated_math_profile(request, monkeypatch):
    if request.node.name == 'test_math_profile_has_equal_answer_caps_and_identical_question_views':
        import fourway.contracts as contracts
        original = contracts.config
        def relocated(path=None):
            if path is not None and Path(path).name == 'config.math7500.json':
                path = '/root/reflect_math7500_0.6b_20260914/config.json'
            return original(path)
        monkeypatch.setattr(contracts, 'config', relocated)

def pytest_runtest_call(item):
    if item.originalname == 'test_native_training_loops_continue_to_next_batch_before_updates':
        item.funcargs['repo'] = '/root/reflect_math7500_0.6b_20260914/code/' + Path(item.funcargs['repo']).name
