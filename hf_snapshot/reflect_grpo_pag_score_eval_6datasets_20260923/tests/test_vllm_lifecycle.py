import pytest
from rgrpo import vllm_lifecycle as lifecycle


def test_device_quiescence_before_and_after_memory_transition(monkeypatch):
    events=[]
    monkeypatch.setattr(lifecycle.torch.cuda, "synchronize", lambda: events.append("sync"))
    def remap(*, tags):
        events.append(("remap", tags))
        return 17
    assert lifecycle.synchronized_transition(remap, tags=["weights"]) == 17
    assert events == ["sync", ("remap", ["weights"]), "sync"]


def test_prior_cuda_failure_prevents_unmapping(monkeypatch):
    def pending_error():
        raise RuntimeError("prior kernel fault")
    monkeypatch.setattr(lifecycle.torch.cuda, "synchronize", pending_error)
    with pytest.raises(RuntimeError, match="prior kernel fault"):
        lifecycle.synchronized_transition(lambda: pytest.fail("must not unmap"))


def test_transition_error_is_not_masked(monkeypatch):
    events=[]
    monkeypatch.setattr(lifecycle.torch.cuda,"synchronize",lambda:events.append("sync"))
    def broken():raise ValueError("mapping failed")
    with pytest.raises(ValueError,match="mapping failed"):
        lifecycle.synchronized_transition(broken)
    assert events == ["sync"]
