import json
from pathlib import Path

import pytest

from fourway import queue
from fourway.contracts import atomic_json, file_sha


@pytest.mark.parametrize("fail_at", [None, "pag"])
@pytest.mark.parametrize("methods", [None, ["grpo", "pag", "score"]])
def test_queue_order_and_failure_stops_later_methods(tmp_path, monkeypatch, fail_at, methods):
    order = methods or list(queue.ORDER)
    config = tmp_path / "config.json"
    atomic_json(config, {"output": str(tmp_path), "python": "python", "gpus": "0,1",
                         "reflection_repo": str(tmp_path), "steps": 500})
    atomic_json(tmp_path / "preparation/gpu_preflight.json", {
        "config_sha256": file_sha(config), "methods": {
            f"{m}.stage{s}": {"status": "passed"}
            for m in order for s in ([1, 2] if m == "score" else [1])}})
    monkeypatch.setattr(queue.sys, "argv", ["queue", "--config", str(config)] + (["--methods", *methods] if methods else []))
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    calls = []

    class Child:
        pid = 123

        def __init__(self, cmd, **kwargs):
            self.method = cmd[cmd.index("--method") + 1]
            assert kwargs["start_new_session"]
            calls.append(self.method)

        def wait(self):
            if self.method == fail_at:
                return 7
            run = tmp_path / "runs" / self.method
            checkpoint = run / "global_step_500"
            checkpoint.mkdir(parents=True)
            for name in ["latest", "best"]:
                atomic_json(run / f"{name}.json", {"step": 500, "checkpoint": str(checkpoint)})
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    if fail_at:
        with pytest.raises(RuntimeError, match="later methods remain pending"):
            queue.main()
        assert calls == ["grpo", "pag"]
    else:
        queue.main()
        assert calls == order
    state = json.loads((tmp_path / "queue/status.json").read_text())
    assert state["status"] == ("failed" if fail_at else "complete")
    assert state["order"] == order
    assert set(state["methods"]) == set(order)
    if fail_at:
        assert state["methods"]["grpo"]["status"] == "complete"
        assert state["methods"]["pag"]["status"] == "failed"
        assert state["methods"]["score"]["status"] == "pending"
        if "reflect" in order:
            assert state["methods"]["reflect"]["status"] == "pending"
