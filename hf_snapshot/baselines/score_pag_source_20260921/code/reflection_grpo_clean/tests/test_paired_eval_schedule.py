import json
import pytest
from evaluation.launchers import supervise_paired_eval as scheduler


@pytest.mark.parametrize('fail_first', [False, True])
def test_variants_parallel_and_sizes_sequential(tmp_path, monkeypatch, fail_first):
    monkeypatch.setattr(scheduler, 'OUT', tmp_path)
    monkeypatch.setattr(scheduler, 'report', lambda: None)
    monkeypatch.setattr(scheduler.time, 'sleep', lambda _: None)
    for size in ['0.6b', '4b']:
        (tmp_path / f'qwen3-{size}.generation_smoke.json').write_text('[]')
    processes = []

    class Process:
        def __init__(self, command, **kwargs):
            self.size, self.variant = command[2], command[5]
            if self.size == '4b':
                assert len(processes) >= 2
                assert all(p.returncode == 0 for p in processes[:2])
            self.pid = 100 + len(processes)
            self.returncode = None
            self.polls = 0
            processes.append(self)

        def poll(self):
            self.polls += 1
            if self.polls >= 3:
                self.returncode = int(fail_first and self.size == '0.6b' and self.variant == 'base')
            return self.returncode

    monkeypatch.setattr(scheduler.subprocess, 'Popen', Process)
    if fail_first:
        with pytest.raises(RuntimeError, match='next size not launched'):
            scheduler.main()
        assert len(processes) == 2
    else:
        scheduler.main()
        assert [(p.size, p.variant) for p in processes] == [
            ('0.6b', 'base'), ('0.6b', 'phase2'), ('4b', 'base'), ('4b', 'phase2')]
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['qwen3-4b-base']['status'] == ('blocked_by_previous_stage_failure' if fail_first else 'completed')
