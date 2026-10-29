"""Run fresh no-update GPU probes, then launch the authorized training queue."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('/root/reflection_grpo_clean')
CONFIG = REPO / 'experiments/four_methods/config.math7500.json'
ROOT = Path('/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914')
sys.path.insert(0, str(REPO))
from fourway.contracts import atomic_json, file_sha


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    state = dict(pid=os.getpid(), started_at=now(), status='preflight', methods=['grpo', 'pag', 'score'])
    destination = ROOT / 'preparation/restart_status.json'
    atomic_json(destination, state)
    try:
        for method in state['methods']:
            state.update(current_method=method)
            atomic_json(destination, state)
            print(f'{now()} GPU_PREFLIGHT {method}', flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'fourway.cli', 'preflight', '--gpu',
                            '--method', method, '--config', str(CONFIG)], cwd=REPO,
                           stdin=subprocess.DEVNULL, check=True)
        queue = ROOT / 'queue'
        queue.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'fourway.queue', '--config', str(CONFIG),
                   '--methods', 'grpo', 'pag', 'score']
        with (queue / 'queue.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (queue / 'pid').write_text(str(child.pid) + '\n')
        atomic_json(queue / 'launch.json', dict(pid=child.pid, started_at=now(), command=command,
                    config_sha256=file_sha(CONFIG), start_new_session=True, stdin='DEVNULL',
                    stdout=str(queue / 'queue.log'), order=state['methods'], gpus='0,1'))
        state.update(status='queue_started', queue_pid=child.pid, completed_at=now())
        atomic_json(destination, state)
        print(f'{now()} QUEUE_STARTED pid={child.pid}', flush=True)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', ended_at=now())
        atomic_json(destination, state)
        raise


if __name__ == '__main__':
    main()
