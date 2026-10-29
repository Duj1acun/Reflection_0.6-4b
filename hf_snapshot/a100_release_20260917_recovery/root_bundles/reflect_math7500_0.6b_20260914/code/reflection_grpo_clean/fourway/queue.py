"""Explicit sequential launch of selected methods, stopping on any failure."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .contracts import DEFAULT_CONFIG, atomic_json, config, file_sha

ORDER = ("grpo", "pag", "score", "reflect")


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--methods", nargs="+", choices=ORDER, default=list(ORDER),
                        help="Methods in execution order; select grpo pag score for Base baselines only")
    args = parser.parse_args()
    order = args.methods
    if len(set(order)) != len(order):
        parser.error("--methods must not contain duplicates")
    config_path = str(Path(args.config).resolve())
    c = config(config_path)
    digest = file_sha(config_path)
    out = Path(c["output"])
    queue = out / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (queue / "status.json").exists():
        raise RuntimeError("Queue status already exists; refusing duplicate launch")
    preflight = json.loads((out / "preparation/gpu_preflight.json").read_text())
    if preflight["config_sha256"] != digest:
        raise RuntimeError("Configuration changed after GPU preflight")
    for method in order:
        for stage in ([1, 2] if method == "score" else [1]):
            if preflight["methods"].get(f"{method}.stage{stage}", {}).get("status") != "passed":
                raise RuntimeError(f"GPU preflight missing: {method} stage {stage}")
        run = out / "runs" / method
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"Existing run must not be overwritten: {run}")
    state = {"status": "running", "pid": os.getpid(), "started_at": now(),
             "config": config_path, "config_sha256": digest, "gpus": c["gpus"],
             "order": list(order), "current_method": None,
             "methods": {m: {"status": "pending"} for m in order}}
    atomic_json(queue / "status.json", state)
    child = None

    def stopped(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise InterruptedError(f"Queue received signal {signum}")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        for method in order:
            if file_sha(config_path) != digest:
                raise RuntimeError("Configuration changed while queue was running")
            state["current_method"] = method
            item = state["methods"][method]
            item.update(status="running", started_at=now())
            print(f"{now()} START {method} GPUs={c['gpus']}", flush=True)
            child = subprocess.Popen(
                [c["python"], "-u", "-m", "fourway.cli", "train", "--method", method,
                 "--config", config_path], cwd=c["reflection_repo"],
                stdin=subprocess.DEVNULL, start_new_session=True)
            item["launcher_pid"] = child.pid
            atomic_json(queue / "status.json", state)
            code = child.wait()
            item.update(returncode=code, ended_at=now())
            if code:
                raise RuntimeError(f"{method} failed with exit code {code}; later methods remain pending")
            latest = json.loads((out / "runs" / method / "latest.json").read_text())
            best = json.loads((out / "runs" / method / "best.json").read_text())
            if latest["step"] != c["steps"] or not Path(latest["checkpoint"]).is_dir():
                raise RuntimeError(f"{method} exited without the final checkpoint")
            if not Path(best["checkpoint"]).is_dir():
                raise RuntimeError(f"{method} best checkpoint is missing")
            item.update(status="complete", latest=latest["checkpoint"], best=best["checkpoint"])
            atomic_json(queue / "status.json", state)
            print(f"{now()} COMPLETE {method}", flush=True)
        state.update(status="complete", current_method=None, ended_at=now())
        atomic_json(queue / "status.json", state)
    except BaseException as exc:
        method = state["current_method"]
        if method and state["methods"][method]["status"] == "running":
            state["methods"][method].update(status="failed", ended_at=now())
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at=now())
        atomic_json(queue / "status.json", state)
        raise


if __name__ == "__main__":
    main()
