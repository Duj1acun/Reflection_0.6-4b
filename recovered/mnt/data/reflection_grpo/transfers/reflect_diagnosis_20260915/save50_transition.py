"""Apply the user's save-frequency change by resuming a complete checkpoint.

Only save_frequency changes in the experiment configuration. Native resume
restores actor, Adam, scheduler, RNG and dataloader state; no fresh-run fallback.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback

BUNDLE = Path(__file__).resolve().parent
OLD_CONFIG = BUNDLE / "config.initial-save5.json"
NEW_CONFIG = BUNDLE / "config.save50.json"
STEP = 5

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_configs():
    old = json.loads(OLD_CONFIG.read_text())
    new = json.loads(NEW_CONFIG.read_text())
    delta = {k: [old.get(k), new.get(k)] for k in set(old) | set(new) if old.get(k) != new.get(k)}
    assert delta == {"save_frequency": [5, 50]}, delta
    sys.path.insert(0, new["reflection_repo"])
    from fourway.configuration import build
    from omegaconf import OmegaConf
    a, b = build(old, "reflect"), build(new, "reflect")
    assert a.trainer.save_freq == 5 and b.trainer.save_freq == 50
    a.trainer.save_freq = 50
    assert OmegaConf.to_container(a, resolve=True) == OmegaConf.to_container(b, resolve=True)
    proof = json.loads((Path(old["output"]) / "preparation/gpu_preflight.json").read_text())
    assert proof["config_sha256"] == sha(OLD_CONFIG)
    assert proof["methods"]["reflect.stage1"]["status"] == "passed"
    return new, b

def checkpoint_ready(run):
    cp = run / f"global_step_{STEP}"
    required = ["data.pt", "judge_state.json", "actor/model_world_size_1_rank_0.pt",
                "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt"]
    if not all((cp / p).is_file() and (cp / p).stat().st_size > 0 for p in required):
        return False
    try:
        latest = json.loads((run / "latest.json").read_text())
        return latest["step"] == STEP and int((run / "latest_checkpointed_iteration.txt").read_text()) == STEP
    except (OSError, ValueError, KeyError):
        return False

def live_session(sid):
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / "stat").read_text().split(")")[-1].split()
            if int(fields[3]) == sid and fields[0] != "Z":
                result.append(int(p.name))
        except (OSError, ValueError, IndexError):
            pass
    return result

def taskrunner_log(c):
    texts = []
    for p in (Path(c["ray_tmp_dir"]) / "session_latest/logs").glob("worker-*.out"):
        text = p.read_text(errors="replace")
        if "TaskRunner" in text[:500]:
            texts.append(text)
    return "\n".join(texts)

def child():
    c, cfg = check_configs()
    from omegaconf import OmegaConf
    run = Path(c["output"]) / "runs/reflect"
    assert checkpoint_ready(run), "Complete resume checkpoint required"
    assert json.loads((BUNDLE / "config.json").read_text()) == c
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = str(run / f"global_step_{STEP}")
    cfg.trainer.val_before_train = False  # Completed initial validation is retained.
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 50
    assert cfg.trainer.total_training_steps == 500
    assert cfg.trainer.del_local_ckpt_after_load is False
    write_json(run / "stage1.resumed-save50.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    write_json(run / "stage1.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    print("SAVE50_NATIVE_RESUME " + json.dumps({"from_step": STEP, "next_update": STEP + 1,
          "save_freq": 50, "test_freq": 50, "total_steps": 500}), flush=True)
    from verl.trainer.main_ppo import run_ppo
    try:
        run_ppo(cfg)
    finally:
        import ray
        ray.shutdown()

def controller():
    c, _ = check_configs()
    out = Path(c["output"])
    run = out / "runs/reflect"
    old_status = json.loads((BUNDLE / "SERVER_STATUS.json").read_text())
    old_pid = int(old_status["launcher_pid"])
    assert old_pid == 148069 and os.getsid(old_pid) == old_pid
    assert os.getsid(os.getpid()) != old_pid
    audit = {"requested_save_frequency": 50, "old_save_frequency": 5,
             "status": "waiting_for_transition_checkpoint", "checkpoint_step": STEP,
             "controller_pid": os.getpid(), "old_launcher_pid": old_pid,
             "old_config_sha256": sha(OLD_CONFIG), "new_config_sha256": sha(NEW_CONFIG),
             "preflight_reuse_reason": "Resolved learning/GPU configuration identical; only trainer.save_freq differs."}
    def audit_state(stage, **extra):
        audit.update(status=stage, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        write_json(BUNDLE / "SAVE_FREQUENCY_CHANGE.json", audit)
        print(stage, flush=True)
    audit_state("checking_unchanged_inputs")
    from fourway.cli import preflight, environment
    preflight(c)
    audit_state("waiting_for_transition_checkpoint")
    deadline = time.monotonic() + 3600
    while not (checkpoint_ready(run) and re.search(r'step:5 - .*branch/optimizer_steps:1', taskrunner_log(c))):
        if time.monotonic() > deadline:
            raise TimeoutError("No complete transition checkpoint; training left untouched")
        if not live_session(old_pid):
            raise RuntimeError("Original training stopped before a complete checkpoint")
        time.sleep(3)
    audit_state("stopping_at_saved_checkpoint", checkpoint=str(run / f"global_step_{STEP}"))
    old_taskrunner = taskrunner_log(c)
    (run / "stage1.before_save50.taskrunner.log").write_text(old_taskrunner)
    write_json(BUNDLE / "SERVER_STATUS.json", dict(old_status, launcher_pid=os.getpid(),
               stage="reconfiguring", previous_launcher_pid=old_pid, requested_save_frequency=50))
    # Every process in this SID belongs to this detached experiment.
    os.killpg(old_pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while live_session(old_pid) and time.monotonic() < deadline:
        time.sleep(1)
    for pid in live_session(old_pid):
        try:
            if os.getsid(pid) == old_pid:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while live_session(old_pid) and time.monotonic() < deadline:
        time.sleep(1)
    assert not live_session(old_pid), "Old workers still alive; refusing a duplicate GPU job"
    assert checkpoint_ready(run)
    write_json(BUNDLE / "config.json", c)
    for name in ["stage1.resolved.json", "fourway_run.json"]:
        p = run / name
        if p.exists():
            (run / (name + ".initial-save5")).write_bytes(p.read_bytes())
    write_json(run / "fourway_run.json", {"config": c, "config_sha256": sha(BUNDLE / "config.json"),
               "previous_config_sha256": sha(OLD_CONFIG), "resumed_from_step": STEP,
               "resume_launcher_sha256": sha(Path(__file__).resolve())})
    write_json(run / "resume_save50_audit.json", audit)
    repo, env = environment(c, BUNDLE / "config.json", "reflect")
    lock_path = Path(c["lock_dir"]) / "fourway_gpu1.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(old_status, launcher_pid=os.getpid(), stage="formal_training", error=None,
                      restart_from_sft=False, resumed_from_step=STEP, save_frequency=50,
                      config_sha256=sha(BUNDLE / "config.json"),
                      updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(BUNDLE / "SERVER_STATUS.json", status)
        (BUNDLE / "restart.pid").write_text(str(os.getpid()) + "\n")
        (out / "launcher.pid").write_text(str(os.getpid()) + "\n")
        audit_state("resuming_with_save_frequency_50")
        with (run / "stage1.log").open("a") as log:
            log.write("\nARCHIVED_DIRECT_TASKRUNNER_OUTPUT_BEFORE_SAVE50\n" + old_taskrunner)
            log.write("\nUSER_REQUESTED_SAVE_FREQUENCY_50: resuming complete step5 checkpoint\n")
            log.flush()
            result = subprocess.run([c["python"], "-u", str(Path(__file__).resolve()), "--resume-child"],
                                    cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        status.update(stage="completed" if result.returncode == 0 else "failed",
                      error=None if result.returncode == 0 else f"resume child exit {result.returncode}")
        write_json(BUNDLE / "SERVER_STATUS.json", status)
        audit_state(status["stage"])
        if result.returncode:
            raise RuntimeError(status["error"])

if __name__ == "__main__":
    try:
        if "--resume-child" in sys.argv:
            child()
        elif "--check-config-only" in sys.argv:
            check_configs()
            print("SAVE50_CONFIG_ONLY_DIFF_VERIFIED", flush=True)
        else:
            controller()
    except BaseException:
        traceback.print_exc()
        if "--resume-child" not in sys.argv and "--check-config-only" not in sys.argv:
            p = BUNDLE / "SAVE_FREQUENCY_CHANGE.json"
            if p.exists():
                failure = json.loads(p.read_text())
                failure.update(status="failed", error=traceback.format_exc())
                write_json(p, failure)
            p = BUNDLE / "SERVER_STATUS.json"
            if p.exists():
                failure = json.loads(p.read_text())
                if failure.get("launcher_pid") == os.getpid():
                    failure.update(stage="failed", error="save-frequency transition failed; inspect save50_transition.log")
                    write_json(p, failure)
        raise
