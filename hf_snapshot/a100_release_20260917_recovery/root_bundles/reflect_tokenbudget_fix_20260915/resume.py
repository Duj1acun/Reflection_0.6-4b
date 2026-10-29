"""Check the token-budget repair, then resume REFLECT's complete step-50 state."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback

B = Path(__file__).resolve().parent
C = json.loads((B / "config.json").read_text())
sys.path.insert(0, C["reflection_repo"])
from fourway.contracts import atomic_json, file_sha


def status(stage, **extra):
    p = B / "SERVER_STATUS.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(stage=stage, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    atomic_json(p, d)
    print(stage, flush=True)


def verify():
    from fourway.configuration import build
    from omegaconf import OmegaConf
    old = json.loads((B / "config.before.json").read_text())
    delta = {k: [old.get(k), C.get(k)] for k in set(old) | set(C) if old.get(k) != C.get(k)}
    assert set(delta) == {"tokens_per_gpu", "reflection_repo", "ray_tmp_dir"}, delta
    assert delta["tokens_per_gpu"] == [8192, 16384]
    a, b = build(old, "reflect"), build(C, "reflect")
    a.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 16384
    a.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = 16384
    a.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu = 16384
    a.ray_init._temp_dir = b.ray_init._temp_dir
    assert OmegaConf.to_container(a, resolve=True) == OmegaConf.to_container(b, resolve=True)
    cp = Path(C["output"]) / "runs/reflect/global_step_50"
    required = ["data.pt", "judge_state.json", "actor/model_world_size_1_rank_0.pt",
                "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt"]
    assert all((cp / n).is_file() and (cp / n).stat().st_size > 0 for n in required)
    # Deserialize scheduler/RNG/data state; the native loader restores the large model and Adam tensors.
    import torch
    extra = torch.load(cp / required[-1], map_location="cpu", weights_only=False)
    data_state = torch.load(cp / "data.pt", map_location="cpu", weights_only=False)
    assert "lr_scheduler" in extra and "rng" in extra, extra.keys()
    atomic_json(B / "CONFIG_VERIFIED.json", {"delta": delta, "checkpoint": str(cp),
        "checkpoint_files": {n: (cp / n).stat().st_size for n in required},
        "extra_state_keys": list(extra), "data_state_type": type(data_state).__name__,
        "config_sha256": file_sha(B / "config.json")})
    return b, cp


def child():
    from omegaconf import OmegaConf
    cfg, cp = verify()
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = str(cp)
    cfg.trainer.val_before_train = False
    assert cfg.trainer.total_training_steps == 500
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 50
    assert cfg.trainer.del_local_ckpt_after_load is False
    run = Path(cfg.trainer.default_local_dir)
    atomic_json(run / "stage1.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    print("TOKEN_BUDGET_NATIVE_RESUME from_step=50 next_update=51 total_steps=500 tokens_per_gpu=16384", flush=True)
    from verl.trainer.main_ppo import run_ppo
    try:
        run_ppo(cfg)
    finally:
        import ray
        ray.shutdown()


def controller():
    from fourway.cli import environment
    run = Path(C["output"]) / "runs/reflect"
    with (Path(C["lock_dir"]) / "fourway_gpu1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status("config_check", launcher_pid=os.getpid(), detached=True, gpu="1", output=C["output"],
               resumed_from_step=50, save_frequency=50, config_sha256=file_sha(B / "config.json"), error=None)
        verify()
        repo, env = environment(C, B / "config.json", "reflect")
        env.pop("FOURWAY_LONG_SEQUENCE_PROBE", None)
        status("cpu_preflight")
        with (B / "cpu_preflight.log").open("w") as log:
            r = subprocess.run([C["python"], "-m", "fourway.cli", "preflight", "--config", str(B / "config.json")],
                               cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert r.returncode == 0, "CPU preflight failed"
        status("gpu_preflight", regression="8710 and 16384 token forward/backward, then 64 x 8 native batch")
        probe_env = dict(env, FOURWAY_PROBE="1", FOURWAY_LONG_SEQUENCE_PROBE="1")
        with (B / "gpu_preflight.log").open("w") as log:
            r = subprocess.run([C["python"], "-m", "fourway.entry", "--method", "reflect", "--probe"],
                               cwd=repo, env=probe_env, stdout=log, stderr=subprocess.STDOUT)
        assert r.returncode == 0, "GPU preflight failed"
        evidence = (B / "gpu_preflight.log").read_text(errors="replace")
        for p in (Path(C["ray_tmp_dir"]) / "session_latest/logs").glob("worker-*.out"):
            evidence += p.read_text(errors="replace")
        assert "FOURWAY_PROBE_COMPLETED" in evidence and "FOURWAY_NO_UPDATE_PROBE" in evidence
        for n in [8710, 16384]:
            assert f"'length': {n}" in evidence, f"Missing long-sequence proof: {n}"
        proofs = [l for l in evidence.splitlines() if any(m in l for m in
                  ["TOKEN_BUDGET_LONG_GPU_PASSED", "TOKEN_BUDGET_OLD_FAILURE_REPRODUCED", "FOURWAY_NO_UPDATE_PROBE"])]
        atomic_json(B / "GPU_VERIFIED.json", {"config_sha256": file_sha(B / "config.json"), "proofs": proofs})
        atomic_json(Path(C["output"]) / "preparation/gpu_preflight.json", {
            "config_sha256": file_sha(B / "config.json"), "methods": {"reflect.stage1": {
                "status": "passed", "log": str(B / "gpu_preflight.log"), "long_sequence_lengths": [8710, 16384]}}})
        verify()
        archive = run / "before_tokenbudget_fix_20260915"
        archive.mkdir(exist_ok=False)
        for name in ["stage1.log", "stage1.resolved.json", "fourway_run.json"]:
            p = run / name
            if p.exists():
                p.rename(archive / name)
        atomic_json(run / "fourway_run.json", {"config": C, "config_sha256": file_sha(B / "config.json"),
            "resumed_from_step": 50, "previous_run_metadata": str(archive / "fourway_run.json")})
        status("formal_training", gpu_preflight="passed", formal_training_started=True, error=None)
        (B / "restart.pid").write_text(str(os.getpid()) + "\n")
        with (run / "stage1.log").open("w") as log:
            r = subprocess.run([C["python"], "-u", str(B / "resume.py"), "--child"],
                               cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        status("completed" if r.returncode == 0 else "failed", error=None if r.returncode == 0 else f"Training child exit {r.returncode}")


if __name__ == "__main__":
    try:
        child() if "--child" in sys.argv else controller()
    except BaseException:
        traceback.print_exc()
        if "--child" not in sys.argv:
            status("failed", error=traceback.format_exc())
        raise
