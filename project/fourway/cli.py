"""Explicit preparation/preflight/training entrypoints; no implicit training."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .contracts import DEFAULT_CONFIG, METHODS, atomic_json, config, file_sha, generated_budget

def environment(c, path, method, probe=False):
    from .configuration import REFLECT_ENV
    env = os.environ.copy()
    # Old experiment .env files are deliberately not sourced by this launcher.
    for k in list(env):
        if k.startswith(("TRAJ_GRPO_", "HYBRID_V2_", "RAY_")) or k == "PYTORCH_CUDA_ALLOC_CONF":
            env.pop(k, None)
    env.update(REFLECT_ENV if method == "reflect" else {})
    repo = c["pag_repo"] if method in {"pag", "score"} else c["reflection_repo"]
    env.update(FOURWAY_CONFIG=str(Path(path).resolve()), FOURWAY_METHOD=method,
               FOURWAY_PROBE="1" if probe else "0", CUDA_VISIBLE_DEVICES=c["gpus"],
               CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_USE_V1="0", RAY_INIT_ADDRESS="local", RAY_TMPDIR="/mnt/data/tmp/fw",
               RAY_INIT_NUM_GPUS="2", RAY_INCLUDE_DASHBOARD="0",
               PYTHONPATH=os.pathsep.join(dict.fromkeys([repo, c["reflection_repo"]])),
               PATH=str(Path(c["python"]).parent)+os.pathsep+env.get("PATH", ""),
               HF_HOME="/mnt/data/cache/huggingface", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CHAT_TEMPLATE_ENABLE_THINKING="false", OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled")
    return repo, env

def run_child(c, args, method, stage, probe=False):
    import fcntl
    from contextlib import ExitStack
    repo, env = environment(c, args.config, method, probe)
    run = Path(c["output"])/("probes" if probe else "runs")/method
    if probe:
        run = run/f"stage{stage}"
    elif args.command == "evaluate":
        run = Path(c["output"])/"evaluations"/method/Path(args.checkpoint).name
    run.mkdir(parents=True, exist_ok=True)
    logfile = run/f"stage{stage}.log"
    if logfile.exists() and not probe and args.command == "train":
        raise ValueError(f"Existing run log: {logfile}; refusing accidental restart")
    cmd = [c["python"], "-m", "fourway.entry", "--method", method, "--stage", str(stage)]
    if probe:
        cmd += ["--probe"]
    if getattr(args, "checkpoint", None):
        cmd += ["--evaluate", args.checkpoint]
    print("Running", method, "stage", stage, "probe" if probe else args.command, "log:", logfile, flush=True)
    with ExitStack() as stack:
        for gpu in sorted(c["gpus"].split(",")):
            lock_path = Path("/mnt/data/reflection_grpo/locks")/f"fourway_gpu{gpu}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another four-method job owns GPU {gpu}")
        with logfile.open("a" if probe or args.command == "evaluate" else "w") as log:
            result = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{method} stage {stage} failed ({result.returncode}); inspect {logfile}")
    return logfile

def preflight(c):
    from omegaconf import OmegaConf
    from .configuration import build
    import pyarrow.parquet as pq
    import importlib.metadata
    merged = Path(c["output"])/"initialization/reflect_merged/merge_manifest.json"
    if not merged.exists():
        raise ValueError("Run prepare before preflight: merged SFT initialization is missing")
    from .prepare import merge
    merge(c)  # Existing manifest: verify Base, adapter and merged model hashes.
    data = Path(c["output"])/"data"
    manifest = json.loads((data/"manifest.json").read_text())
    if c.get("scorer_version") == "fourway-math-v5-20260914":
        from .audit_math import audit
        audit(c)
    inputs = json.loads((data/"batch_schedule.json").read_text())
    assert len(inputs) == c["steps"] and all(len(x)==c["train_batch_size"] for x in inputs)
    profiles = {}
    for method in METHODS:
        for split in ("train", "validation"):
            item = manifest["methods"][method][split]
            if file_sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Prepared data changed: {item['path']}")
            rows = pq.read_table(item["path"]).to_pylist()
            ids = [r["extra_info"]["index"] for r in rows]
            profiles[(method, split)] = ids
            assert ids == profiles[("grpo", split)]
        generated_budget(c, method)
        for stage in ([1,2] if method == "score" else [1]):
            cfg = build(c, method, stage)
            output = Path(c["output"])/"preparation"/f"{method}.stage{stage}.resolved.json"
            atomic_json(output, OmegaConf.to_container(cfg, resolve=True))
    versions = {k: importlib.metadata.version(k) for k in ["torch", "transformers", "vllm", "math-verify", "latex2sympy2_extended", "torchdata", "peft"]}
    expected = {"torch": "2.6.0", "transformers": "4.51.1", "vllm": "0.8.5", "math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0"}
    for name, v in expected.items():
        if versions[name].split("+")[0] != v:
            raise ValueError(f"Environment mismatch: {name}={versions[name]} != {v}")
    report = {"status": "passed", "versions": versions, "train_questions": len(profiles[("grpo","train")]),
              "validation_questions": len(profiles[("grpo","validation")]), "schedule_steps": len(inputs)}
    atomic_json(Path(c["output"])/"preparation/cpu_preflight.json", report)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "preflight", "dry-run", "train", "evaluate"])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--gpu", action="store_true", help="Preflight only: two-GPU backward probes with optimizer updates disabled")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    if args.checkpoint and args.command != "evaluate":
        p.error("--checkpoint is only supported by evaluate")
    c = config(args.config)
    if args.command == "prepare":
        from .prepare import prepare, merge
        prepare(c)
        merge(c)
        print(preflight(c))
    elif args.command == "dry-run":
        from .configuration import build
        from omegaconf import OmegaConf
        for method in ([args.method] if args.method else METHODS):
            for stage in ([1,2] if method == "score" else [1]):
                print(OmegaConf.to_yaml(build(c, method, stage), resolve=True))
    elif args.command == "preflight":
        print(preflight(c))
        if args.gpu:
            path = Path(c["output"])/"preparation/gpu_preflight.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            reports = old.get("methods", {}) if old.get("config_sha256") == file_sha(args.config) else {}
            for method in ([args.method] if args.method else METHODS):
                for stage in ([1,2] if method == "score" else [1]):
                    log = run_child(c, args, method, stage, probe=True)
                    text = log.read_text()
                    if "FOURWAY_NO_UPDATE_PROBE" not in text or "FOURWAY_PROBE_COMPLETED" not in text:
                        raise RuntimeError(f"Probe did not verify backward/no-update: {log}")
                    reports[f"{method}.stage{stage}"] = {"status": "passed", "log": str(log)}
                    atomic_json(path, {"config_sha256": file_sha(args.config), "methods": reports})
    else:
        if not args.method:
            p.error("train/evaluate require --method; there is no implicit four-task queue")
        preflight(c)
        if args.command == "evaluate" and not args.checkpoint:
            p.error("evaluate requires --checkpoint pointing to global_step_N")
        if args.command == "train":
            report = json.loads((Path(c["output"])/"preparation/gpu_preflight.json").read_text())
            if report["config_sha256"] != file_sha(args.config):
                raise ValueError("Configuration changed since GPU preflight")
            for stage in ([1,2] if args.method == "score" else [1]):
                if report["methods"].get(f"{args.method}.stage{stage}", {}).get("status") != "passed":
                    raise ValueError("Method GPU preflight not passed")
            run = Path(c["output"])/"runs"/args.method
            if run.exists() and any(run.iterdir()):
                raise ValueError(f"Refusing to overwrite existing experiment {run}")
            atomic_json(run/"fourway_run.json", {"config": c, "config_sha256": file_sha(args.config)})
        stages = [1,2] if args.method == "score" and args.command == "train" else [1]
        if args.command == "evaluate":
            import re
            match = re.fullmatch(r"global_step_(\d+)", Path(args.checkpoint).name)
            if not match:
                p.error("checkpoint directory must be named global_step_N")
            if args.method == "score":
                stages = [1 if int(match.group(1)) <= c["score_stage1_steps"] else 2]
        for stage in stages:
            run_child(c, args, args.method, stage)

if __name__ == "__main__":
    main()
