#!/usr/bin/env python3
"""Relocate and resume the recovered REFLECT global_step_150 checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


BUNDLE = Path(__file__).resolve().parent
EXPERIMENT_NAME = "reflect_math7500_0.6b_gpu1_branch4_stable_20260915"
BASE_BUNDLE_NAME = "reflect_math7500_0.6b_20260914"
CODE_BUNDLE_NAME = "reflect_tokenbudget_fix_20260915"


def paths(bundle: Path) -> dict[str, Path]:
    base = bundle / "root_bundles" / BASE_BUNDLE_NAME
    code_bundle = bundle / "root_bundles" / CODE_BUNDLE_NAME
    output = bundle / "data" / "reflection_grpo" / "experiments" / EXPERIMENT_NAME
    return {
        "bundle": bundle,
        "base": base,
        "code_bundle": code_bundle,
        "repo": code_bundle / "code" / "reflection_grpo_clean",
        "pag_repo": base / "code" / "Policy-As-GenVerifier",
        "output": output,
        "checkpoint": output / "runs" / "reflect" / "global_step_150",
        "merged": base / "models" / "reflect_merged",
        "config": bundle / "resume_reflect_step150.config.json",
        "log": output / "runs" / "reflect" / "resume_step150.log",
    }


def make_config(p: dict[str, Path], python: str, gpu: str) -> dict:
    source = p["code_bundle"] / "config.json"
    c = json.loads(source.read_text())
    c.update(
        output=str(p["output"]),
        python=str(Path(python).resolve()),
        reflection_repo=str(p["repo"]),
        pag_repo=str(p["pag_repo"]),
        base_model=str(p["base"] / "models" / "Qwen3-0.6B"),
        sft_adapter=str(p["base"] / "models" / "adapter"),
        train_source=str(p["base"] / "assets" / "sources" / "math7500.parquet"),
        math500_source=str(p["base"] / "assets" / "sources" / "math500.parquet"),
        amc23_source=str(p["base"] / "assets" / "sources" / "amc23.dataset.jsonl"),
        ray_tmp_dir=str(p["bundle"] / "runtime" / "ray_reflect_resume"),
        lock_dir=str(p["bundle"] / "runtime" / "locks"),
        hf_home=str(p["base"] / "runtime" / "huggingface"),
        gpus=gpu,
    )
    p["config"].write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
    return c


def ensure_merged_link(p: dict[str, Path]) -> None:
    link = p["output"] / "initialization" / "reflect_merged"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == p["merged"].resolve():
        # Recreate an absolute legacy link as a portable relative link.
        if not link.readlink().is_absolute():
            return
        link.unlink()
    elif link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    relative_target = Path(os.path.relpath(p["merged"], link.parent))
    link.symlink_to(relative_target, target_is_directory=True)


def check(p: dict[str, Path], c: dict) -> None:
    if "," in c["gpus"]:
        raise RuntimeError(
            "global_step_150 is an FSDP world-size-1 checkpoint; exact resume requires one GPU"
        )
    required = {
        "actor/model_world_size_1_rank_0.pt": 3_006_686_570,
        "actor/optim_world_size_1_rank_0.pt": 4_768_484_443,
        "actor/extra_state_world_size_1_rank_0.pt": 14_696,
        "data.pt": 1_108,
        "judge_state.json": 129,
    }
    for rel, size in required.items():
        f = p["checkpoint"] / rel
        if not f.is_file() or f.stat().st_size != size:
            raise RuntimeError(f"Missing or wrong-sized checkpoint file: {f}")
    for f in [
        p["repo"] / "fourway" / "configuration.py",
        p["pag_repo"] / "verl" / "trainer" / "main_ppo.py",
        p["merged"] / "config.json",
        p["output"] / "data" / "reflect.train.parquet",
        p["output"] / "data" / "reflect.validation.parquet",
    ]:
        if not f.exists():
            raise RuntimeError(f"Required recovery artifact is missing: {f}")
    expected = {
        "torch": "2.6.0",
        "transformers": "4.51.1",
        "vllm": "0.8.5",
        "ray": "2.47.1",
        "math-verify": "0.9.0",
        "latex2sympy2_extended": "1.11.0",
        "flash_attn": "2.7.4.post1",
    }
    for package, version in expected.items():
        actual = importlib.metadata.version(package).split("+")[0]
        if actual != version:
            raise RuntimeError(f"Environment mismatch: {package}={actual}, expected {version}")
    import torch

    for rel in [
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "data.pt",
    ]:
        obj = torch.load(p["checkpoint"] / rel, map_location="cpu", mmap=True, weights_only=False)
        del obj
    print(f"CHECK_OK checkpoint={p['checkpoint']} next_update=151 total_steps={c['steps']}")


def check_gpu(gpu: str, allow_low_memory: bool) -> None:
    import torch

    idx = int(gpu)
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {gpu} is unavailable")
    prop = torch.cuda.get_device_properties(idx)
    total_gib = prop.total_memory / 2**30
    free_bytes, _ = torch.cuda.mem_get_info(idx)
    free_gib = free_bytes / 2**30
    print(f"GPU_CHECK index={idx} name={prop.name} total={total_gib:.1f}GiB free={free_gib:.1f}GiB")
    if not allow_low_memory and (total_gib < 70 or free_gib < 65):
        raise RuntimeError(
            "The recovered run used about 79 GiB. Use one idle 80 GB GPU; "
            "--allow-low-memory only bypasses this guard and may OOM."
        )


def child(p: dict[str, Path], c: dict) -> None:
    sys.path.insert(0, str(p["repo"]))
    from fourway.configuration import build
    from fourway.contracts import atomic_json
    from omegaconf import OmegaConf

    cfg = build(c, "reflect")
    cfg.trainer.resume_mode = "resume_path"
    cfg.trainer.resume_from_path = str(p["checkpoint"])
    cfg.trainer.val_before_train = False
    cfg.trainer.del_local_ckpt_after_load = False
    if cfg.trainer.n_gpus_per_node != 1:
        raise RuntimeError("Exact resume requires FSDP world size 1")
    atomic_json(
        p["output"] / "runs" / "reflect" / "resume_step150.resolved.json",
        OmegaConf.to_container(cfg, resolve=True),
    )
    print("NATIVE_RESUME from_step=150 next_update=151 total_steps=500", flush=True)
    from verl.trainer.main_ppo import run_ppo

    try:
        run_ppo(cfg)
    finally:
        import ray

        ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0", help="One physical GPU index; exact resume is world size 1")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    p = paths(args.bundle.resolve())
    ensure_merged_link(p)
    c = make_config(p, args.python, args.gpu)
    check(p, c)
    if args.child:
        child(p, c)
        return
    if not args.train:
        print(f"CONFIG_WRITTEN {p['config']}")
        return
    check_gpu(args.gpu, args.allow_low_memory)
    sys.path.insert(0, str(p["repo"]))
    from fourway.cli import environment

    repo, env = environment(c, p["config"], "reflect")
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    command = [c["python"], "-u", str(Path(__file__).resolve()), "--bundle", str(p["bundle"]),
               "--python", c["python"], "--gpu", args.gpu, "--child"]
    print("RUNNING", " ".join(command), "log=", p["log"], flush=True)
    with p["log"].open("a") as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"Training exited with code {result.returncode}; inspect {p['log']}")


if __name__ == "__main__":
    main()
