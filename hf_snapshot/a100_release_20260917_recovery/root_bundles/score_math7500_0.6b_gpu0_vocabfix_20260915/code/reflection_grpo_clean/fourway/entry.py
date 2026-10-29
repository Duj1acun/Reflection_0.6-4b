"""Child process: its working directory selects the correct native verl."""
import argparse
import os
from pathlib import Path
from .contracts import atomic_json, config
from .configuration import build

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--evaluate")
    args = parser.parse_args()
    c = config()
    import verl
    expected = c["pag_repo"] if args.method in {"pag", "score"} else c["reflection_repo"]
    if not Path(verl.__file__).resolve().is_relative_to(Path(expected).resolve()):
        raise RuntimeError(f"Wrong verl checkout imported: {verl.__file__}")
    cfg = build(c, args.method, args.stage, args.probe, args.evaluate)
    from omegaconf import OmegaConf
    run = Path(cfg.trainer.default_local_dir)
    atomic_json(run/f"stage{args.stage}.resolved.json", OmegaConf.to_container(cfg, resolve=True))
    if args.probe:
        # Small isolated input, native FSDP/critic/PPO paths, no optimizer updates.
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pq.read_table(cfg.data.train_files[0]).slice(0, int(cfg.data.train_batch_size))
        path = run/"probe.parquet"
        pq.write_table(table, path)
        cfg.data.train_files = [str(path)]
        original = pq.read_table(cfg.data.val_files[0])
        validation = pa.concat_tables([original.slice(0, 1), original.slice(500, 1)])
        val_path = run/"probe.validation.parquet"
        pq.write_table(validation, val_path)
        cfg.data.val_files = [str(val_path)]
        cfg.trainer.val_before_train = True
    from verl.trainer.main_ppo import run_ppo
    try:
        run_ppo(cfg)
        if args.probe:
            print("FOURWAY_PROBE_COMPLETED: native backward completed with AdamW.step disabled", flush=True)
    finally:
        import ray
        ray.shutdown()

if __name__ == "__main__":
    main()
