# A100 training recovery bundle — 2026-09-17

This directory is a recovery bundle for the interrupted Reflection-GRPO
experiments. Despite the parent Hugging Face repository name, the checkpoints
in this directory are for **Qwen3-0.6B**, not Qwen3-4B.

## Recoverable training states

### REFLECT (exact native state resume)

- Experiment: `reflect_math7500_0.6b_gpu1_branch4_stable_20260915`
- Last completed update in the log: 158
- Complete checkpoint: `global_step_150`
- Resume starts with update 151 and redoes updates 151–158.
- Preserved state: actor model, Adam optimizer, LR scheduler, RNG, stateful
  dataloader position, and judge counters.
- Checkpoint topology: FSDP world size 1.

Use the final code and configuration under:

`root_bundles/reflect_tokenbudget_fix_20260915`

The historical `resume.py` in that directory was written for step 50. Do not
run it unchanged. Set `trainer.resume_mode=resume_path` and point
`trainer.resume_from_path` to the included `global_step_150` directory.

### SCORE (weights-only recovery)

- Experiment: `score_math7500_0.6b_gpu0_vocabfix_20260915`
- Last completed update in the log: 137
- Last checkpoint: `global_step_100`
- Actor model and optimizer, critic model, scheduler/RNG, dataloader state, and
  judge state are preserved.
- The critic optimizer transfer was interrupted and is intentionally excluded.

The current native loader requires the critic optimizer, so SCORE cannot use an
exact native-state resume without the missing source artifact. A weights-only
loader with fresh optimizer state is required.

## Directory map

- `root_bundles/` maps to the original `/root/` reproducibility bundles.
- `data/reflection_grpo/` maps to the original `/data/reflection_grpo/` tree.
- `migration_metadata/` contains the source migration and recovery reports.
- `EXCLUDED_ARTIFACTS.json` records deliberately omitted corrupt files.
- `SHA256SUMS` verifies every uploaded regular file except itself.

For maximum path compatibility on a clean recovery server:

```bash
rsync -a root_bundles/ /root/
mkdir -p /data/reflection_grpo
rsync -a data/reflection_grpo/ /data/reflection_grpo/
```

The original Conda environment was not transferred. Rebuild a Python 3.10
environment using `data/reflection_grpo/setup_logs/runtime_closure.json` and
`runtime_pins.txt`. Important versions include Torch 2.6.0+cu124, vLLM 0.8.5,
Transformers 4.51.1, FlashAttention 2.7.4.post1, and Ray 2.47.1.

Run CPU and GPU preflight checks before loading a checkpoint. The REFLECT run
reserved almost 79 GiB of GPU memory, so an 80 GB GPU is required. Run REFLECT
and SCORE sequentially; the source 64 GB host hit Ray's 95% host-memory kill
threshold while both experiments were active.

## Integrity notes

The REFLECT step-150 model, optimizer, extra state, and dataloader state passed
ZIP central-directory checks. The large model and optimizer shards also passed
full CRC reads on 2026-09-18. Migration records report successful
`torch.load(..., mmap=True)` checks for the preserved critical tensors.

This package contains source code, model weights, datasets, configuration,
validation outputs, diagnostics, and environment metadata needed for recovery.
It intentionally does not contain access tokens, SSH credentials, the original
Conda environment, Ray temporary logs, PID files as live process state, or the
known-corrupt checkpoint shards listed in `EXCLUDED_ARTIFACTS.json`.
