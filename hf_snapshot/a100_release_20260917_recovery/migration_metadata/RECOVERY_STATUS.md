# Recovery status after source shutdown

Checked at 2026-09-17 after both SSH endpoints began refusing connections.
The migration did not reach `MIGRATION_COMPLETE` or `MIGRATION_VERIFIED`.

## Fully preserved critical artifacts

- All selected `/root` reproducibility bundles (2.7 GB), including the base
  Qwen3-0.6B model, SFT adapter, merged initialization, code, configs, data,
  and environment pins.
- Experiment metadata, configs, logs, validation summaries, and samples from
  every directory under `/data/reflection_grpo/experiments`.
- Complete REFLECT checkpoint:
  `reflect_math7500_0.6b_gpu1_branch4_stable_20260915/runs/reflect/global_step_150`
  - actor model: 3,006,686,570 bytes
  - actor optimizer: 4,768,484,443 bytes
- SCORE step-100 actor and critic model states:
  - actor model: 3,006,686,570 bytes
  - critic model: 2,384,361,210 bytes
  - actor optimizer: 4,768,468,571 bytes

These five model/optimizer files were opened successfully with
`torch.load(..., mmap=True)` on the destination machine. The incomplete SCORE
critic optimizer fails the same check because its zip central directory is
missing, confirming that it was interrupted mid-transfer.

## Incomplete or unavailable

- SCORE step-100 critic optimizer is partial: 2,805,497,856 bytes locally;
  the expected complete file is about 4.77 GB. Exact SCORE optimizer-state
  resume is therefore unavailable, although the complete actor and critic
  weights can be loaded with fresh optimizer state.
- Legacy REFLECT step-200 actor tensor is partial (27,492,352 bytes).
- The 12 GB Conda environment was not copied. Environment pins and setup logs
  were preserved, so it can be rebuilt.
- Ray temporary log trees were not copied.

## Resume if the source returns

After recreating the SSH control connection, rerun:

```bash
/root/migrate_a100_release_20260917.sh
```

Rsync will continue the partial tensors instead of starting from zero.
