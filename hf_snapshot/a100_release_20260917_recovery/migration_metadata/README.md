# A100 instance migration — 2026-09-17

> Source access was lost before the full migration completed. See
> `RECOVERY_STATUS.md` for the exact preserved and incomplete artifacts.

Source server: `root@180.127.11.169:33316`.

This directory preserves the experiments and runtime needed after the two-A100
instance is released. Remote absolute paths are represented below `remote_root`,
`remote_data`, and `remote_tmp`.

## Priority artifacts

1. `remote_data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_stable_20260915`
   - last completed update: 158
   - complete recoverable checkpoint: `runs/reflect/global_step_150`
   - failure: Ray host-RAM monitor killed the worker at 95% node memory use
2. `remote_data/reflection_grpo/experiments/score_math7500_0.6b_gpu0_vocabfix_20260915`
   - last completed update: 137
   - complete recoverable checkpoint: `runs/score/global_step_100`
   - failure: the same Ray host-RAM threshold
3. `remote_data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu0_main_3rounds_sftprompt_nogate`
   - historical main-method comparison checkpoint: `runs/reflect/global_step_200`

All other directories under `experiments` are retained because they contain
preflight evidence, earlier baselines, validation samples, scorer diagnostics,
and configuration history.

`remote_data/reflection_grpo/setup_logs` preserves the environment setup and
installation diagnostics.

## Reproducibility bundles

- `remote_root/reflect_math7500_0.6b_20260914`: base Qwen3-0.6B, SFT adapter,
  merged initialization, source datasets, code snapshots, and environment pins.
- `remote_root/reflect_tokenbudget_fix_20260915`: final REFLECT code/config.
- `remote_root/score_math7500_0.6b_gpu0_vocabfix_20260915`: final SCORE code/config.
- Other dated REFLECT/SCORE bundles preserve the diagnosis and restart history.

The base bundle was seeded using hard links from the already verified local copy
at `/mnt/data/reflection_grpo/transfers/reflect_math7500_0.6b_20260914`, then
synchronized against the remote directory.

## Transfer state

- `MIGRATION.log`: completed rsync stages and statistics.
- `nohup.log`: live transfer output.
- `MIGRATION_COMPLETE`: created only after all experiment checkpoints, the exact
  conda runtime, and both Ray log trees have completed.
- Monitor with `/root/check_a100_migration.sh`.

The main job copies the current REFLECT step-150 checkpoint first. A second job
prefetches the SCORE step-100 actor model through the mobile network endpoint.
Its log is `SCORE_CORE_PARALLEL.log`; the marker `SCORE_ACTOR_MODEL_COMPLETE`
is created after that model is complete. The main job later copies and verifies
the complete SCORE checkpoint, including the critic and both optimizers.

Rsync uses partial files and may be safely restarted with
`/root/migrate_a100_release_20260917.sh` if the network connection is interrupted.
