# SCoRe and PAG full source snapshot — 2026-09-21

This bundle preserves the existing local implementations, not a validated Qwen2.5-Math-1.5B/LIMO experiment. No training has been launched for this export. Original source directories were not modified.

## Contents
- `code/Policy-As-GenVerifier`: both baseline entrypoints, native verl trainers, rollout/reward/worker implementation, evaluation, tests, setup and dependency files.
- `code/reflection_grpo_clean`: shared `fourway` preparation/configuration/training/preflight/scoring/checkpoint orchestration, native code and tests.
- `MANIFEST.json`: SHA256 and length of every copied file. Bytecode and caches excluded; no model weights or runtime environment included.

Source: recovered `root_bundles/score_math7500_0.6b_gpu0_vocabfix_20260915/code`. Recovery contains no .git, so upstream commit is not asserted. Upstream: https://github.com/Jackory/Policy-As-GenVerifier . Local patches are preserved.

## Entry points
- `code/Policy-As-GenVerifier/quick_start/qwen1p5b_SCoRe.sh`
- `code/Policy-As-GenVerifier/quick_start/qwen1p5b_pag.sh`
- `code/reflection_grpo_clean/fourway/cli.py`
- `code/reflection_grpo_clean/experiments/four_methods/`

## Before a new run
Historical launchers/configurations reference old absolute paths and Qwen3-0.6B/MATH7500. Do not run unchanged. Use separate code/environment/config/output/Ray temporary directories and explicit GPUs. Adapt LIMO question/answer data preparation, model/tokenizer, context budget, critic/reference initialization and evaluation datasets. Validate SCoRe stage transition and PAG role masks/rewards. Run isolated preflight before training. Existing source includes setup/requirements files; installed runtime is not bundled or claimed portable.

Available input: Qwen/Qwen2.5-Math-1.5B (base). LIMO: Duj1acun/Reflect_LIMO dataset revision bae904561e40e91ffae31611cb6c207631bd0c35, limo_merged.jsonl, 1604 rows, fields question/solution/answer, SHA256 a48c249fbb060ee3b82f00a9b6a0039e038ec0364772ea0c78f502d1c3c4950d. Input data is not duplicated here. RL should generate its own trajectories; reference answers are for reward evaluation, not prompt feedback.
