#!/usr/bin/env bash
set -euo pipefail

ln -sfn /data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_stable_20260915 /root/reflect_tokenbudget_fix_20260915/results
ln -sfn /data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu0_main_3rounds /root/reflect_math7500_0.6b_20260914/results
ln -sfn /data/reflection_grpo/experiments/score_math7500_0.6b_gpu0_vocabfix_20260915 /root/score_math7500_0.6b_gpu0_vocabfix_20260915/results
ln -sfn /root/reflect_math7500_0.6b_20260914/models/reflect_merged /data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_stable_20260915/initialization/reflect_merged
ln -sfn /root/reflect_math7500_0.6b_20260914/models/reflect_merged /data/reflection_grpo/experiments/score_math7500_0.6b_gpu0_vocabfix_20260915/initialization/reflect_merged
