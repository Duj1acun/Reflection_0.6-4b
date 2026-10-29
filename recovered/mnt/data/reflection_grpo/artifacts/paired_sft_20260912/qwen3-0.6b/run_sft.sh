#!/usr/bin/env bash
set -uo pipefail
cd /root/reflection_grpo_clean
echo running > /mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-0.6b/sft.status
date -u '+%FT%TZ sft_started'
bash training/sft/train_paired_sft.sh 0.6b train 0
sft_rc=$?
if (( sft_rc == 0 )); then echo completed > /mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-0.6b/sft.status; else echo "failed exit_code=$sft_rc" > /mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-0.6b/sft.status; fi
date -u '+%FT%TZ sft_exited'
exit "$sft_rc"
