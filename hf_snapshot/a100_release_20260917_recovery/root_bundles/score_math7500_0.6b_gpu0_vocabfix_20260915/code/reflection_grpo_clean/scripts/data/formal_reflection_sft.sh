#!/usr/bin/env bash
# Preparation is offline. Paid calls require the explicit "run" subcommand.
set -euo pipefail
cd /root/reflection_grpo_clean
action="${1:-prepare}"
case "$action" in
  prepare|run|export) ;;
  *) echo 'Usage: bash scripts/data/formal_reflection_sft.sh [prepare|run|export]' >&2; exit 2 ;;
esac
if [[ "$action" == run && -z "${PARATERA_API_KEY:-}" ]]; then
  echo 'Set PARATERA_API_KEY before run. No API request sent.' >&2
  exit 2
fi
exec /home/vipuser/miniconda3/envs/wrx_env/bin/python3 scripts/data/build_two_stage_sft.py "$action" \
  --output /root/reflection_grpo_clean/outputs/formal_reflection_sft_cut_mixed_2000each_deepseek_pro_thinking32k_20260912 \
  --teacher DeepSeek-V4-Pro --thinking enabled --max-tokens 32768 \
  --workers 64 --attempts 3 --phase1-prefix cut
