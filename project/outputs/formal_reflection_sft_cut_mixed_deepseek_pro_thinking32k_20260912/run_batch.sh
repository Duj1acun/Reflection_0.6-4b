#!/usr/bin/env bash
set -uo pipefail
cd /root/reflection_grpo_clean
task_dir=outputs/formal_reflection_sft_cut_mixed_deepseek_pro_thinking32k_20260912
date -u '+%FT%TZ batch_started'
echo running > "$task_dir/batch.status"
bash scripts/data/formal_reflection_sft.sh run
run_rc=$?
unset PARATERA_API_KEY
if (( run_rc != 0 )); then
  echo "run_failed exit_code=$run_rc" > "$task_dir/batch.status"
  exit "$run_rc"
fi
echo exporting > "$task_dir/batch.status"
bash scripts/data/formal_reflection_sft.sh export
export_rc=$?
if (( export_rc != 0 )); then
  echo "export_failed exit_code=$export_rc" > "$task_dir/batch.status"
  exit "$export_rc"
fi
echo completed > "$task_dir/batch.status"
date -u '+%FT%TZ batch_completed'
