#!/usr/bin/env bash
set -uo pipefail
cd /root/reflection_grpo_clean
task_dir=outputs/formal_reflection_sft_cut_mixed_2000each_deepseek_pro_thinking32k_20260912
python_bin=/home/vipuser/miniconda3/envs/wrx_env/bin/python3
common_args=(--output "$task_dir" --teacher DeepSeek-V4-Pro --thinking enabled --max-tokens 32768 --workers 64 --attempts 3 --phase1-prefix cut)
date -u '+%FT%TZ batch_started'
echo running > "$task_dir/batch.status"
"$python_bin" scripts/data/build_two_stage_sft.py run "${common_args[@]}"
run_rc=$?
unset PARATERA_API_KEY
if (( run_rc != 0 )); then
  echo "run_failed exit_code=$run_rc" > "$task_dir/batch.status"
  exit "$run_rc"
fi
echo exporting > "$task_dir/batch.status"
"$python_bin" scripts/data/build_two_stage_sft.py export "${common_args[@]}"
export_rc=$?
if (( export_rc != 0 )); then
  echo "export_failed exit_code=$export_rc" > "$task_dir/batch.status"
  exit "$export_rc"
fi
echo completed > "$task_dir/batch.status"
date -u '+%FT%TZ batch_completed'
