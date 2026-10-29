#!/usr/bin/env bash
# Hybrid-GRPO: 联合奖励设计的反思推理训练脚本
#
# 当前优势分配逻辑（与 rgrpo/core_hybrid_grpo.py 保持一致）：
#   penalty(n) = (n/4)^2   n=1→0.0625, n=2→0.25, n=3→0.5625
#   direct 样本：正确 → +1.0；错误 → -(3/4)^2 = -0.5625
#   continuation prefix：正确 → +1.0 - penalty(n_refl)；
#                        错误 → -(3/4)^2 - penalty(n_refl)
#   主池 GRPO：对 direct / reflective prefix 做组内 z-score
#   token 级回传：
#     prefix token -> prefix_adv
#     cont token   -> local_cont_adv（同一 reflection event 内部归一化）

export CUDA_VISIBLE_DEVICES=0,1
NGPUS_PER_NODE=2

set -xeo pipefail

export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
export RAY_TMPDIR=/ssd/data/wangruxin/ray_tmp_hybrid_grpo
# vLLM 的 CuMemAllocator 与 PyTorch expandable_segments 不兼容，不能在这里打开。

# ============ 项目配置 ============
project_name='Hybrid-GRPO-with-step-reward-v3-427'
exp_name="${EXP_NAME:-hybrid-grpo-qwen2.5-math-1.5b-limo}"
resume_mode="${RESUME_MODE:-disable}"  # disable | latest | specific

# ============ 算法配置 ============
adv_estimator=hybrid_grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.01

clip_ratio_low=0.15
clip_ratio_high=0.15

# ============ 数据配置 ============
max_prompt_length=1024
max_response_length="${MAX_RESPONSE_LENGTH:-4096}"
per_round_max_tokens="${PER_ROUND_MAX_TOKENS:-2048}"
loss_agg_mode="seq-mean-token-mean"

train_prompt_bsz=16
n_resp_per_prompt=8             # G=8
k_continuations=4               # K=4（每反思事件生成 4 条续推，用于计算 refl_reward）
train_prompt_mini_bsz=16
val_batch_size=128
val_before_train="${VAL_BEFORE_TRAIN:-Ture}"
test_freq="${TEST_FREQ:-100}"  # 按 step 计的验证频率
save_freq="${SAVE_FREQ:-100}"  # 按 step 计的存档频率
total_training_steps="${TOTAL_TRAINING_STEPS:-1000}"  # 训练总步数
# ray_trainer 外层仍按 total_epochs 循环，因此这里给一个足够大的上限，
# 真正的停止条件由 total_training_steps 控制。
total_epochs_cap="${TOTAL_EPOCHS_CAP:-100000}"

# ============ 集群配置 ============
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# ============ 路径配置 ============
RAY_DATA_HOME="/ssd/data/wangruxin/R_grpo_sft"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ssd/data/wangruxin/Qwen2.5-Math-1.5B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-/ssd/data/wangruxin/reasoning_paths_directional/sft_training/outputs/phase2/final}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${RAY_DATA_HOME}/sft_training/outputs/phase2/final_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${MERGED_SFT_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/hybrid_grpo/${exp_name}}"
TRAIN_FILE="${RAY_DATA_HOME}/data/mixed_train_data.jsonl"
TEST_FILE="${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl"

# 若合并模型不存在，自动从 LoRA adapter 合并
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ "${MODEL_PATH}" == "${MERGED_SFT_MODEL_PATH}" ]] && [[ -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "[INFO] Merging SFT adapter into base model..."
        BASE_MODEL_PATH="${BASE_MODEL_PATH}" SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH}" python3 - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base    = os.environ["BASE_MODEL_PATH"]
adapter = os.environ["SFT_ADAPTER_PATH"]
out     = os.environ["MERGED_SFT_MODEL_PATH"]

if os.path.isfile(os.path.join(out, "config.json")):
    print(f"[INFO] Merged model already exists at {out}, skipping merge.")
else:
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok_src = adapter if os.path.isfile(os.path.join(adapter, "tokenizer.json")) else base
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    print(f"[INFO] Saved merged model to {out}")
PY
    else
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not包含 HuggingFace 模型（缺少 config.json）"
        exit 1
    fi
fi

echo "[INFO] Using model path: ${MODEL_PATH}"
mkdir -p "${CKPTS_DIR}"

# ============ 日志配置 ============
LOG_DIR="${LOG_DIR:-/ssd/data/wangruxin/outputs/train_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Training log → ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# ============ 生成配置 ============
temperature="${TEMPERATURE:-0.7}"
top_p="${TOP_P:-0.95}"
top_k=-1

# ============ 性能配置 ============
sp_size=1
use_dynamic_bsz=True
gen_tp=1
offload=False
actor_lr="${ACTOR_LR:-1e-6}"
actor_lr_scheduler_type="${ACTOR_LR_SCHEDULER_TYPE:-cosine}"
actor_lr_warmup_steps_ratio="${ACTOR_LR_WARMUP_STEPS_RATIO:-0.05}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.1}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${train_prompt_mini_bsz}}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-${train_prompt_mini_bsz}}"
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-128}"

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=question \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_batch_size} \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    +actor_rollout_ref.actor.optim.lr_scheduler_type=${actor_lr_scheduler_type} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${actor_lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.optim.weight_decay=${actor_weight_decay} \
    actor_rollout_ref.actor.grad_clip=1.5 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    +actor_rollout_ref.rollout.k_continuations=${k_continuations} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.max_tokens=${per_round_max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.rgrpo.max_reflections=3 \
    algorithm.filter_groups.enable=True \
    custom_reward_function.name=compute_hybrid_math_score \
    custom_reward_function.path=rgrpo.core_hybrid_grpo \
    reward_model.reward_manager=naive \
    env.env_name=none \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epochs_cap} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${resume_mode}"
