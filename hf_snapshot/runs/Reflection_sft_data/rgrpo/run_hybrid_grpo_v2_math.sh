export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3}
NGPUS_PER_NODE=2

set -xeo pipefail
export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
export RAY_TMPDIR=/ssd/data/wangruxin/ray_tmp_hybrid_v3
# vLLM CuMemAllocator is incompatible with expandable_segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# vLLM 稳定性：强制 FlashAttention 后端避免 fallback 路径在高负载下触发 IMA；
# NCCL_LAUNCH_MODE=PARALLEL 防止 vLLM 内 NCCL race 导致 logits_processor 崩。
# 这两条是为了避免 v3 在 step 282 出现的 "illegal memory access" 类崩溃。
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export NCCL_LAUNCH_MODE="${NCCL_LAUNCH_MODE:-PARALLEL}"

# ★ 启用 v2 monkey-patch（rgrpo/__init__.py 会据此导入 core_hybrid_grpo_v2）
export RGRPO_HYBRID_V2=1
# Prefix-level counterfactual: for each reflection event, also sample from the
# same pre-reflection prefix without the reflection tag. The event reward then
# compares reflect-vs-continue under the same thought prefix.
export HYBRID_V2_PREFIX_CONTRAST="${HYBRID_V2_PREFIX_CONTRAST:-1}"
export HYBRID_V2_PREFIX_CONTRAST_PENALTY_COEF="${HYBRID_V2_PREFIX_CONTRAST_PENALTY_COEF:-1.0}"

# ============ 项目配置 ============
project_name='Hybrid-GRPO-Mathlighteval-qwen2.5-math-1.5b-v3'
exp_name="${EXP_NAME:-hybrid-grpo-qwen2.5-math-1.5b-v5}"
# resume_mode=auto：如果遇到 v3 那种偶发 vLLM IMA 崩溃，重新拉起脚本可自动从最近 ckpt 续训。
resume_mode="${RESUME_MODE:-auto}"
resume_from_path="${RESUME_FROM_PATH:-null}"

# ============ 算法配置 ============
adv_estimator=hybrid_grpo   # ← 名字保持 v1，monkey-patch 会替换底层实现

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True

kl_loss_coef="${KL_LOSS_COEF:-0.005}"

clip_ratio_low=0.2
clip_ratio_high=0.2

# ============ 数据配置 ============
# Token 预算说明（避免 cont 被截断）：
#   基模型 Qwen2.5-Math-1.5B 的 max_position_embeddings=4096，无 rope_scaling，
#   因此 max_model_len 硬上限 = 4096。
#   cont rollout 的 prompt = header + reflection_tag，由 _truncate_cont_prompt
#   (rollout_loop.py:642) 上限为 max_prompt_length + max_response_length//2。
#   要保证 cont prompt + cont response ≤ max_model_len，需满足
#       max_prompt_length + 1.5 × max_response_length ≤ 4096
#   1024 + 1.5 × 1536 = 3072 → 取 max_response_length=1536。
max_prompt_length=1024

train_prompt_bsz=16
n_resp_per_prompt=8             # G=8
k_continuations="${K_CONTINUATIONS:-4}"  # K 条续推/event，可用 K_CONTINUATIONS 覆盖
export HYBRID_V2_PREFIX_CONTRAST_K="${HYBRID_V2_PREFIX_CONTRAST_K:-${k_continuations}}"
train_prompt_mini_bsz=32
val_batch_size=256
val_before_train="${VAL_BEFORE_TRAIN:-True}"   #
test_freq="${TEST_FREQ:-20}"                    # 验证 50 次
max_response_length="${MAX_RESPONSE_LENGTH:-1536}"   # cont 不截断的硬约束；可用 MAX_RESPONSE_LENGTH 覆盖
# 注：+actor_rollout_ref.rollout.max_tokens 是 dead config（verl/agent_system 无任何代码读取），
#     真正的生成上限 = SamplingParams.max_tokens = config.response_length = max_response_length。
per_round_max_tokens="${PER_ROUND_MAX_TOKENS:-1536}"
max_model_len="${MAX_MODEL_LEN:-4096}"               # 显式指定，避免回退到 prompt+response=3072 之类
loss_agg_mode="seq-mean-token-mean"
save_freq="${SAVE_FREQ:-50}"                    # 缩短存档间隔，配合 resume_mode=auto 防丢
total_training_steps="${TOTAL_TRAINING_STEPS:-1000}"
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
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/hybrid_grpo_v2/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl}"

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
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not contain a HuggingFace model (missing config.json)"
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
echo "[INFO] α schedule: ${HYBRID_V2_ALPHA_BASE} → ${HYBRID_V2_ALPHA_FINAL} over ${HYBRID_V2_ALPHA_WARMUP_STEPS} steps"
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
actor_grad_clip="${ACTOR_GRAD_CLIP:-0.5}"   # v1 是 1.5；v2 调严避免 grad_norm=nan
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${train_prompt_mini_bsz}}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"   # OOM 发生在 actor backward，优先降低每卡 PPO token 峰值
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"  # 4 GPU × 32 = 128 size divisor；direct batch=32*8=256 % 128 == 0 ✓。实际 logp 吞吐由下面的 max_token_len 决定，这里只影响 batch divisibility
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
# Throttle 三件套保持 v3 原值——为了维持 v3 的训练数值一致性。v3 在 step 282
# 因 vLLM IMA 崩；现在的对策是 swap_space=4（CPU swap 兜底，不改采样数值），
# 配合 resume_mode=auto + save_freq=20。如果仍崩，续训时再叠 enforce_eager=True
# （那时只引入 ~0.01% token drift，且只作用于后半段训练）。
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-128}"
# vLLM swap_space：KV cache 撑不下时临时落到 CPU 内存，避免 scheduler 做激进
# preempt 触发的 IMA。对 token 输出严格无影响（per-request RNG state 完整保留）。
rollout_swap_space="${ROLLOUT_SWAP_SPACE:-4}"

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
    actor_rollout_ref.actor.grad_clip=${actor_grad_clip} \
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
    +actor_rollout_ref.rollout.swap_space=${rollout_swap_space} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
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
    custom_reward_function.name=compute_hybrid_v2_math_score \
    custom_reward_function.path=rgrpo.core_hybrid_grpo_v2 \
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
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path=${resume_from_path}
