#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/data/runs/Reflection_sft_data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_vanilla_math12000}"
export RAY_object_spilling_config="${RAY_object_spilling_config:-{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/mnt/data/ray_spill\"}}}"
mkdir -p /mnt/data/ray_spill
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
unset RAY_ADDRESS RAY_INIT_ADDRESS || true
PYTHON_BIN="${PYTHON_BIN:-/root/.conda/envs/wrx_env/bin/python}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/rl_data/math12000_pool/math12000_pool.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/runs/checkpoints/vanilla_grpo_math12000}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  ray_init.num_cpus="${RAY_NUM_CPUS:-8}" ray_init.object_store_memory="${RAY_OBJECT_STORE_MEMORY:-4294967296}" \
  data.train_files="${TRAIN_FILE}" data.val_files="${VAL_FILE}" \
  data.prompt_key=question data.reward_fn_key=data_source data.truncation=left \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-384}" data.max_response_length="${MAX_RESPONSE_LENGTH:-3072}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ:-2}" data.val_batch_size="${VAL_BATCH_SIZE:-2}" \
  data.return_raw_chat=True +data.dataloader_num_workers="${DATALOADER_WORKERS:-0}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-2}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}" \
  actor_rollout_ref.rollout.n=1 env.rollout.n="${GROUP_SIZE:-4}" \
  +actor_rollout_ref.rollout.k_continuations="${REFLECTION_CONTINUATIONS:-2}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS:-8}" actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-4096}" \
    actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    custom_reward_function.name=compute_math_score \
    custom_reward_function.path="${ROOT}/rgrpo/core_rgrpo.py" \
    reward_model.reward_manager=naive env.env_name=none \
    trainer.logger="['console']" trainer.project_name=math12000 \
    trainer.experiment_name=vanilla_grpo trainer.n_gpus_per_node="${TRAINER_GPUS:-2}" trainer.nnodes=1 \
    trainer.val_before_train=False trainer.test_freq=0 \
    trainer.save_freq="${SAVE_FREQ:-10}" trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
    trainer.total_epochs=100000 trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}"
