set -x

math500=datasets/math500.parquet
math7500=datasets/math7500.parquet
aime2024=datasets/aime2024.parquet
aime2025=datasets/aime2025.parquet
minervamath=datasets/minervamath.parquet
dapo17k=datasets/dapo17k.parquet

PROJECT_NAME='PAG'
CKPT_PATH=checkpoints
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

RESUME_PATH=None # your trained model, path end with global_step_xxx


n=4
rollout_type=pag
num_turns=2

EXPERIMENT_NAME="eval_pag_qwen1p5b"

python3 -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.val_before_train=True \
    trainer.val_only=True \
    algorithm.adv_estimator=grpo \
    data.train_files=[$dapo17k] \
    data.val_files="['$math500']" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2028 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.num_turns=$num_turns \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.rollout_type=$rollout_type \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.num_turns=$num_turns \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=20 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME
