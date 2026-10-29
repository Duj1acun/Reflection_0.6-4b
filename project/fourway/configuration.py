"""Resolve both verl versions from one explicit experiment configuration."""
from pathlib import Path
from .contracts import METHODS

REFLECT_ENV = {
    "TRAJ_GRPO_REWARD_PHI": "4", "TRAJ_GRPO_REWARD_STYLE": "efficiency",
    "TRAJ_GRPO_EFF_FREE_REFLECTIONS": "1", "TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT": "0",
    "TRAJ_GRPO_REFLECTION_FREQ_PENALTY": "0.0", "TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY": "0.0",
    "TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY": "1", "TRAJ_GRPO_WRONG_NO_REFLECT_N": "3",
    "TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY": "0", "TRAJ_GRPO_TRAIN_FULL_CHAIN": "1",
    "TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK": "0", "TRAJ_GRPO_DECOUPLED_BASELINE": "1",
    "TRAJ_GRPO_REFLECT_ADV_FLOOR": "-0.5", "TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS": "0",
    "TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED": "1",
}

def build(c, method, stage=1, probe=False, evaluate=None):
    from omegaconf import OmegaConf
    if method not in METHODS:
        raise ValueError(method)
    repo = c["pag_repo"] if method in {"pag", "score"} else c["reflection_repo"]
    cfg = OmegaConf.load(Path(repo) / "verl/trainer/config/ppo_trainer.yaml")
    output = Path(c["output"])
    run = output / ("probes" if probe else "runs") / method
    if probe:
        run = run / f"stage{stage}"
    elif evaluate:
        run = output / "evaluations" / method / Path(evaluate).name
    model = str(output / "initialization/reflect_merged") if method == "reflect" else c["base_model"]
    args = {
        "data.train_files": [str(output / "data" / f"{method}.train.parquet")],
        "data.val_files": [str(output / "data" / f"{method}.validation.parquet")],
        "data.prompt_key": "prompt", "data.return_raw_chat": True,
        "data.filter_overlong_prompts": False, "data.truncation": "error",
        "data.seed": c["seed"], "data.shuffle": True,
        "data.train_batch_size": 4 if probe else c["train_batch_size"],
        "data.val_batch_size": c["validation"]["batch_size"],
        "data.max_prompt_length": c["max_prompt_length"],
        "data.max_response_length": max(c["stage_tokens"][method]),
        "actor_rollout_ref.model.path": model,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 4 if probe else c["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": c["ppo_micro_batch_size_per_gpu"],
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": c["tokens_per_gpu"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.optim.lr": c["actor_lr"],
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.warmup_style": "constant",
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_invalid_action_penalty": False,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.clip_ratio_low": 0.20,
        "actor_rollout_ref.actor.clip_ratio_high": 0.26,
        "actor_rollout_ref.actor.clip_ratio_c": 3.0,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.n": 2 if probe else c["train_n"],
        "actor_rollout_ref.rollout.max_model_len": c["max_model_len"],
        "actor_rollout_ref.rollout.seed": c["seed"],
        "actor_rollout_ref.rollout.bad_words": ["<think>", "</think>"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": c["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": c["max_num_batched_tokens"],
        "actor_rollout_ref.rollout.max_num_seqs": c["max_num_seqs"],
        "actor_rollout_ref.rollout.enforce_eager": c.get("enforce_eager", True),
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.val_kwargs.n": 4,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": c["validation"]["temperature"],
        "actor_rollout_ref.rollout.val_kwargs.top_p": c["validation"]["top_p"],
        "actor_rollout_ref.rollout.val_kwargs.top_k": c["validation"]["top_k"],
        "algorithm.adv_estimator": "gae" if method in {"pag", "score"} else "grpo",
        "algorithm.use_kl_in_reward": method == "score" and stage == 1,
        "algorithm.kl_ctrl.kl_coef": 0.0,
        "trainer.project_name": c["experiment"], "trainer.experiment_name": method,
        "trainer.logger": ["console"], "trainer.n_gpus_per_node": 2, "trainer.nnodes": 1,
        "trainer.default_local_dir": str(run), "trainer.resume_mode": "disable",
        "trainer.total_training_steps": 1 if probe else (c["score_stage1_steps"] if method == "score" and stage == 1 else c["steps"]),
        "trainer.total_epochs": 100000, "trainer.val_before_train": c.get("val_before_train", True) and not probe,
        "trainer.test_freq": -1 if probe else c["validation"]["frequency"],
        "trainer.save_freq": -1 if probe else c["validation"]["frequency"],
        "trainer.max_actor_ckpt_to_keep": None, "trainer.max_critic_ckpt_to_keep": None,
        "trainer.seed": c["seed"], "fourway.method": method, "fourway.stage": stage,
        "fourway.probe": probe, "fourway.output": c["output"],
    }
    for k, v in c["train_sampling"].items():
        args[f"actor_rollout_ref.rollout.{k}"] = v
    for role in ("actor_rollout_ref.rollout", "actor_rollout_ref.ref"):
        args[role+".log_prob_use_dynamic_bsz"] = True
        args[role+".log_prob_micro_batch_size_per_gpu"] = 2
        args[role+".log_prob_max_token_len_per_gpu"] = c["tokens_per_gpu"]
    args["actor_rollout_ref.ref.fsdp_config.param_offload"] = False
    if method in {"pag", "score"}:
        args.update({"actor_rollout_ref.rollout.rollout_type": "pag" if method == "pag" else "multiturn",
                     "actor_rollout_ref.rollout.num_turns": 2, "actor_rollout_ref.rollout.val_kwargs.num_turns": 2,
                     "algorithm.norm_type": "role" if method == "pag" else "whole",
                     "algorithm.kl_ctrl.multiturn_kl": method == "score" and stage == 1,
                     "algorithm.kl_ctrl.kl_coef_turn1": 0.1 if method == "score" and stage == 1 else 0.0,
                     "algorithm.kl_ctrl.kl_coef_turn2": 0.0,
                     "reward_model.policy_rs": method == "pag" or stage == 2,
                     "reward_model.rs_coef": 1.0,
                     "critic.model.path": model, "critic.optim.lr": c["critic_lr"],
                     "critic.optim.weight_decay": 0.01, "critic.optim.lr_warmup_steps": 0,
                     "critic.use_dynamic_bsz": True, "critic.model.use_remove_padding": True,
                     "critic.model.enable_gradient_checkpointing": True,
                     "critic.model.fsdp_config.param_offload": False,
                     "critic.model.fsdp_config.optimizer_offload": False,
                     "critic.ppo_micro_batch_size_per_gpu": 2,
                     "critic.ppo_max_token_len_per_gpu": c["tokens_per_gpu"]})
    else:
        args.update({"ray_init.num_cpus": 16, "ray_init.object_store_memory": 2147483648,
                     "ray_init._temp_dir": "/mnt/data/tmp/fw", "env.env_name": "none", "reward_model.reward_manager": "naive",
                     "actor_rollout_ref.model.lora_rank": 0,
                     "actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs": c["max_num_seqs"],
                     "data.dataloader_num_workers": 0,
                     "custom_reward_function.name": "compute_trajectory_grpo_score" if method == "reflect" else "compute_math_score",
                     "custom_reward_function.path": "rgrpo.core_trajectory_grpo_decoupled_dg" if method == "reflect" else "rgrpo.core_hybrid_grpo_v2"})
        if method == "reflect":
            args.update({"algorithm.reflection_grpo.enable": True, "algorithm.reflection_grpo.max_iters": 2,
                         "algorithm.grpo_skip_std_for_uniform_outcome": True, "algorithm.filter_groups.enable": False})
    if method == "score" and stage == 2 and not probe:
        args.update({"trainer.resume_mode": "resume_path",
                     "trainer.resume_from_path": str(run / f"global_step_{c['score_stage1_steps']}"),
                     "trainer.val_before_train": False})
    if evaluate:
        args.update({"trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(Path(evaluate).resolve()),
                     "trainer.val_before_train": True, "trainer.val_only": True})
    for key, value in args.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    OmegaConf.resolve(cfg)
    return cfg
