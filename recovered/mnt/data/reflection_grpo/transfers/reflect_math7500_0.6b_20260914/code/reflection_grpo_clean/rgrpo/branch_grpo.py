"""Root-balanced GRPO for recursively branched reflection trajectories.

Each initial policy sample owns mass 1. Splitting into K children divides mass
by K. Both group statistics and PPO loss use this mass; padding has mass zero.
The reward formula is inherited from REFLECT+, without the decoupled patch.
"""
import numpy as np
import torch


WEIGHT = "branch_weight"
ROOT = "branch_root_id"


def compute_trajectory_grpo_score(*args, extra_info=None, **kwargs):
    from rgrpo.core_trajectory_grpo import compute_trajectory_grpo_score as score
    # Padding must not invoke the expensive mathematical verifier.
    if extra_info and extra_info.get("branch_padding", False):
        return {"score": 0., "task_reward": 0., "is_correct": 0.}
    return score(*args, extra_info=extra_info, **kwargs)


def compute_branch_advantage(data, norm_adv_by_std=True, skip_uniform=True):
    """Weighted population mean/variance per question, not per reflection node."""
    weights = data.batch[WEIGHT].float()
    scores = data.batch["token_level_rewards"].sum(-1).float()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid branch weights")
    root_ids = data.non_tensor_batch[ROOT]
    uids = data.non_tensor_batch["uid"]
    outcomes = data.non_tensor_batch.get("is_correct", data.non_tensor_batch.get("task_reward"))
    advantages = torch.zeros_like(scores)
    groups = {}
    roots = {}
    for row, (uid, root) in enumerate(zip(uids, root_ids)):
        groups.setdefault(str(uid), []).append(row)
        roots.setdefault((str(uid), str(root)), []).append(row)
    for rows in roots.values():
        if not torch.isclose(weights[rows].sum(), weights.new_tensor(1.), atol=1e-6):
            raise ValueError("Each initial sample must retain total branch mass 1")
    for rows in groups.values():
        index = torch.as_tensor(rows, device=scores.device)
        w, r = weights[index], scores[index]
        valid = w > 0
        mean = (w * r).sum() / w.sum()
        variance = (w * (r - mean).square()).sum() / w.sum()
        a = r - mean
        uniform = False
        if skip_uniform and outcomes is not None:
            correct = torch.as_tensor(np.asarray(outcomes)[rows].astype(float), device=scores.device)[valid] > .5
            uniform = bool(correct.all() or (~correct).all())
        if norm_adv_by_std and not uniform:
            a = a / (variance.sqrt() + 1e-6)
        advantages[index] = torch.where(valid, a, 0.)
    width = data.batch["responses"].shape[-1]
    mask = data.batch.get("loss_mask", data.batch["response_mask"])[:, -width:]
    data.batch["advantages"] = advantages.unsqueeze(-1) * mask
    data.batch["returns"] = data.batch["advantages"]
    return data


def branch_metrics(data):
    w = data.batch[WEIGHT].float()
    mass = w.sum()
    metrics = {"branch/root_count": float(mass),
               "branch/leaf_count": int((w > 0).sum()),
               "branch/padding_count": int((w == 0).sum())}
    scores = data.batch["token_level_rewards"].sum(-1)
    metrics["branch/reward"] = float((w * scores).sum() / mass)
    for key, label in (("task_reward", "accuracy"), ("num_reflections", "reflections")):
        values = data.non_tensor_batch.get(key)
        if values is not None:
            values = torch.as_tensor(np.nan_to_num(np.asarray(values, dtype=float)), device=w.device)
            metrics[f"branch/{label}"] = float((w * values).sum() / mass)
    return metrics


def root_balanced_order(data, world_size):
    """Assign whole equal-sized root blocks to ranks; never split a tree."""
    from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
    slots = int(data.meta_info["branch_slots"])
    count = len(data) // slots
    if len(data) % slots or count % world_size:
        raise ValueError("Initial sample count must be divisible by actor world size")
    costs = data.batch["reflection_segment_attention_mask"].sum(dim=(1, 2)).reshape(count, slots).sum(-1).tolist()
    partitions = get_seqlen_balanced_partitions(costs, world_size, equal_size=True)
    order = [root * slots + slot for group in partitions for root in group for slot in range(slots)]
    return torch.tensor(order)


def update_branch_policy(actor, data):
    """Root-sized PPO minibatches, leaf microbatches and mass-normalized loss.

    Multiplying advantages by nonnegative leaf mass commutes with vanilla PPO
    clipping. The final microbatch scale converts its mean to a weighted sum
    divided by the number of initial samples in this optimizer minibatch.
    """
    from rgrpo.trajectory_context import SEGMENT_KEYS
    from verl.trainer.ppo.core_algos import compute_policy_loss
    from verl.utils.seqlen_balancing import rearrange_micro_batches
    from verl.utils.py_functional import append_to_dict

    config = actor.config
    if config.loss_agg_mode != "seq-mean-token-mean" or config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Branch GRPO requires vanilla PPO and seq-mean-token-mean")
    if config.use_kl_loss or config.entropy_coeff != 0:
        raise ValueError("Branch GRPO currently requires KL loss and entropy bonus disabled")
    slots = int(data.meta_info["branch_slots"])
    if len(data) % slots:
        raise ValueError("Actor received a partial branch tree")
    root_ids = np.asarray(data.non_tensor_batch[ROOT]).reshape(-1, slots)
    if not np.all(root_ids == root_ids[:, :1]):
        raise ValueError("Branch root blocks were split/reordered incorrectly")
    weights = data.batch[WEIGHT].reshape(-1, slots)
    if not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-6):
        raise ValueError("Actor root masses must equal 1")
    keys = ["responses", "input_ids", "attention_mask", "position_ids", "loss_mask",
            "old_log_probs", "advantages", WEIGHT, *SEGMENT_KEYS]
    batch = data.select(batch_keys=keys).batch
    minibatch_rows = config.ppo_mini_batch_size * slots
    if len(batch) % minibatch_rows:
        raise ValueError("Initial samples must divide evenly into PPO minibatches")
    metrics = {}
    for _ in range(config.ppo_epochs):
        for mini in batch.split(minibatch_rows):
            mass = mini[WEIGHT].sum().detach()
            if config.use_dynamic_bsz:
                micros, _ = rearrange_micro_batches(
                    mini, config.ppo_max_token_len_per_gpu * actor.ulysses_sequence_parallel_size)
            else:
                micros = mini.split(config.ppo_micro_batch_size_per_gpu)
            actor.actor_optimizer.zero_grad()
            for micro in micros:
                _, logp = actor._forward_micro_batch(micro, temperature=data.meta_info["temperature"])
                weighted_adv = micro["advantages"] * micro[WEIGHT].unsqueeze(-1)
                pg, clip, kl, lower = compute_policy_loss(
                    old_log_prob=micro["old_log_probs"], log_prob=logp,
                    advantages=weighted_adv, response_mask=micro["loss_mask"],
                    cliprange=config.clip_ratio, cliprange_low=config.clip_ratio_low,
                    cliprange_high=config.clip_ratio_high, clip_ratio_c=config.get("clip_ratio_c", 3.),
                    loss_agg_mode="seq-mean-token-mean")
                loss = pg * (len(micro) / mass)
                loss.backward()
                # Sum these contributions, not an unweighted mean of microbatch means.
                metrics["branch/pg_loss"] = metrics.get("branch/pg_loss", 0.) + float(loss.detach())
            norm = actor._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": float(norm.detach())})
            metrics["branch/optimizer_steps"] = metrics.get("branch/optimizer_steps", 0) + 1
    actor.actor_optimizer.zero_grad()
    return metrics
