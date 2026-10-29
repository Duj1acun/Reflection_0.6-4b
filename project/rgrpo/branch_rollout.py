"""Four-way reflection rollouts with conserved mass and exact segment contexts."""
import os
import uuid

import numpy as np
import torch

from rgrpo.branch_grpo import ROOT, WEIGHT
from rgrpo.core_rgrpo import REFL_PATTERN
from rgrpo.trajectory_context import pack_segment_contexts
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask


def truthy(name, default):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "n"}


def parse_reflection(tokenizer, ids):
    text = tokenizer.batch_decode([ids], skip_special_tokens=True)[0]
    match = REFL_PATTERN.search(text)
    if match is None:
        return None
    end = min(len(ids), len(tokenizer.encode(text[:match.end()], add_special_tokens=False)))
    start = min(end, len(tokenizer.encode(text[:match.start()], add_special_tokens=False)))
    # Include the token containing '<', including a possible BPE-merged space.
    opener = len(tokenizer.encode(text[:match.start() + 1], add_special_tokens=False))
    start = min(start, max(0, min(opener - 1, len(ids))))
    reflection = tokenizer.encode(match.group(0).rstrip() + "\n", add_special_tokens=False)
    return start, end, reflection


def generate_children(collector, worker, prompts, meta_info, count):
    tokenizer = collector.tokenizer
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    raw = np.empty(len(prompts), dtype=object)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = torch.tensor(prompt)
        mask[row, -len(prompt):] = 1
        raw[row] = np.asarray(prompt, dtype=np.int64)
    meta = dict(meta_info)
    meta.update(n=count, do_sample=True, validate=False)
    eos = meta.get("eos_token_id", tokenizer.eos_token_id)
    meta["stop_token_ids"] = [eos] if isinstance(eos, int) else list(eos)
    batch = DataProto.from_dict(
        {"input_ids": ids, "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask)},
        non_tensors={"raw_prompt_ids": raw}, meta_info=meta)
    padded, padding = pad_dataproto_to_divisor(batch, worker.world_size)
    generated = unpad_dataproto(worker.generate_sequences(padded), padding * count)
    if len(generated) != len(prompts) * count:
        raise ValueError("Branch rollout lost children; refusing to change parent mass")
    return [row[row != tokenizer.pad_token_id].tolist() for row in generated.batch["responses"]]


def branch_reflection_rollout(collector, gen_batch, worker):
    config, tokenizer = collector.config, collector.tokenizer
    branching = config.algorithm.reflection_grpo.get("branching", {})
    count = int(branching.get("children", 4))
    max_iters = int(config.algorithm.reflection_grpo.max_iters)
    if count < 1 or max_iters < 1:
        raise ValueError("children and max_iters must be positive")
    base, _ = collector.no_env_loop(gen_batch, worker, is_train=True)
    if len(base) == 0:
        raise ValueError("Empty initial branch rollout")
    full_chain = truthy("TRAJ_GRPO_TRAIN_FULL_CHAIN", "1")
    last_only = truthy("TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY", "0")
    prompt_width = base.batch["prompts"].shape[-1]
    headers = [base.batch["prompts"][row][base.batch["attention_mask"][row, :prompt_width].bool()].tolist()
               for row in range(len(base))]
    eos = base.meta_info.get("eos_token_id", tokenizer.eos_token_id)
    eos = {eos} if isinstance(eos, int) else set(eos)
    active = []
    leaves = [[] for _ in range(len(base))]
    for root, header in enumerate(headers):
        response = base.batch["responses"][root]
        active.append(dict(root=root, prompt=header,
                           stage=response[response != tokenizer.pad_token_id].tolist(),
                           ids=[], mask=[], segments=[], weight=1., reflections=0,
                           last_start=-1, truncated=False, terminated=False))
    while active:
        pending = []
        for node in active:
            parsed = parse_reflection(tokenizer, node["stage"])
            split = parsed is not None and node["reflections"] < max_iters - 1
            retained = node["stage"][:parsed[1]] if split else node["stage"]
            start = parsed[0] if parsed is not None else 0
            stage_mask = ([int(full_chain)] * start + [1] * (len(retained) - start)
                          if parsed is not None else [1] * len(retained))
            updated = dict(node)
            updated.update(
                ids=node["ids"] + retained,
                mask=node["mask"] + stage_mask,
                segments=node["segments"] + ([(list(node["prompt"]), list(retained))] if retained else []),
                last_start=len(node["ids"]) + start if parsed is not None else node["last_start"],
                truncated=node["truncated"] or (len(node["stage"]) >= config.data.max_response_length
                                                and not eos.intersection(node["stage"])),
            )
            if not split:
                updated["terminated"] = parsed is not None
                leaves[node["root"]].append(updated)
                continue
            updated["reflections"] += 1
            cap = int(config.data.max_prompt_length + config.data.max_response_length // 2)
            updated["prompt"] = (headers[node["root"]] + parsed[2])[-cap:]
            updated["weight"] /= count
            pending.append(updated)
        if not pending:
            break
        replies = generate_children(collector, worker, [node["prompt"] for node in pending], base.meta_info, count)
        active = [dict(node, stage=replies[i * count + child])
                  for i, node in enumerate(pending) for child in range(count)]
    result = pack_leaves(base, leaves, tokenizer, full_chain, last_only)
    for info in result.non_tensor_batch["extra_info"]:
        info["phi"] = max_iters
    return result, 0.


def pack_leaves(base, leaves, tokenizer, full_chain, last_only):
    """Pad each root to equal slots for root-preserving DP dispatch.

    Padding has zero statistical/loss mass and is not judged by the reward
    function. A one-token dummy loss mask keeps generic PPO metrics finite.
    """
    slots = max(map(len, leaves))
    root_rows = np.repeat(np.arange(len(base)), slots)
    result = base.select_idxs(root_rows)
    result.batch.pop("rollout_log_probs", None)
    width = max(1, max(len(leaf["ids"]) for group in leaves for leaf in group))
    responses = result.batch["responses"].new_full((len(result), width), tokenizer.pad_token_id)
    attention = torch.zeros_like(responses)
    loss_mask = torch.zeros_like(responses)
    weights = torch.zeros(len(result), dtype=torch.float32, device=responses.device)
    extra = np.empty(len(result), dtype=object)
    root_ids = np.empty(len(result), dtype=object)
    segments = []
    eos = tokenizer.eos_token_id
    eos = eos[0] if isinstance(eos, (tuple, list)) else eos
    source_info = base.non_tensor_batch.get("extra_info", [{}] * len(base))
    for root, group in enumerate(leaves):
        if not np.isclose(sum(leaf["weight"] for leaf in group), 1.):
            raise ValueError("Branch tree lost probability mass")
        root_id = str(uuid.uuid4())
        for slot in range(slots):
            row = root * slots + slot
            info = dict(source_info[root]) if isinstance(source_info[root], dict) else {}
            root_ids[row] = root_id
            if slot >= len(group):
                responses[row, 0] = eos
                attention[row, 0] = loss_mask[row, 0] = 1
                segments.append([([eos], [eos])])
                info.update(branch_padding=True, num_reflections=0, n_iters=1)
            else:
                leaf = group[slot]
                length = len(leaf["ids"])
                if not length:
                    raise ValueError("Empty leaf response cannot be trained")
                responses[row, :length] = responses.new_tensor(leaf["ids"])
                attention[row, :length] = 1
                mask = list(leaf["mask"])
                if last_only and not full_chain and leaf["last_start"] >= 0:
                    mask = [0] * leaf["last_start"] + [1] * (length - leaf["last_start"])
                loss_mask[row, :length] = loss_mask.new_tensor(mask)
                weights[row] = leaf["weight"]
                segments.append(leaf["segments"])
                info.update(branch_padding=False, num_reflections=leaf["reflections"],
                            n_iters=leaf["reflections"] + 1, terminated_by_cap=leaf["terminated"],
                            truncated=leaf["truncated"])
            extra[row] = info
    prompts = result.batch["prompts"]
    pwidth = prompts.shape[-1]
    prompt_mask = result.batch["attention_mask"][:, :pwidth]
    prompt_positions = result.batch["position_ids"][:, :pwidth]
    result.batch["responses"] = responses
    result.batch["input_ids"] = torch.cat([prompts, responses], -1)
    result.batch["attention_mask"] = torch.cat([prompt_mask, attention], -1)
    result.batch["position_ids"] = torch.cat([
        prompt_positions, prompt_positions[:, -1:] + torch.arange(1, width + 1, device=responses.device)], -1)
    result.batch["loss_mask"] = loss_mask
    result.batch[WEIGHT] = weights
    for key, value in pack_segment_contexts(segments, responses, tokenizer.pad_token_id).items():
        result.batch[key] = value
    result.non_tensor_batch["extra_info"] = extra
    result.non_tensor_batch[ROOT] = root_ids
    result.non_tensor_batch["traj_uid"] = np.array([str(uuid.uuid4()) for _ in range(len(result))], dtype=object)
    result.meta_info["branch_slots"] = slots
    return result
