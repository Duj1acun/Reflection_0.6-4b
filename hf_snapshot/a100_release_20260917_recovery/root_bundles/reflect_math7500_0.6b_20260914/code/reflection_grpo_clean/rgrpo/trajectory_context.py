"""Score trajectory tokens under the exact prompts used to sample each segment.

The concatenated trajectory is still the unit of reward, GRPO grouping and loss
aggregation. These auxiliary tensors change only the policy forward context.
They retain the trajectory batch dimension for DataProto/FSDP sharding.
"""

import torch


PREFIX = "reflection_segment_"
SEGMENT_KEYS = tuple(PREFIX + name for name in (
    "input_ids", "attention_mask", "position_ids", "responses", "offsets", "lengths"
))


def pack_segment_contexts(trajectories, responses, pad_token_id):
    """Pack lists of (actual_prompt_ids, retained_response_ids) per trajectory.

    Only the sampled prefix retained by the collector is scored (generation
    after a reflection trigger may have been discarded). Inserted prompt tokens
    are context, never actions. Validate the mapping against the reward view.
    """
    batch_size = len(trajectories)
    if batch_size != responses.shape[0]:
        raise ValueError("Segment/trajectory batch sizes differ")
    slots = max(1, max((len(parts) for parts in trajectories), default=0))
    prompt_width = max(1, max((len(p) for parts in trajectories for p, _ in parts), default=0))
    response_width = max(1, max((len(r) for parts in trajectories for _, r in parts), default=0))
    shape = (batch_size, slots)
    inputs = responses.new_full((*shape, prompt_width + response_width), pad_token_id)
    segment_responses = responses.new_full((*shape, response_width), pad_token_id)
    attention = torch.zeros_like(inputs)
    offsets = responses.new_zeros(shape)
    lengths = responses.new_zeros(shape)
    # Inactive slots use a harmless two-token sequence, never an all-padding
    # FlashAttention row. Its outputs have zero weight but stay in the graph.
    attention[:, :, prompt_width - 1:prompt_width + 1] = 1
    for row, parts in enumerate(trajectories):
        offset = 0
        for slot, (prompt, reply) in enumerate(parts):
            if not prompt or not reply:
                raise ValueError("A recorded segment must have a prompt and response")
            p, r = len(prompt), len(reply)
            reply_tensor = responses.new_tensor(reply)
            if offset + r > responses.shape[1] or not torch.equal(responses[row, offset:offset + r], reply_tensor):
                raise ValueError("Segment response does not match the retained trajectory")
            inputs[row, slot, prompt_width - p:prompt_width] = responses.new_tensor(prompt)
            inputs[row, slot, prompt_width:prompt_width + r] = reply_tensor
            segment_responses[row, slot, :r] = reply_tensor
            attention[row, slot].zero_()
            attention[row, slot, prompt_width - p:prompt_width + r] = 1
            offsets[row, slot], lengths[row, slot] = offset, r
            offset += r
        if torch.any(responses[row, offset:] != pad_token_id):
            raise ValueError("Some trajectory tokens have no sampling context")
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return dict(zip(SEGMENT_KEYS, (inputs, attention, positions, segment_responses, offsets, lengths)))


def forward_segment_contexts(micro_batch, forward, temperature, calculate_entropy=False):
    """Run the ordinary actor forward on segments and scatter to trajectory IDs.

    A single forward includes inactive slots, so all FSDP ranks participate in
    the same forward/backward collectives even with different reflection counts.
    index_add preserves gradients; loss normalization remains trajectory-level.
    """
    missing = [key for key in SEGMENT_KEYS if key not in micro_batch]
    if missing:
        raise ValueError(f"Incomplete reflection sampling contexts: {missing}")
    inputs = micro_batch[PREFIX + "input_ids"]
    batch_size, slots, _ = inputs.shape
    flat = {
        name: micro_batch[PREFIX + name].flatten(0, 1)
        for name in ("input_ids", "attention_mask", "position_ids", "responses")
    }
    entropy, log_probs = forward(flat, temperature=temperature, calculate_entropy=calculate_entropy)
    width = flat["responses"].shape[-1]
    trajectory_width = micro_batch["responses"].shape[-1]
    offsets = micro_batch[PREFIX + "offsets"].unsqueeze(-1)
    lengths = micro_batch[PREFIX + "lengths"].unsqueeze(-1)
    token = torch.arange(width, device=inputs.device).view(1, 1, -1)
    valid = token < lengths
    row = torch.arange(batch_size, device=inputs.device).view(-1, 1, 1)
    destination = row * trajectory_width + offsets + token
    destination = torch.where(valid, destination, 0).reshape(-1)

    def scatter(values):
        values = values.reshape(batch_size, slots, width)
        values = torch.where(valid, values, 0).reshape(-1)
        return values.new_zeros(batch_size * trajectory_width).index_add(
            0, destination, values
        ).view(batch_size, trajectory_width)

    return scatter(entropy) if calculate_entropy else None, scatter(log_probs)
