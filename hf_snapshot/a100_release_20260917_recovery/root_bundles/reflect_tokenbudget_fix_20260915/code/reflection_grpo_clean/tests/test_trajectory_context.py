"""Regression tests for sampling-context-correct reflection policy gradients."""
import os
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from rgrpo.trajectory_context import PREFIX, SEGMENT_KEYS, pack_segment_contexts
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor


@pytest.fixture(autouse=True)
def cpu_test_runtime(monkeypatch, request):
    import verl.utils.debug.performance as performance
    import verl.utils.torch_functional as functional
    # The installed flash-attn CE kernel and memory logger assume CUDA even
    # for CPU models. Use the existing torch fallback for CPU-only tests.
    if request.node.name != "test_cuda_remove_padding_context_forward":
        monkeypatch.setattr(functional, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
        monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0", "0", "0", "0"))


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]

    def batch_decode(self, rows, **kwargs):
        return ["".join(chr(int(i) - 2) for i in row if int(i) > 1) for row in rows]


def tensor(text):
    return torch.tensor([CharacterTokenizer().encode(text)])


def collect(monkeypatch, full_chain=True, max_iters=3, prompt_limit=1024, first_reflection="Fix A."):
    """Use the actual collector, replacing only model generation/tokenization."""
    monkeypatch.setenv("TRAJ_GRPO_TRAIN_FULL_CHAIN", str(int(full_chain)))
    monkeypatch.setenv("TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY", "0")
    header = "QUESTION: 2+2?\nANSWER:"
    stages = [
        f"WRONG_A<reflection>{first_reflection}</reflection>DISCARDED_A",
        "WRONG_B<reflection>Fix B.</reflection>DISCARDED_B",
        "FINAL: 4.",
    ]
    collector = object.__new__(TrajectoryCollector)
    collector.tokenizer = CharacterTokenizer()
    collector.config = OmegaConf.create({
        "data": {"max_prompt_length": prompt_limit, "max_response_length": 128},
        "algorithm": {"reflection_grpo": {"max_iters": max_iters}},
    })

    def initial(*args):
        inputs = tensor(header + stages[0])
        return DataProto.from_dict({
            "prompts": tensor(header), "responses": tensor(stages[0]),
            "input_ids": inputs, "attention_mask": torch.ones_like(inputs),
            "position_ids": torch.arange(inputs.shape[1]).unsqueeze(0),
        }, meta_info={"eos_token_id": 1}), 0.

    collector.no_env_loop = initial

    class Worker:
        world_size = 1

        def __init__(self):
            self.prompts = []

        def generate_sequences(self, data):
            self.prompts.append(data.batch["input_ids"][0][data.batch["attention_mask"][0].bool()].tolist())
            return DataProto.from_dict({"responses": tensor(stages[len(self.prompts)])})

    worker = Worker()
    result, _ = collector.reflection_grpo_loop(None, worker, is_train=True)
    return result, worker.prompts, header, stages


@pytest.mark.parametrize("full_chain", [True, False])
@pytest.mark.parametrize("max_iters", [1, 2, 3])
def test_collector_preserves_actual_prompts_and_action_mapping(monkeypatch, full_chain, max_iters):
    data, sampled_prompts, header, stages = collect(monkeypatch, full_chain, max_iters)
    lengths = data.batch[PREFIX + "lengths"][0]
    offsets = data.batch[PREFIX + "offsets"][0]
    response_width = data.batch[PREFIX + "responses"].shape[-1]
    prompt_width = data.batch[PREFIX + "input_ids"].shape[-1] - response_width
    expected_prompts = [CharacterTokenizer().encode(header)] + sampled_prompts
    assert len(lengths) == max_iters
    for slot, expected in enumerate(expected_prompts):
        ids = data.batch[PREFIX + "input_ids"][0, slot]
        mask = data.batch[PREFIX + "attention_mask"][0, slot]
        assert ids[:prompt_width][mask[:prompt_width].bool()].tolist() == expected
        offset, length = int(offsets[slot]), int(lengths[slot])
        assert torch.equal(data.batch["responses"][0, offset:offset + length],
                           data.batch[PREFIX + "responses"][0, slot, :length])
        assert data.batch[PREFIX + "position_ids"][0, slot, prompt_width].item() == len(expected)
    if max_iters > 1:
        assert "WRONG_A" not in CharacterTokenizer().batch_decode([sampled_prompts[0]])[0]
        assert "DISCARDED_A" not in CharacterTokenizer().batch_decode(data.batch["responses"])[0]
        assert int(data.batch["loss_mask"][0, 0]) == int(full_chain)
    if max_iters > 2:
        final_prompt = CharacterTokenizer().batch_decode([sampled_prompts[1]])[0]
        assert "Fix A" not in final_prompt and "Fix B" in final_prompt


def make_actor(seed=7):
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=512, attention_dropout=0.,
        attn_implementation="eager",
    ))
    config = OmegaConf.create({
        "use_remove_padding": False, "use_fused_kernels": False,
        "use_torch_compile": False, "ulysses_sequence_parallel_size": 1,
        "ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 1,
        "ppo_max_token_len_per_gpu": 1024, "use_dynamic_bsz": False,
        "ppo_epochs": 1, "use_kl_loss": False, "grad_clip": 1.,
        "clip_ratio": .2, "clip_ratio_low": .2, "clip_ratio_high": .26,
        "entropy_coeff": 0., "loss_agg_mode": "seq-mean-token-mean",
        "policy_loss": {"loss_mode": "vanilla"},
    })
    actor = DataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=.01))
    actor.device_name = "cpu"
    return actor


def mixed_batch():
    # Trajectory 0 has two segments; trajectory 1 is direct (one inactive slot).
    segments = [[([2, 3], [4, 5]), ([2, 3, 9], [6])], [([7], [8])]]
    responses = torch.tensor([[4, 5, 6], [8, 0, 0]])
    inputs = torch.tensor([[2, 3, 4, 5, 6], [0, 7, 8, 0, 0]])
    mask = (inputs != 0).long()
    data = DataProto.from_dict({
        "responses": responses, "input_ids": inputs, "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        "loss_mask": (responses != 0).long(),
        **pack_segment_contexts(segments, responses, 0),
    }, meta_info={"temperature": 1., "micro_batch_size": 1, "max_token_len": 1024,
                  "use_dynamic_bsz": False, "multi_turn": True})
    return data, segments


def independent_probabilities(actor, segments):
    rows = []
    for parts in segments:
        probabilities = []
        for prompt, reply in parts:
            x = torch.tensor([prompt + reply])
            _, logp = actor._forward_micro_batch({
                "input_ids": x, "attention_mask": torch.ones_like(x),
                "position_ids": torch.arange(x.shape[1]).unsqueeze(0),
                "responses": torch.tensor([reply]),
            }, temperature=1.)
            probabilities.append(logp[0])
        rows.append(torch.cat(probabilities))
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)


@pytest.mark.parametrize("dynamic", [False, True])
def test_actor_probabilities_and_gradients_match_independent_segments(dynamic):
    actor = make_actor()
    data, segments = mixed_batch()
    data.meta_info["use_dynamic_bsz"] = dynamic
    data.meta_info["max_token_len"] = 8  # force dynamic splitting/reordering
    actual, _ = actor.compute_log_prob(data)
    expected = independent_probabilities(actor, segments)
    valid = data.batch["loss_mask"].bool()
    assert torch.allclose(actual[valid], expected[valid], atol=.02, rtol=.005)
    # The old concatenated-history forward must differ on the continuation.
    plain = {key: data.batch[key] for key in ("input_ids", "attention_mask", "position_ids", "responses")}
    _, wrong = actor._forward_micro_batch(plain, temperature=1.)
    assert abs(float(wrong[0, 2] - expected[0, 2])) > 1e-4

    _, differentiable = actor._forward_micro_batch(data.batch, temperature=1.)
    loss = agg_loss(-differentiable, data.batch["loss_mask"], "seq-mean-token-mean")
    loss.backward()
    actual_grad = actor.actor_module.model.embed_tokens.weight.grad.clone()
    actor.actor_module.zero_grad()
    expected = independent_probabilities(actor, segments)
    # A trajectory has weight 1/2 regardless of its number of segments.
    expected_loss = -(expected[0, :3].mean() + expected[1, :1].mean()) / 2
    expected_loss.backward()
    expected_grad = actor.actor_module.model.embed_tokens.weight.grad
    assert torch.allclose(actual_grad, expected_grad, atol=.015, rtol=.04)
    assert torch.isfinite(actual_grad).all()


def test_real_actor_update_uses_segment_contexts(monkeypatch):
    import verl.workers.actor.dp_actor as actor_module
    monkeypatch.setattr(actor_module, "get_torch_device", lambda: torch.cpu)
    actor = make_actor()
    data, _ = mixed_batch()
    old, _ = actor.compute_log_prob(data)
    data.batch["old_log_probs"] = old
    data.batch["advantages"] = data.batch["loss_mask"].float()
    before = actor.actor_module.model.embed_tokens.weight.detach().clone()
    seen = []
    original = actor._forward_micro_batch

    def spy(batch=None, **kwargs):
        batch = kwargs.pop("micro_batch", batch)
        seen.append(SEGMENT_KEYS[0] in batch)
        return original(batch, **kwargs)

    monkeypatch.setattr(actor, "_forward_micro_batch", spy)
    metrics = actor.update_policy(data)
    assert sum(seen) == 2  # two trajectory micro-batches, both retain context tensors
    assert torch.isfinite(torch.tensor(metrics["actor/grad_norm"])).all()
    assert not torch.equal(before, actor.actor_module.model.embed_tokens.weight)


def test_dataproto_reordering_and_missing_context_fail_closed():
    data, segments = mixed_batch()
    reordered = data.select_idxs(np.array([1, 0]))
    assert torch.equal(reordered.batch[PREFIX + "lengths"], data.batch[PREFIX + "lengths"].flip(0))
    shards = reordered.chunk(2)
    assert torch.equal(DataProto.concat(shards).batch[PREFIX + "input_ids"], reordered.batch[PREFIX + "input_ids"])
    with pytest.raises(ValueError, match="does not match"):
        pack_segment_contexts(segments, torch.ones_like(data.batch["responses"]), 0)
    data.batch.pop(PREFIX + "lengths")
    with pytest.raises(ValueError, match="Incomplete"):
        make_actor()._forward_micro_batch(data.batch, temperature=1.)


def test_collector_records_truncated_continuation_prompt(monkeypatch):
    data, prompts, _, _ = collect(monkeypatch, max_iters=2, prompt_limit=23, first_reflection="x" * 90)
    assert len(prompts[0]) == 23 + 128 // 2
    rwidth = data.batch[PREFIX + "responses"].shape[-1]
    pwidth = data.batch[PREFIX + "input_ids"].shape[-1] - rwidth
    ids = data.batch[PREFIX + "input_ids"][0, 1, :pwidth]
    mask = data.batch[PREFIX + "attention_mask"][0, 1, :pwidth]
    assert ids[mask.bool()].tolist() == prompts[0]


@pytest.mark.skipif(os.environ.get("RUN_CUDA_CONTEXT_TEST") != "1", reason="explicit opt-in for a free GPU")
def test_cuda_remove_padding_context_forward(monkeypatch):
    """Exercise the same FlashAttention/remove-padding path as the run script."""
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    import verl.workers.actor.dp_actor as actor_module

    actor = make_actor()
    actor.actor_module.config._attn_implementation = "flash_attention_2"
    actor.actor_module = Qwen2ForCausalLM(actor.actor_module.config).to(device="cuda", dtype=torch.bfloat16)
    apply_monkey_patch(actor.actor_module, use_remove_padding=True)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.01)
    actor.device_name = "cuda"
    actor.use_remove_padding = True
    data, _ = mixed_batch()
    data = data.to("cuda")
    # Independent padded forwards serve as the oracle for the packed path.
    flat = {name: data.batch[PREFIX + name].flatten(0, 1)
            for name in ("input_ids", "attention_mask", "position_ids", "responses")}
    actor.use_remove_padding = False
    with torch.no_grad():
        _, oracle = actor._forward_micro_batch(flat, temperature=1.)
    expected = torch.stack([torch.cat([oracle[0, :2], oracle[1, :1]]),
                            torch.cat([oracle[2, :1], oracle.new_zeros(2)])])
    actor.use_remove_padding = True
    old, entropy = actor.compute_log_prob(data, calculate_entropy=True)
    mask = data.batch["loss_mask"].bool()
    assert torch.allclose(old[mask], expected[mask], atol=.03, rtol=.005)
    assert torch.isfinite(entropy).all()
    data.batch["old_log_probs"] = old
    data.batch["advantages"] = mask.float()
    actor.config.use_dynamic_bsz = True
    actor.config.ppo_max_token_len_per_gpu = 8
    before = actor.actor_module.model.embed_tokens.weight.detach().clone()
    metrics = actor.update_policy(data)
    assert torch.isfinite(torch.tensor(metrics["actor/grad_norm"])).all()
    assert not torch.equal(before, actor.actor_module.model.embed_tokens.weight)
