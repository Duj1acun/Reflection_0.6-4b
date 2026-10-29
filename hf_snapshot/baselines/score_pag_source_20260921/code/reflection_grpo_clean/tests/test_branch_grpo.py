"""Branch rollout, weighted optimizer, and shared-launcher regression tests."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from rgrpo.branch_grpo import ROOT, WEIGHT, compute_branch_advantage, root_balanced_order
from rgrpo.trajectory_context import PREFIX
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.reward_manager.naive import NaiveRewardManager
from test_trajectory_context import CharacterTokenizer, cpu_test_runtime, make_actor, mixed_batch


def collect_tree(monkeypatch, recursive=False, roots=8):
    monkeypatch.setenv("TRAJ_GRPO_TRAIN_FULL_CHAIN", "1")
    monkeypatch.setenv("TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY", "0")
    collector = object.__new__(TrajectoryCollector)
    tok = collector.tokenizer = CharacterTokenizer()
    collector.config = OmegaConf.create({
        "data": {"max_prompt_length": 128, "max_response_length": 128},
        "algorithm": {"reflection_grpo": {"enable": True, "max_iters": 3 if recursive else 2,
                                          "branching": {"enable": True, "children": 4}}},
    })
    texts = ["wrong<reflection>Fix A.</reflection>discarded"] * min(2, roots) + ["answer"] * max(0, roots - 2)
    prompt = torch.tensor([tok.encode("Q:")] * roots)
    replies = torch.nn.utils.rnn.pad_sequence([torch.tensor(tok.encode(t)) for t in texts], batch_first=True)
    inputs = torch.cat([prompt, replies], -1)
    mask = (inputs != 0).long()
    base = DataProto.from_dict({
        "prompts": prompt, "responses": replies, "input_ids": inputs,
        "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
    }, non_tensors={
        "uid": np.array(["question"] * roots, dtype=object),
        "extra_info": np.array([{} for _ in range(roots)], dtype=object),
        "data_source": np.array(["math"] * roots, dtype=object),
        "reward_model": np.array([{"ground_truth": "4"} for _ in range(roots)], dtype=object),
    }, meta_info={"eos_token_id": 1})
    collector.no_env_loop = lambda *args, **kwargs: (base, 0.)

    class Worker:
        world_size = 1

        def __init__(self):
            self.calls = []

        def generate_sequences(self, data):
            assert data.meta_info["n"] == 4
            assert data.meta_info["validate"] is False
            prompts = [ids[m.bool()].tolist() for ids, m in zip(data.batch["input_ids"], data.batch["attention_mask"])]
            self.calls.append(prompts)
            stages = ["final4"] * (len(data) * 4)
            if recursive and len(self.calls) == 1:
                stages[0] = "bad<reflection>Fix B.</reflection>discarded2"
            return DataProto.from_dict({"responses": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(tok.encode(t)) for t in stages], batch_first=True)})

    worker = Worker()
    result, _ = collector.multi_turn_loop(None, worker, envs=None, is_train=True)
    return result, worker, collector


@pytest.mark.parametrize("recursive,leaf_count,slots", [(False, 14, 4), (True, 17, 7)])
def test_tree_mass_and_exact_contexts(monkeypatch, recursive, leaf_count, slots):
    data, worker, _ = collect_tree(monkeypatch, recursive)
    weights = data.batch[WEIGHT]
    assert len(data) == 8 * slots
    assert int((weights > 0).sum()) == leaf_count
    assert torch.allclose(weights.reshape(8, slots).sum(-1), torch.ones(8))
    assert weights.sum() == 8
    assert len(set(data.non_tensor_batch["uid"])) == 1
    assert len(set(data.non_tensor_batch[ROOT])) == 8
    tok = CharacterTokenizer()
    assert all("discarded" not in t for t in tok.batch_decode(data.batch["responses"]))
    assert all(tok.batch_decode([p])[0] == "Q:<reflection>Fix A.</reflection>\n" for p in worker.calls[0])
    if recursive:
        assert (weights == 1 / 16).sum() == 4
        assert tok.batch_decode([worker.calls[1][0]])[0] == "Q:<reflection>Fix B.</reflection>\n"
    response_width = data.batch[PREFIX + "responses"].shape[-1]
    prompt_width = data.batch[PREFIX + "input_ids"].shape[-1] - response_width
    for row in range(len(data)):
        for slot, length in enumerate(data.batch[PREFIX + "lengths"][row]):
            length = int(length)
            if not length:
                continue
            offset = int(data.batch[PREFIX + "offsets"][row, slot])
            assert torch.equal(data.batch["responses"][row, offset:offset + length],
                               data.batch[PREFIX + "responses"][row, slot, :length])
            if weights[row] > 0 and slot > 0:
                ids = data.batch[PREFIX + "input_ids"][row, slot, :prompt_width]
                mask = data.batch[PREFIX + "attention_mask"][row, slot, :prompt_width]
                assert ids[mask.bool()].tolist() in worker.calls[slot - 1]


def test_validation_does_not_branch(monkeypatch):
    _, _, collector = collect_tree(monkeypatch)
    collector.reflection_grpo_loop = lambda **kwargs: (kwargs["is_train"], "ordinary")
    assert collector.multi_turn_loop(None, None, None, is_train=False) == (False, "ordinary")


def test_weighted_group_statistics_and_padding(monkeypatch):
    data, _, _ = collect_tree(monkeypatch)
    w = data.batch[WEIGHT]
    rewards = torch.zeros(len(data))
    rewards[0] = 1
    rewards[8] = 1  # one direct root, plus one of four children of root 0
    rewards[w == 0] = 100  # zero-mass padding must not enter statistics
    data.batch["token_level_rewards"] = rewards[:, None].expand_as(data.batch["responses"]) / data.batch["responses"].shape[-1]
    data.non_tensor_batch["is_correct"] = (rewards > .5).numpy().astype(float)
    compute_advantage(data, "grpo", norm_adv_by_std_in_grpo=True)
    mean = 1.25 / 8
    expected = (rewards - mean) / ((mean * (1 - mean)) ** .5 + 1e-6)
    actual = data.batch["advantages"][:, 0]
    assert torch.allclose(actual[w > 0], expected[w > 0], atol=1e-5)
    assert (actual[w == 0] == 0).all()
    assert abs(float((w * actual).sum())) < 1e-5
    # Uniform task outcome skips std, not centering; shaped reward still differs.
    data.non_tensor_batch["is_correct"] = np.zeros(len(data))
    compute_branch_advantage(data, skip_uniform=True)
    assert torch.allclose(data.batch["advantages"][w > 0, 0], (rewards - mean)[w > 0], atol=1e-6)


def test_root_balancing_and_reward_manager_padding(monkeypatch):
    data, _, _ = collect_tree(monkeypatch, recursive=True)
    data.reorder(root_balanced_order(data, 2))
    chunks = data.chunk(2)
    assert not set(chunks[0].non_tensor_batch[ROOT]) & set(chunks[1].non_tensor_batch[ROOT])
    for chunk in chunks:
        assert torch.allclose(chunk.batch[WEIGHT].reshape(-1, 7).sum(-1), torch.ones(4))
    from rgrpo.branch_grpo import compute_trajectory_grpo_score
    import rgrpo.core_trajectory_grpo as reward_module
    seen = []

    def score(**kwargs):
        seen.append(kwargs)
        return {"score": 1., "task_reward": 1., "is_correct": 1., "num_reflections": 2.}

    monkeypatch.setattr(reward_module, "compute_trajectory_grpo_score", score)
    monkeypatch.setenv("HYBRID_V2_PRECOMPUTE_MATH_SCORE", "0")
    tok = CharacterTokenizer()
    tok.decode = lambda ids, **kwargs: tok.batch_decode([ids], **kwargs)[0]
    result = NaiveRewardManager(tok, 0, compute_trajectory_grpo_score)(data, return_dict=True)
    assert len(seen) == 17
    assert all(len(v) == len(data) for v in result["reward_extra_info"].values())
    assert (result["reward_tensor"][data.batch[WEIGHT] == 0] == 0).all()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("entropy_coeff", [0., .001])
@pytest.mark.parametrize("padding", [False, True])
def test_duplicate_children_do_not_multiply_optimizer_gradient(monkeypatch, dynamic, entropy_coeff, padding):
    import verl.workers.actor.dp_actor as actor_module
    monkeypatch.setattr(actor_module, "get_torch_device", lambda: torch.cpu)
    results = []
    for children in (1, 4):
        actor = make_actor()
        actor.config.use_dynamic_bsz = dynamic
        actor.config.ppo_max_token_len_per_gpu = 16
        actor.config.entropy_coeff = entropy_coeff
        data, _ = mixed_batch()
        data = data.repeat(children, interleave=True)
        data.batch[WEIGHT] = torch.full((len(data),), 1 / children)
        if padding:
            data.batch[WEIGHT].zero_()
            data.batch[WEIGHT][::children] = 1.
        data.non_tensor_batch[ROOT] = np.repeat(np.array(["a", "b"], dtype=object), children)
        data.meta_info["branch_slots"] = children
        old, _ = actor.compute_log_prob(data)
        # Exercise clipped and unclipped terms with different sequence lengths.
        data.batch["old_log_probs"] = old.detach() + .35
        data.batch["advantages"] = torch.tensor([1., -.4]).repeat_interleave(children)[:, None] * data.batch["loss_mask"]
        metrics = actor.update_policy(data)
        assert metrics["branch/optimizer_steps"] == 1
        results.append((actor.actor_module.model.embed_tokens.weight.detach().clone(), metrics))
    assert torch.allclose(results[0][0], results[1][0], atol=3e-5, rtol=1e-4)
    assert results[0][1]["branch/policy_loss"] == pytest.approx(results[1][1]["branch/policy_loss"], abs=.003)


def test_adjust_batch_refuses_to_duplicate_leaves(monkeypatch):
    from agent_system.multi_turn_rollout.utils import adjust_batch
    data, _, _ = collect_tree(monkeypatch, roots=1)
    config = OmegaConf.create({"trainer": {"n_gpus_per_node": 1, "nnodes": 1},
                               "actor_rollout_ref": {"ref": {"log_prob_micro_batch_size_per_gpu": 32},
                                                     "rollout": {"log_prob_micro_batch_size_per_gpu": 32},
                                                     "actor": {"ppo_micro_batch_size_per_gpu": 4}}})
    with pytest.raises(ValueError, match="refusing to copy/delete"):
        adjust_batch(config, data)


def test_initial_sampling_unpads_all_repeated_outputs():
    collector = object.__new__(TrajectoryCollector)
    collector.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 4}}})
    prompt = torch.tensor([[2, 3], [4, 5], [6, 7]])
    batch = DataProto.from_dict({"input_ids": prompt, "attention_mask": torch.ones_like(prompt),
                                "position_ids": torch.tensor([[0, 1]] * 3)},
                               non_tensors={"raw_prompt_ids": np.array([list(p) for p in prompt.tolist()], dtype=object),
                                            "extra_info": np.array([{} for _ in range(3)], dtype=object)})

    class Worker:
        world_size = 2

        def generate_sequences(self, data):
            assert len(data) == 4  # 3 real prompts + 1 padding prompt
            prompts = data.batch["input_ids"].repeat_interleave(4, 0)
            responses = torch.ones((16, 1), dtype=torch.long)
            inputs = torch.cat([prompts, responses], -1)
            return DataProto.from_dict({"prompts": prompts, "responses": responses, "input_ids": inputs,
                                        "attention_mask": torch.ones_like(inputs),
                                        "position_ids": torch.arange(3).expand(16, 3)})

    result, _ = collector.no_env_loop(batch, Worker(), is_train=True)
    assert len(result) == 12  # remove four children of the padding prompt
    assert len(set(result.non_tensor_batch["uid"])) == 3
    assert torch.equal(result.batch["prompts"], prompt.repeat_interleave(4, 0))


def test_launcher_matches_shared_parameters_and_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = root / "training" / "rl"
    # Intercept only python3: no model, Ray or GPU is loaded for this test.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text('#!/usr/bin/python3\nimport json,sys\nprint("ARGV=" + json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SKIP_GPU_PREFLIGHT="1",
               MODEL_PATH=str(model), OUTPUT_ROOT=str(tmp_path / "output"),
               RAY_TMPDIR=str(tmp_path / "ray"), ACTOR_LR="2e-6", TRAIN_PROMPT_BSZ="32",
               PPO_MINI_BATCH_SIZE="32", MAX_ITERS="2")
    env.pop("FOURWAY_CONFIG", None)
    old = subprocess.run(["bash", str(scripts / "train_reflect_plus.sh")], env=env, text=True, capture_output=True, check=True)
    old_args = json.loads(next(line[5:] for line in old.stdout.splitlines() if line.startswith("ARGV=")))
    env["OUTPUT_ROOT"] = str(tmp_path / "dry-output")
    new = subprocess.run(["bash", str(scripts / "train_reflect_branch_grpo.sh"), "--dry-run"],
                         env=env, text=True, capture_output=True, check=True)
    new_args = shlex.split(next(line for line in new.stdout.splitlines() if line.startswith("python3 ")))[1:]
    assert not Path(env["OUTPUT_ROOT"]).exists()
    old_map = dict(arg.split("=", 1) for arg in old_args if "=" in arg)
    new_map = dict(arg.split("=", 1) for arg in new_args if "=" in arg)
    differences = {key for key in old_map.keys() | new_map.keys() if old_map.get(key) != new_map.get(key)}
    assert differences == {"custom_reward_function.path", "trainer.experiment_name", "trainer.default_local_dir",
                           "+algorithm.reflection_grpo.branching.enable", "+algorithm.reflection_grpo.branching.children"}
    assert new_map["actor_rollout_ref.actor.optim.lr"] == "2e-6"
    assert new_map["+algorithm.reflection_grpo.branching.children"] == "4"
    assert new_map["custom_reward_function.path"] == "rgrpo.branch_grpo"
    env["FOURWAY_CONFIG"] = "must-not-be-ignored.yaml"
    rejected = subprocess.run(["bash", str(scripts / "train_reflect_branch_grpo.sh"), "--dry-run"],
                              env=env, text=True, capture_output=True)
    assert rejected.returncode != 0 and "FOURWAY_CONFIG" in rejected.stderr
