"""Opt-in integration with either repository's native trainer and workers."""
import json
import os
from pathlib import Path

from .contracts import atomic_json, config, rank_key, summarize, token_runs, generated_budget

def worker_setup():
    import random
    import numpy as np
    import torch
    seed = config()["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if os.environ.get("FOURWAY_PROBE") == "1" and not getattr(torch.optim.AdamW, "_fourway_no_update", False):
        # Exercise real native forward/backward/gradient synchronization, but
        # never apply a parameter update.
        def no_update(optimizer, closure=None):
            if closure is not None:
                raise RuntimeError("Probe does not support optimizer closures")
            grads = [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError("Probe gradients missing or nonfinite")
            # Allocate the real optimizer-state footprint without changing any
            # parameter, so a passing probe doesn't hide Adam's first-step OOM.
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None and not optimizer.state[p]:
                        optimizer.state[p]["exp_avg"] = torch.zeros_like(p)
                        optimizer.state[p]["exp_avg_sq"] = torch.zeros_like(p)
            scratch = [torch.empty_like(g) for g in grads]
            print("FOURWAY_NO_UPDATE_PROBE", {"gradient_tensors": len(grads),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}, flush=True)
            del scratch
            return None
        torch.optim.AdamW.step = no_update
        torch.optim.AdamW._fourway_no_update = True

    if os.environ.get("FOURWAY_LONG_SEQUENCE_PROBE") == "1":
        from .token_budget_probe import install as install_long_sequence_probe
        install_long_sequence_probe()

def extract_trajectory(batch, index, tok, method):
    ids = batch.batch["responses"][index].tolist()
    prompt = batch.batch["prompts"][index].tolist()
    prompt = [v for v in prompt if v != tok.pad_token_id]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    if method == "reflect" and "fourway_segments" in batch.non_tensor_batch:
        segments = list(batch.non_tensor_batch["fourway_segments"][index])
        if hasattr(segments, "tolist"):
            segments = segments.tolist()
        total = int(batch.non_tensor_batch["fourway_generated_tokens"][index])
        text = tok.decode([v for v in ids if v != tok.pad_token_id], skip_special_tokens=True)
        from evaluation.evaluators.paired_scoring import answer_text
        final = answer_text(text)
    elif method in {"pag", "score"}:
        mask = batch.batch["multiturn_mask"][index].bool().tolist()
        segments = []
        for j, (start, end) in enumerate(token_runs(mask)):
            output = ids[start:end]
            segments.append({"role": "verify" if method == "pag" and j % 2 else "answer",
                             "prompt_token_ids": prompt + ids[:start], "output_token_ids": output,
                             "text": tok.decode(output, skip_special_tokens=True)})
        answers = [s for s in segments if s["role"] == "answer"]
        if not answers:
            raise RuntimeError("Missing answer segment")
        final = answers[-1]["text"]
        total = sum(len(s["output_token_ids"]) for s in segments)
    else:
        output = [v for v in ids if v != tok.pad_token_id]
        segments = [{"role": "answer", "prompt_token_ids": prompt, "output_token_ids": output,
                     "text": tok.decode(output, skip_special_tokens=True)}]
        final, total = segments[0]["text"], len(output)
    forbidden = {tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")}
    if any(forbidden.intersection(s["output_token_ids"]) for s in segments):
        raise RuntimeError("Generated thinking token")
    c = config()
    if total > generated_budget(c, method):
        raise RuntimeError(f"Trajectory exceeded total generation budget: {total}")
    truncated = any(s["output_token_ids"] and s["output_token_ids"][-1] not in eos for s in segments)
    return segments, final, total, truncated

def validate(self):
    """An unresolved validation is missing data, never a zero-accuracy score."""
    from .resilience import is_judge_error
    from .scoring import JudgeError
    c = config()
    try:
        result = _validate_complete(self)
    except Exception as exc:
        if (not is_judge_error(exc) or c.get("judge_failure_policy", "raise") != "skip_batch"
                or self.config.trainer.get("val_only", False)):
            raise
        count = getattr(self, "_fourway_consecutive_validation_failures", 0) + 1
        self._fourway_consecutive_validation_failures = count
        self._fourway_failed_validation_step = int(self.global_steps)
        import traceback
        destination = Path(self.config.trainer.default_local_dir) / "validation" / f"step_{self.global_steps:04d}.failed.json"
        atomic_json(destination, dict(step=int(self.global_steps), status="judge_failed",
                                     consecutive_failures=count, error=str(exc), traceback=traceback.format_exc()))
        print(f"FOURWAY_VALIDATION_SKIPPED step={self.global_steps} diagnostic={destination}", flush=True)
        if count >= int(c.get("judge_max_consecutive_failures", 3)):
            raise JudgeError(f"Stopping after {count} consecutive failed validations; diagnostic={destination}") from exc
        return {"val/judge_failed": 1, "val/consecutive_judge_failures": count}
    self._fourway_consecutive_validation_failures = 0
    self._fourway_failed_validation_step = None
    return result


def _validate_complete(self):
    import time
    import numpy as np
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from .scoring import judge
    c, method = config(), self.config.fourway.method
    records = []
    started = time.time()
    canonical = {r["uid"]: r for r in (json.loads(x) for x in (Path(c["output"])/"data/validation.canonical.jsonl").read_text().splitlines())}
    for batch_index, test_data in enumerate(self.val_dataloader):
        batch = DataProto.from_single_dict(test_data).repeat(repeat_times=4, interleave=True)
        originals = list(batch.non_tensor_batch["extra_info"])
        gen = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[k for k in ("raw_prompt_ids", "raw_prompt", "data_source") if k in batch.non_tensor_batch])
        gen.meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id,
                         "do_sample": True, "validate": True, "num_turns": 2,
                         "fourway_seed": c["validation"]["seed"] + batch_index * 1009}
        if method in {"grpo", "reflect"}:
            result, _ = self.traj_collector.multi_turn_loop(gen_batch=gen, actor_rollout_wg=self.actor_rollout_wg,
                                                           envs=self.val_envs, is_train=False)
        else:
            padded, padding = pad_dataproto_to_divisor(gen, self.actor_rollout_wg.world_size)
            result = unpad_dataproto(self.actor_rollout_wg.generate_sequences(padded), pad_size=padding)
        if len(result) != len(originals):
            raise RuntimeError("Validation trajectory count changed during rollout")
        for i, info in enumerate(originals):
            source = canonical[str(info["index"])]
            segments, final, total, truncated = extract_trajectory(result, i, self.tokenizer, method)
            scored = judge(final, source["gold"], source["dataset"])
            records.append(dict(uid=source["uid"], dataset=source["dataset"], problem=source["problem"],
                                gold=source["gold"], sample_index=i % 4, segments=segments, final_answer_text=final,
                                generated_tokens=total, truncated=truncated,
                                has_reflection_open=any("<reflection>" in s["text"] for s in segments), **scored))
        expected_samples = 8 if self.config.fourway.probe else 2160
        print(f"FOURWAY_VALIDATION {method} step={self.global_steps} samples={len(records)}/{expected_samples}", flush=True)
    counts = {"math500": 1, "amc23": 1} if self.config.fourway.probe else None
    result = dict(summarize(records, counts), step=int(self.global_steps), method=method, stage=int(self.config.fourway.stage),
                  elapsed_seconds=time.time()-started, validation_is_used_for_selection=True)
    out = Path(self.config.trainer.default_local_dir) / "validation"
    out.mkdir(parents=True, exist_ok=True)
    samples = out/f"step_{self.global_steps:04d}.samples.jsonl"
    tmp = samples.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records))
    tmp.replace(samples)
    atomic_json(out/f"step_{self.global_steps:04d}.summary.json", result)
    return {"val/macro_pass_at_4": result["macro_pass_at_4"], "val/macro_mean_at_4": result["macro_mean_at_4"],
            **{f"val/{ds}/{k}": v[k] for ds,v in result["datasets"].items() for k in ("pass_at_4", "mean_at_4")}}

def install(cfg):
    """Called inside the Ray TaskRunner before the trainer is constructed."""
    import torch
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    worker_setup()
    if getattr(RayPPOTrainer, "_fourway_installed", False):
        return
    RayPPOTrainer._fourway_installed = True
    RayPPOTrainer._validate = validate
    old_save, old_load = RayPPOTrainer._save_checkpoint, RayPPOTrainer._load_checkpoint
    def save(self):
        old_save(self)
        run = Path(self.config.trainer.default_local_dir)
        step = int(self.global_steps)
        atomic_json(run / f"global_step_{step}" / "judge_state.json", {
            key: getattr(self, key, 0) for key in (
                "_fourway_skipped_batches", "_fourway_consecutive_judge_failures",
                "_fourway_consecutive_validation_failures")})
        if getattr(self, "_fourway_failed_validation_step", None) == step:
            # Keep the recoverable training checkpoint without ranking incomplete
            # validation or deleting the last successfully evaluated checkpoint.
            atomic_json(run / "latest.json", dict(step=step, validation_status="judge_failed",
                        checkpoint=str(run / f"global_step_{step}")))
            return
        summary = run / "validation" / f"step_{step:04d}.summary.json"
        if not summary.exists():
            if step % config()["validation"]["frequency"] == 0:
                raise RuntimeError("Checkpoint has no complete validation; refusing best selection")
            # Frequent recoverable saves do not participate in best selection.
            atomic_json(run / "latest.json", dict(step=step, validation_status="not_scheduled",
                        checkpoint=str(run / f"global_step_{step}")))
            best_path = run / "best.json"
            keep = {step}
            if best_path.exists():
                keep.add(int(json.loads(best_path.read_text())["step"]))
            if (run / "fourway_run.json").exists():
                import shutil
                for p in run.glob("global_step_*"):
                    if p.is_dir() and p.name.removeprefix("global_step_").isdigit() and int(p.name.split("_")[-1]) not in keep:
                        shutil.rmtree(p)
            return
        current = json.loads(summary.read_text())
        best_path = run/"best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if step > 0 and (best is None or rank_key(current) > rank_key(best)):
            best = dict(current, checkpoint=str(run/f"global_step_{step}"))
            atomic_json(best_path, best)
        atomic_json(run/"latest.json", dict(current, checkpoint=str(run/f"global_step_{step}")))
        # Only remove owned, superseded snapshots after both model save and judge succeed.
        if (run/"fourway_run.json").exists() and best:
            import shutil
            keep = {step, best["step"]}
            if self.config.fourway.method == "score":
                keep.add(config()["score_stage1_steps"])
            for p in run.glob("global_step_*"):
                if p.is_dir() and p.name.removeprefix("global_step_").isdigit() and int(p.name.split("_")[-1]) not in keep:
                    shutil.rmtree(p)
    def load(self):
        if self.config.trainer.resume_mode == "resume_path":
            p = Path(self.config.trainer.resume_from_path)
            required = [p/"actor", p/"data.pt"]
            if self.use_critic:
                required.append(p/"critic")
            if any(not x.exists() for x in required):
                raise RuntimeError(f"Incomplete resume checkpoint: {p}")
        result = old_load(self)
        # Native loaders restore global_steps and the stateful dataloader first.
        state_path = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "judge_state.json"
        if self.config.trainer.resume_mode == "resume_path":
            state_path = Path(self.config.trainer.resume_from_path) / "judge_state.json"
        if self.config.trainer.resume_mode != "disable" and state_path.exists():
            state = json.loads(state_path.read_text())
            for key in ("_fourway_skipped_batches", "_fourway_consecutive_judge_failures",
                        "_fourway_consecutive_validation_failures"):
                setattr(self, key, int(state.get(key, 0)))
        return result
    RayPPOTrainer._save_checkpoint = save
    RayPPOTrainer._load_checkpoint = load
    import verl.trainer.ppo.ray_trainer as trainer_module
    old_metrics = trainer_module.compute_data_metrics
    def metrics(batch, *args, **kwargs):
        result = old_metrics(batch, *args, **kwargs)
        if "branch_weight" in batch.batch:
            total = int(batch.meta_info["branch_generated_tokens"])
            roots = float(batch.batch["branch_weight"].sum())
            limit = generated_budget(config(), cfg.fourway.method)
            paths = [int(info.get("path_generated_tokens", 0)) for info in batch.non_tensor_batch["extra_info"]]
            if max(paths) > limit:
                raise RuntimeError("Branch path exceeded generation budget")
            result["fourway/generated_tokens"] = total
            result["fourway/mean_generated_tokens"] = total / roots
            result["branch/max_path_generated_tokens"] = max(paths)
            return result
        if "fourway_generated_tokens" in batch.non_tensor_batch:
            costs = torch.as_tensor(batch.non_tensor_batch["fourway_generated_tokens"].astype("int64"))
        elif "multiturn_mask" in batch.batch:
            costs = batch.batch["multiturn_mask"].sum(-1)
        else:
            costs = (batch.batch["responses"] != batch.meta_info.get("pad_token_id", 151643)).sum(-1)
        result["fourway/generated_tokens"] = int(costs.sum().item())
        result["fourway/mean_generated_tokens"] = float(costs.float().mean().item())
        if int(costs.max()) > generated_budget(config(), cfg.fourway.method):
            raise RuntimeError("Training exceeded generation budget")
        return result
    trainer_module.compute_data_metrics = metrics
