"""Opt-in GPU regression using the actual initialized native actor and vLLM pool."""
import os


def install():
    assert os.environ.get("FOURWAY_PROBE") == "1"
    import torch
    from verl import DataProto
    from verl.utils.seqlen_balancing import rearrange_micro_batches
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    cls = DataParallelPPOActor
    if getattr(cls, "_token_budget_probe", False):
        return
    cls._token_budget_probe = True
    original = cls.compute_log_prob

    def checked(self, data, calculate_entropy=False):
        if not getattr(self, "_long_sequence_verified", False):
            self._long_sequence_verified = True
            assert self.config.ppo_max_token_len_per_gpu == 16384
            assert data.meta_info["max_token_len"] == 16384
            assert getattr(torch.optim.AdamW, "_fourway_no_update", False)
            device = data.batch["input_ids"].device
            for length in (8710, 16384):
                ids = (torch.arange(length, device=device) % 1000 + 100).long()[None, :]
                sample = DataProto.from_dict(tensors={
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "position_ids": torch.arange(length, device=device)[None, :],
                    "responses": ids[:, 1:].clone(),
                }, meta_info=dict(data.meta_info))
                try:
                    rearrange_micro_batches(sample.batch, max_token_len=8192)
                except AssertionError as exc:
                    assert "max_token_len" in str(exc)
                    print("TOKEN_BUDGET_OLD_FAILURE_REPRODUCED", length, str(exc), flush=True)
                else:
                    raise AssertionError("Old 8192 budget unexpectedly accepted long sequence")
                logp, entropy = original(self, sample, calculate_entropy=True)
                assert logp.shape == entropy.shape == (1, length - 1)
                assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
                del logp, entropy
                self.actor_module.train()
                self.actor_optimizer.zero_grad(set_to_none=True)
                _, logp = self._forward_micro_batch(sample.batch, temperature=data.meta_info["temperature"])
                loss = -logp.mean()
                assert torch.isfinite(loss)
                loss.backward()
                norm = self._optimizer_step()  # AdamW.step is the no-update probe implementation.
                assert torch.isfinite(norm) and float(norm) > 0
                self.actor_optimizer.zero_grad(set_to_none=True)
                print("TOKEN_BUDGET_LONG_GPU_PASSED", {"length": length, "grad_norm": float(norm),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}, flush=True)
                del sample, ids, logp, loss
                torch.cuda.empty_cache()
        return original(self, data, calculate_entropy=calculate_entropy)

    cls.compute_log_prob = checked
