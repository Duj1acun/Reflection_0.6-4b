"""Isolated vLLM throughput/long-output pressure check; never trains a model."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke-only", action="store_true", help="Only short decoding plus lifecycle/cleanup checks")
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    import torch
    from vllm import LLM, SamplingParams
    from .contracts import messages, atomic_json, file_sha
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c["base_model"], local_files_only=True)
    records = [json.loads(l) for l in (Path(c["output"])/"data/train.canonical.jsonl").read_text().splitlines()]
    prompts = [tokenizer.apply_chat_template(messages(r["problem"], "grpo"),
               tokenize=True, add_generation_prompt=True, enable_thinking=False) for r in records[:32]]
    # Identical prompt/seed list across candidates, 8 independent samples/question.
    prompts = [{"prompt_token_ids": x} for x in prompts[:8] for _ in range(8)]
    if args.smoke_only:
        prompts = prompts[:4]
    engine = LLM(model=c["base_model"], dtype="bfloat16", tensor_parallel_size=1,
                 max_model_len=c["max_model_len"], gpu_memory_utilization=c["gpu_memory_utilization"],
                 enforce_eager=c.get("enforce_eager", True), enable_sleep_mode=True,
                 max_num_seqs=c["max_num_seqs"], max_num_batched_tokens=c["max_num_batched_tokens"],
                 enable_prefix_caching=True, enable_chunked_prefill=True, seed=c["seed"])
    warm = SamplingParams(max_tokens=16, temperature=0, bad_words=["<think>", "</think>"])
    engine.generate(prompts[:4], warm, use_tqdm=False)
    # Native trainer sleeps/wakes vLLM between training and rollout. Ensure graph
    # execution works after that lifecycle transition as well as initial loading.
    before = [x.outputs[0].token_ids for x in engine.generate(prompts[:2], warm, use_tqdm=False)]
    engine.sleep(level=1)
    engine.wake_up()
    after = [x.outputs[0].token_ids for x in engine.generate(prompts[:2], warm, use_tqdm=False)]
    if before != after:
        raise RuntimeError("Greedy outputs changed after sleep/wake without a weight update")
    stats = []
    for name, force in ([("smoke", False)] if args.smoke_only else [("natural", False), ("forced_4096", True)]):
        engine.reset_prefix_cache()
        params = [SamplingParams(max_tokens=16 if args.smoke_only else 4096, ignore_eos=force,
                  temperature=1.0, top_p=1.0, seed=c["seed"]+i,
                  bad_words=["<think>", "</think>"], detokenize=False) for i in range(len(prompts))]
        pre = engine.llm_engine.scheduler[0].num_cumulative_preemption
        start = time.monotonic()
        outputs = engine.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        tokens = sum(len(x.outputs[0].token_ids) for x in outputs)
        stats.append(dict(name=name, samples=len(outputs), generated_tokens=tokens,
                          seconds=elapsed, tokens_per_second=tokens/elapsed,
                          preemptions=engine.llm_engine.scheduler[0].num_cumulative_preemption-pre,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        print(json.dumps(stats[-1]), flush=True)
    atomic_json(args.output, dict(config_sha256=file_sha(args.config), config=c,
                sleep_wake_greedy_match=True, stats=stats,
                scope=("Single-GPU standalone vLLM, 4 short requests; lifecycle/cleanup smoke check."
                       if args.smoke_only else
                       "Single-GPU standalone vLLM, 64 requests; forced output tests cache pressure, not quality. Native two-GPU FSDP probes are separate.")))
    # Destroy distributed groups explicitly before interpreter teardown. Leaving
    # NCCL/CUDA graphs to global destructors can crash the standalone V0 process.
    engine.sleep(level=1)
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()
    del engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
