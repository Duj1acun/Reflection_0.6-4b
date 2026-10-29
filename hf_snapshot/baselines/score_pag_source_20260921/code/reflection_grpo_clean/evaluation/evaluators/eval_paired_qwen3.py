"""Four-model pass@4 evaluation with independently scheduled model variants."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.datasets import load_dataset
from evaluation.evaluators.paired_scoring import REFLECTION, SCORER_VERSION, score_sample, summarize

DATASETS = ['math500', 'amc23', 'aime24', 'aime25', 'gpqa']
TOKEN_BUDGETS = {'math500': 4096, 'amc23': 4096, 'aime24': 8192, 'aime25': 8192, 'gpqa': 4096}
BASE_PROMPT = 'Please reason step by step, and put your final answer within \\boxed{}.'
SFT_PROMPT = ('You are a math reasoning assistant. When you make an error during reasoning, '
              'use <reflection>...</reflection> to reflect on the mistake, '
              'then continue reasoning based on the reflection. '
              'Put your final answer within \\boxed{}.')


def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def saved_samples(path, rows, n=4):
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.open() if line.strip()]
    if len(records) % n or len(records) > len(rows) * n:
        raise ValueError(f'Incomplete saved question group: {path}; preserve and inspect before resume')
    for qi in range(len(records) // n):
        group = records[qi*n:(qi+1)*n]
        row = rows[qi]
        if {r['sample_index'] for r in group} != set(range(n)):
            raise ValueError('Duplicate or missing saved sample indices')
        if any(r['uid'] != row['uid'] or r['gold'] != row['gold'] or r['problem'] != row['problem'] for r in group):
            raise ValueError('Saved samples do not match the current ordered dataset')
    return records


def get_rows(name):
    if name in {'aime24', 'aime25'}:
        import pyarrow.parquet as pq
        year = 2000 + int(name[-2:])
        path = f'/root/Policy-As-GenVerifier/datasets/aime{year}.parquet'
        return [{'uid': f'{name}-{i}',
                 'problem': next(m['content'] for m in r['prompt'] if m['role'] == 'user'),
                 'gold': str(r['reward_model']['ground_truth']), 'kind': 'math'}
                for i, r in enumerate(pq.read_table(path).to_pylist())]
    return load_dataset(name)


def header_ids(tok, problem, prompt):
    text = tok.apply_chat_template([{'role': 'system', 'content': prompt},
                                   {'role': 'user', 'content': problem}],
                                  tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    if not text.endswith('<think>\n\n</think>\n\n'):
        raise ValueError('Qwen3 non-thinking header missing')
    return tok.encode(text, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', required=True, choices=['0.6b', '4b'])
    ap.add_argument('--variant', choices=['base', 'phase2', 'both'], default='both')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['preflight', 'smoke', 'run'], default='preflight')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--max-model-len', type=int, default=20480)
    ap.add_argument('--max-iters', type=int, default=2)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    ap.add_argument('--max-num-seqs', type=int, default=128)
    ap.add_argument('--max-num-batched-tokens', type=int, default=16384)
    ap.add_argument('--question-batch-size', type=int, default=64)
    ap.add_argument('--enforce-eager', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = f'qwen3-{args.size}'
    worker_id = f'{model}-{args.variant}'
    base = Path('/mnt/data/models') / ('Qwen3-0.6B' if args.size == '0.6b' else 'Qwen3-4B')
    phase2 = Path('/mnt/data/reflection_grpo/artifacts/paired_sft_20260912') / model / 'phase2'
    selection = json.loads((phase2 / 'checkpoint_selection.json').read_text())
    adapter = Path(selection['best_checkpoint'])
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if Path(config['base_model_name_or_path']).resolve() != base.resolve():
        raise ValueError('Adapter base mismatch')
    if sha(adapter / 'adapter_model.safetensors') != sha(phase2 / 'final/adapter_model.safetensors'):
        raise ValueError('Final adapter differs from selected best checkpoint')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    think_ids = {tok.convert_tokens_to_ids(t) for t in ['<think>', '</think>']}
    def check_completion(comp):
        if think_ids.intersection(comp.token_ids):
            raise ValueError('Forbidden thinking token appeared in generated token IDs')
    datasets = {name: get_rows(name) for name in DATASETS}
    audit = {}
    for name, rows in datasets.items():
        if not rows or len({r['uid'] for r in rows}) != len(rows):
            raise ValueError(f'Invalid dataset identities: {name}')
        lengths = [len(header_ids(tok, r['problem'], p)) for r in rows for p in [BASE_PROMPT, SFT_PROMPT]]
        if max(lengths) + 2 * TOKEN_BUDGETS[name] + 64 > args.max_model_len:
            raise ValueError(f'{name}: context must fit question + maximum reflection + continuation')
        if max(lengths) + TOKEN_BUDGETS[name] > args.max_model_len:
            raise ValueError(f'{name} input exceeds context; no silent truncation')
        serialized = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows)
        audit[name] = {'n': len(rows), 'max_prompt_tokens': max(lengths),
                       'sha256': hashlib.sha256(serialized.encode()).hexdigest()}
        (out / f'{worker_id}__{name}.dataset.jsonl').write_text(serialized)
    settings = {**vars(args), 'base_model': str(base), 'adapter': str(adapter),
                'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
                'base_prompt': BASE_PROMPT, 'sft_prompt': SFT_PROMPT,
                'enable_thinking': False, 'n': 4, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
                'max_new_tokens_per_round': TOKEN_BUDGETS,
                'length_policy': 'fixed dataset output budget each round; never truncate input; raise if context insufficient',
                'datasets': audit, 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                    [Path(__file__), ROOT / 'evaluation/evaluators/paired_scoring.py', ROOT / 'evaluation/datasets.py']}}
    config_path = out / f'{worker_id}.{args.mode}.config.json'
    if args.mode == 'run' and config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ['base_model','adapter','adapter_sha256','base_prompt','sft_prompt',
                    'enable_thinking','n','temperature','top_p','top_k','seed','max_iters',
                    'max_new_tokens_per_round','max_model_len','datasets']:
            if previous[key] != settings[key]:
                raise ValueError(f'Resume would change evaluation semantics: {key}')
    dump(config_path, settings)
    print(json.dumps(settings, ensure_ascii=False), flush=True)
    if args.mode == 'preflight':
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=str(base), dtype='bfloat16', tensor_parallel_size=1,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              enable_lora=True, max_loras=1, max_lora_rank=config['r'],
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
              enable_chunked_prefill=True, enable_prefix_caching=True,
              seed=args.seed, enforce_eager=args.enforce_eager)
    lora = LoRARequest(model + '-phase2', 1, str(adapter))
    # Compare the same prompt under base/LoRA: prove adapter is applied, not just found.
    probe = [{'prompt_token_ids': header_ids(tok, datasets['math500'][i]['problem'], SFT_PROMPT)} for i in range(3)]
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=1)
    base_probe = llm.generate(probe, sp, use_tqdm=False)
    sft_probe = llm.generate(probe, sp, lora_request=lora, use_tqdm=False)
    def signature(outputs):
        return [(list(o.outputs[0].token_ids),
                 [{str(k): v.logprob for k, v in d.items()} for d in o.outputs[0].logprobs]) for o in outputs]
    b_sig, s_sig = signature(base_probe), signature(sft_probe)
    if b_sig == s_sig:
        raise RuntimeError('LoRA smoke failed: logits unchanged on all probes')
    dump(out / f'{worker_id}.lora_smoke.json', {'base': b_sig, 'sft': s_sig, 'adapter_changes_logits': True})
    if args.mode == 'smoke':
        smoke_records = []
        for request, kind in [(None, 'base'), (lora, 'phase2')]:
            results = llm.generate(probe[:1], SamplingParams(n=4, temperature=0.7, top_p=0.8, top_k=20,
                seed=args.seed, max_tokens=128, bad_words=['<think>', '</think>']),
                lora_request=request, use_tqdm=False)
            assert len(results[0].outputs) == 4
            for comp in results[0].outputs:
                check_completion(comp)
                assert '<think>' not in comp.text and '</think>' not in comp.text
                smoke_records.append({'variant':kind,'text':comp.text,'finish_reason':comp.finish_reason,
                    **score_sample((comp.text, datasets['math500'][0]['gold'], 'math500'))})
        dump(out / f'{worker_id}.generation_smoke.json', smoke_records)
        return
    # Template disables thinking; additionally forbid reopening its special tags.
    bad_words = ['<think>', '</think>']
    ctx = mp.get_context('spawn')
    pending_scoring = []
    with ctx.Pool(4) as pool:
        for variant, prompt, request in [('base', BASE_PROMPT, None), ('phase2', SFT_PROMPT, lora)]:
            if args.variant not in {'both', variant}:
                continue
            tag = f'{model}-{variant}'
            for name, rows in datasets.items():
                dest = out / f'{tag}__{name}.samples.jsonl'
                samples = saved_samples(dest, rows)
                resumed_samples = len(samples)
                existing_summary = out / f'{tag}__{name}.summary.json'
                if resumed_samples == len(rows)*4 and existing_summary.exists():
                    checked = summarize(samples, len(rows), 4)
                    previous_summary = json.loads(existing_summary.read_text())
                    if checked['pass_at_n'] != previous_summary['pass_at_n']:
                        raise ValueError('Saved scores differ from existing summary; reconcile before resume')
                    print(f'{tag} {name}: already complete, preserving summary and timing', flush=True)
                    continue
                start = time.time()
                print(f'{tag} {name}: resuming after {resumed_samples//4} saved questions', flush=True)
                with dest.open('a') as f:
                    for offset in range(resumed_samples//4, len(rows), args.question_batch_size):
                        batch_start = time.time()
                        batch = rows[offset:offset + args.question_batch_size]
                        ids = [header_ids(tok, r['problem'], prompt) for r in batch]
                        params = [SamplingParams(n=4, temperature=0.7, top_p=0.8, top_k=20,
                                  max_tokens=TOKEN_BUDGETS[name],
                                  seed=args.seed, bad_words=bad_words) for p in ids]
                        outputs = llm.generate([{'prompt_token_ids': p} for p in ids], params,
                                               lora_request=request, use_tqdm=False)
                        pending = []
                        for row, p, output in zip(batch, ids, outputs):
                            if len(output.outputs) != 4:
                                raise ValueError('vLLM returned fewer than four samples')
                            for comp in output.outputs:
                                check_completion(comp)
                                pending.append({'uid': row['uid'], 'problem': row['problem'], 'gold': row['gold'],
                                    'sample_index': comp.index, 'texts': [comp.text],
                                    'rounds': [{'text': comp.text, 'prompt_tokens': len(p),
                                               'decoded_tokens': len(comp.token_ids), 'finish_reason': comp.finish_reason,
                                               'stop_reason': comp.stop_reason}], 'header_ids': p})
                        # Optional clean-slate continuation, explicitly recorded per round.
                        for _ in range(args.max_iters-1):
                            active = [s for s in pending if REFLECTION.search(s['texts'][-1])]
                            if not active:
                                break
                            cont_ids = [s['header_ids'] + tok.encode(REFLECTION.search(s['texts'][-1]).group(0).rstrip()+'\n',
                                                                  add_special_tokens=False) for s in active]
                            if any(len(p) + TOKEN_BUDGETS[name] > args.max_model_len for p in cont_ids):
                                raise ValueError('Continuation prompt exceeds context; no silent truncation')
                            # Distinct reproducible RNG streams prevent identical reflection
                            # prefixes from collapsing all four trajectories into one sample.
                            cparams = [SamplingParams(n=1, temperature=0.7, top_p=0.8, top_k=20,
                                max_tokens=TOKEN_BUDGETS[name], seed=int.from_bytes(hashlib.sha256(
                                    f'{args.seed}:{s["uid"]}:{s["sample_index"]}:{len(s["rounds"])}'.encode()
                                ).digest()[:4], 'big'), bad_words=bad_words) for s in active]
                            cont = llm.generate([{'prompt_token_ids': p} for p in cont_ids], cparams,
                                                lora_request=request, use_tqdm=False)
                            for sample, p, co in zip(active,cont_ids,cont):
                                comp=co.outputs[0]
                                check_completion(comp)
                                sample['texts'].append(comp.text)
                                sample['rounds'].append({'text':comp.text,'prompt_tokens':len(p),
                                    'decoded_tokens':len(comp.token_ids),'finish_reason':comp.finish_reason,'stop_reason':comp.stop_reason})
                        # Judge the terminal answer only: never fall back to an abandoned earlier round.
                        round_scores = pool.map(score_sample, [(r['text'], s['gold'], name)
                                                              for s in pending for r in s['rounds']])
                        scores_iter = iter(round_scores)
                        for sample in pending:
                            for round_rec in sample['rounds']:
                                round_rec.update(next(scores_iter))
                            score = {k:sample['rounds'][-1][k] for k in
                                     ['extracted','extraction_method','correct','score_status']}
                            sample.pop('header_ids')
                            sample.update(score)
                            text='\n'.join(sample.pop('texts'))
                            sample.update(full_text=text, n_iters=len(sample['rounds']),
                                has_reflection_open=bool(re.search(r'<reflection\b', text, re.I)),
                                complete_reflections=len(REFLECTION.findall(text)),
                                decoded_tokens=sum(r['decoded_tokens'] for r in sample['rounds']),
                                finish_reason=sample['rounds'][-1]['finish_reason'])
                            if '<think>' in text or '</think>' in text:
                                raise ValueError('Thinking tag escaped generation prohibition')
                            f.write(json.dumps(sample, ensure_ascii=False)+'\n')
                            samples.append(sample)
                        f.flush()
                        dump(out / f'{worker_id}.progress.json', {'model':tag, 'dataset':name,
                            'questions_done':offset+len(batch),'questions_total':len(rows),'time':time.time(),
                            'last_batch_s':time.time()-batch_start,
                            'last_batch_decoded_tokens':sum(s['decoded_tokens'] for s in pending),
                            'last_batch_tokens_per_s_including_scoring':sum(s['decoded_tokens'] for s in pending)/(time.time()-batch_start)})
                        print(f'{tag} {name} {offset+len(batch)}/{len(rows)} elapsed={time.time()-start:.0f}s',flush=True)
                # A judge failure must withhold the score, not idle both GPUs
                # and prevent all later benchmarks from being generated.
                unresolved = [{'uid':s['uid'], 'sample_index':s['sample_index'],
                               'status':s['score_status']} for s in samples
                              if s['score_status'] in {'timeout','exception'}]
                pending_path = out / f'{tag}__{name}.scoring_pending.json'
                if unresolved:
                    dump(pending_path, {'model':tag,'dataset':name,'errors':unresolved})
                    pending_scoring.append(name)
                    print(f'{tag} {name}: generation saved; {len(unresolved)} judge errors pending, continuing next dataset', flush=True)
                    continue
                if pending_path.exists():
                    pending_path.unlink()
                metrics = summarize(samples, len(rows), 4)
                first_by_q = {}
                for sample in samples:
                    first_by_q.setdefault(sample['uid'], []).append(sample['rounds'][0]['correct'])
                metrics['initial_pass_at_n'] = sum(any(v) for v in first_by_q.values()) / len(rows)
                metrics['wrong_to_correct'] = sum(not s['rounds'][0]['correct'] and s['correct'] for s in samples)
                metrics['correct_to_wrong'] = sum(s['rounds'][0]['correct'] and not s['correct'] for s in samples)
                metrics['intermediate_judge_errors'] = sum(r['score_status'] in {'timeout','exception'} for s in samples for r in s['rounds'][:-1])
                metrics.update(model=tag, dataset=name, seed=args.seed, elapsed_s=time.time()-start,
                               scorer_version=SCORER_VERSION,
                               elapsed_scope='current process only', samples_resumed=resumed_samples,
                               config_file=f'{worker_id}.run.config.json')
                dump(out / f'{tag}__{name}.summary.json',metrics)
                print(json.dumps(metrics),flush=True)
    dump(out / f'{worker_id}.completed.json', {'completed':not pending_scoring,
         'generation_completed':True,'pending_scoring':pending_scoring,'time':time.time()})


if __name__ == '__main__':
    main()
