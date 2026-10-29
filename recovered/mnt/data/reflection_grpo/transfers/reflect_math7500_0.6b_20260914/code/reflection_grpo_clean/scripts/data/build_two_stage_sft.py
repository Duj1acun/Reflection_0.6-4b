#!/usr/bin/env python3
"""Prepare, annotate, then export backbone-specific two-stage SFT data."""
import argparse
import asyncio
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ('qwen3-0.6b', 'qwen3-4b')
DIAGNOSIS = r'''
You are a reasoning error editor.

Input:
{problem, ground_truth, model_response}

Task:
Find the FIRST incorrect step in model_response.

Output rules:
- If the reasoning is completely correct and concise, output exactly:
perfect

- Otherwise output ONLY valid JSON:

{
  "cut_point": "...",
  "reflection": {
    "verified_state": ["...", "..."],
    "error_step": "...",
    "error_reason": "...",
    "new_direction": "..."
  }
}

Definitions:
cut_point
Must copy verbatim from the first character of model_response, character by character, up to and including the first incorrect step.No omission, paraphrase, summary or skipping is allowed in between; the original text must be kept exactly identical.

verified_state
The correct section should enumerate as many correct intermediate conclusions as possible (e.g., variable definitions, equation forms, correct case distinctions, etc.) so that subsequent reasoning can reuse them directly.
Only include correct intermediate conclusions explicitly established in model_response strictly before the first incorrect step; do not include corrected results, new deductions, or conclusions appearing after that step.

error_step
Copy the exact incorrect step verbatim.

error_reason
Briefly explain why the step is incorrect or redundant.

new_direction
One short imperative instruction describing the correct next step. Do NOT give the full solution.

Constraints:
- No explanations
- No reasoning traces
- No extra text
- Output only JSON or "perfect"
'''
CONTINUATION = r'''You are a concise math reasoning assistant.
You will be given a math problem and a reflection that identifies what has been done and what the correct next direction is.
Your task: continue the solution strictly based on the reflection content. Be maximally concise.
Rules:
1. Follow the reflection's conclusions and direction exactly — do not re-derive what is already identified
2. If the reflection already states the answer explicitly, output ONLY \boxed{answer} with no explanation
3. Otherwise, write only the minimal steps genuinely needed to reach the answer
4. End with \boxed{answer}'''
DIAGNOSIS_USER_TEMPLATE = 'Question: {problem}\nGround Truth: {ground_truth}\nWeak CoT: {model_response}'
CONTINUATION_USER_TEMPLATE = '''Problem: {problem}

{reflection_text}

Continue the solution strictly based on the reflection above. Be concise.'''

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def dump(path, obj):
    # Atomic replacement: a crash cannot leave half a checkpoint.
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def prepare(args):
    if args.output.exists():
        raise ValueError('Output already exists; use a new directory for preparation')
    selected, perfect, sources = [], [], {}
    for model in MODELS:
        path = args.data_root / model / 'scored_rollouts_n4.jsonl'
        before = file_hash(path)
        groups = defaultdict(list)
        for row in read(path):
            groups[row['problem_id']].append(row)
        seen = set()
        for pid, rows in sorted(groups.items()):
            if len(rows) != 4 or {r['sample_index'] for r in rows} != {0, 1, 2, 3}:
                continue
            if len({r['trajectory_id'] for r in rows}) != 4:
                raise ValueError('Duplicate trajectory')
            if any(r['score_status'] not in ('correct', 'incorrect') for r in rows):
                continue
            if len({(r['problem'], str(r['answer'])) for r in rows}) != 1:
                raise ValueError('Conflicting problem or answer within group')
            k = sum(r['score_status'] == 'correct' for r in rows)
            # Same problem text receives the same split in both backbones.
            split = 'eval' if int(hashed(' '.join(rows[0]['problem'].split()))[:8], 16) % 10 == 0 else 'train'
            good, bad, texts = [], [], set()
            for r in sorted(rows, key=lambda r: r['sample_index']):
                if r.get('finish_reason') != 'stop' or not r['completion'].strip():
                    continue
                if r['completion'] in texts:
                    continue
                texts.add(r['completion'])
                uid = hashed(model + ':' + str(r['trajectory_id']))
                if uid in seen:
                    raise ValueError('Duplicate ID')
                seen.add(uid)
                clean = dict(uid=uid, index=uid, student_model=model, problem_id=pid,
                             trajectory_id=r['trajectory_id'], problem=r['problem'],
                             ground_truth=r['answer'], model_response=r['completion'],
                             correct_out_of_four=k, split=split,
                             score_status=r['score_status'], finish_reason=r['finish_reason'])
                (good if r['score_status'] == 'correct' else bad).append(clean)
            if not getattr(args, 'errors_only', False):
                perfect.extend(good[:1])
            if k in (1, 2, 3):
                selected.extend(bad[:2])
        if before != file_hash(path):
            raise ValueError('Source changed during preparation')
        sources[model] = dict(path=str(path), sha256=before)
    args.output.mkdir(parents=True)
    jsonl(args.output / 'candidates.jsonl', selected)
    jsonl(args.output / 'perfect.jsonl', perfect)
    manifest = dict(sources=sources, diagnosis=DIAGNOSIS, continuation=CONTINUATION,
                    diagnosis_user_template=DIAGNOSIS_USER_TEMPLATE,
                    continuation_user_template=CONTINUATION_USER_TEMPLATE,
                    candidates_sha256=file_hash(args.output/'candidates.jsonl'),
                    perfect_sha256=file_hash(args.output/'perfect.jsonl'),
                    reflection_counts=dict(Counter(r['student_model'] for r in selected)),
                    perfect_counts=dict(Counter(r['student_model'] for r in perfect)),
                    errors_only=getattr(args, 'errors_only', False))
    manifest['planned_run'] = dict(teacher=args.teacher, thinking=args.thinking,
                                   max_tokens=args.max_tokens, workers=args.workers,
                                   attempts=args.attempts)
    manifest['phase1_prefix'] = args.phase1_prefix
    dump(args.output/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('reflection_counts','perfect_counts')}, indent=2))

def validate_diagnosis(content, original):
    if content in ('perfect', 'review'):
        raise ValueError('teacher_' + content + '_requires_review')
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate_json_key')
            out[k] = v
        return out
    p = json.loads(content, object_pairs_hook=pairs)
    if not isinstance(p, dict) or set(p) != {'cut_point', 'reflection'}:
        raise ValueError('invalid_schema')
    r = p['reflection']
    if not isinstance(r, dict) or set(r) != {'verified_state','error_step','error_reason','new_direction'}:
        raise ValueError('invalid_reflection_schema')
    if not isinstance(r['verified_state'], list) or any(not isinstance(s,str) or not s.strip() for s in r['verified_state']):
        raise ValueError('invalid_states')
    if any(not isinstance(r[k],str) or not r[k].strip() for k in ('error_step','error_reason','new_direction')):
        raise ValueError('invalid_fields')
    cut, error = p['cut_point'], r['error_step']
    if not isinstance(cut,str) or not cut or not original.startswith(cut):
        raise ValueError('cut_not_exact_prefix')
    pos = cut.rfind(error)
    if pos < 0 or not re.fullmatch(r'[\s$]*', cut[pos+len(error):]):
        raise ValueError('error_not_at_cut_end')
    text = ' '.join(r['verified_state'] + [r['error_reason'],r['new_direction']])
    if re.search(r'<\s*/?\s*(?:think|reflection)\b', text, re.I):
        raise ValueError('unexpected_tags')
    return p

def reflection_text(p):
    r = p['reflection']
    states = ' '.join(r['verified_state']) or 'I have not yet established any correct intermediate conclusions.'
    return '<reflection>\n' + states + '\n' + r['error_reason'] + '\n' + r['new_direction'] + '\n</reflection>'

async def run(args, manifest):
    import httpx
    if manifest.get('planned_run'):
        actual = {k:getattr(args,k) for k in manifest['planned_run']}
        if actual != manifest['planned_run']:
            raise ValueError('Parameters differ from the prepared formal run')
    for row in read(args.output/'candidates.jsonl'):
        if manifest.get('planned_run') and (row.get('score_status')!='incorrect' or row.get('finish_reason')!='stop'):
            raise ValueError('Annotation permits only complete incorrect samples')
    minimax = args.teacher == 'MiniMax-M3'
    key_env = 'MINIMAX_API_KEY' if minimax else 'PARATERA_API_KEY'
    key = os.environ.get(key_env)
    if getattr(args, 'prompt_api_key', False):
        import getpass
        key = getpass.getpass(('MiniMax' if minimax else 'Paratera') + ' API key (hidden): ').strip()
    if not key:
        raise ValueError('Set ' + key_env + '; no key is stored in code')
    config = dict(base_url='https://llmapi.paratera.com/v1', model=args.teacher,
                  temperature=0.1, top_p=0.95, max_tokens=args.max_tokens,
                  thinking={'type':'enabled'}, reasoning_effort='high')
    thinking_enabled = getattr(args, 'thinking', 'enabled') == 'enabled'
    if not thinking_enabled:
        config['thinking'] = {'type':'disabled'}
        config.pop('reasoning_effort')
    if minimax:
        config['base_url'] = 'https://api.minimaxi.com/v1'
        config['max_completion_tokens'] = config.pop('max_tokens')
        config['reasoning_split'] = True
        config.pop('reasoning_effort', None)
        config['thinking'] = {'type':'adaptive' if thinking_enabled else 'disabled'}
    cfgpath = args.output/'run_config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text()) != config:
        raise ValueError('Run configuration changed; use a new output directory')
    if not cfgpath.exists():
        dump(cfgpath, config)
    rows = read(args.output/'candidates.jsonl')
    if args.limit:
        rows = [r for model in MODELS for r in [x for x in rows if x['student_model']==model][:args.limit]]
    if getattr(args, 'total_limit', 0):
        buckets = [[r for r in rows if r['student_model']==m] for m in MODELS]
        rows = [bucket[i] for i in range(max(map(len,buckets), default=0)) for bucket in buckets if i<len(bucket)][:args.total_limit]
    calls = args.output/'calls'
    calls.mkdir(exist_ok=True)
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600,connect=30), follow_redirects=False) as client:
        async def stage(row, name, user):
            stem = row['uid'] + '.' + name
            final = calls/(stem+'.json')
            if final.exists():
                return json.loads(final.read_text())
            body = {k:v for k,v in config.items() if k!='base_url'}
            template = manifest.get(name + '_user_template')
            user_content = template.format(**{k: v.strip() if k=='reflection_text' else v for k,v in user.items()}) if template else json.dumps(user,ensure_ascii=False)
            body['messages'] = [{'role':'system','content':manifest[name]}, {'role':'user','content':user_content}]
            request_path = calls/(stem+'.request.json')
            if request_path.exists() and json.loads(request_path.read_text()) != body:
                raise ValueError('Checkpoint request mismatch')
            if not request_path.exists():
                dump(request_path, body)
            for attempt in range(1, args.attempts+1):
                attempt_path = calls/(stem+f'.attempt{attempt}.json')
                if attempt_path.exists():
                    saved = json.loads(attempt_path.read_text())
                    if saved.get('status') == 'inflight':
                        result = dict(status='review', reason='uncertain_previous_request_not_automatically_repeated')
                        dump(final,result)
                        return result
                    if saved.get('retryable'):
                        continue
                    dump(final,saved)
                    return saved
                dump(attempt_path,dict(status='inflight',started=time.time()))
                result = {}
                delay = min(60, 2**attempt + random.random())
                try:
                    resp = await client.post(config['base_url']+'/chat/completions', json=body, headers={'Authorization':'Bearer '+key})
                    result['http_status'] = resp.status_code
                    if resp.status_code != 200:
                        result.update(status='failed',reason='http_error',retryable=resp.status_code in (429,500,502,503,504))
                        try:
                            delay = min(120, max(delay,float(resp.headers.get('retry-after',0))))
                        except ValueError:
                            pass
                    else:
                        raw = resp.json()
                        choice = raw['choices'][0]
                        msg = choice['message']
                        content = msg.get('content') or ''
                        result.update(content=content,usage=raw.get('usage'),response_model=raw.get('model'),finish_reason=choice.get('finish_reason'),thinking_returned=bool(msg.get('reasoning_content') or msg.get('reasoning_details')))
                        details = (raw.get('usage') or {}).get('completion_tokens_details') or {}
                        result['reasoning_tokens'] = details.get('reasoning_tokens')
                        if str(raw.get('model','')).lower()!=args.teacher.lower() or choice.get('finish_reason')!='stop':
                            raise ValueError('unexpected_model_finish_or_missing_thinking')
                        if thinking_enabled and not result['thinking_returned']:
                            raise ValueError('missing_thinking')
                        if not thinking_enabled and (result['thinking_returned'] or (result['reasoning_tokens'] or 0)>0):
                            raise ValueError('thinking_disable_not_honored')
                        if name=='diagnosis':
                            result['annotation'] = validate_diagnosis(content,row['model_response'])
                        elif content.strip()=='review' or not content.strip() or not re.search(r'\\boxed\s*\{',content) or re.search(r'<\s*/?\s*(?:think|reflection)\b',content,re.I):
                            raise ValueError('continuation_requires_review')
                        result['status']='ok'
                except (httpx.TimeoutException,httpx.TransportError) as exc:
                    # Delivery is uncertain: do not automatically buy a duplicate generation.
                    result.update(status='review',reason=type(exc).__name__)
                except (ValueError,KeyError,IndexError,TypeError) as exc:
                    result.update(status='review',reason=type(exc).__name__,detail=str(exc))
                dump(attempt_path,result)
                if result.get('retryable') and attempt < args.attempts:
                    await asyncio.sleep(delay)
                    continue
                dump(final,result)
                print(row['student_model'],row['trajectory_id'],name,result['status'],flush=True)
                return result
            result = dict(status='failed',reason='attempt_budget_exhausted')
            dump(final,result)
            return result
        async def worker():
            while not queue.empty():
                row = queue.get_nowait()
                d = await stage(row,'diagnosis',dict(problem=row['problem'],ground_truth=row['ground_truth'],model_response=row['model_response']))
                if d['status']=='ok':
                    await stage(row,'continuation',dict(problem=row['problem'],reflection_text=reflection_text(d['annotation'])))
                queue.task_done()
        tasks = [asyncio.create_task(worker()) for _ in range(args.workers)]
        while tasks:
            done, pending = await asyncio.wait(tasks,timeout=30,return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()
            tasks = list(pending)
            if tasks:
                print(f'Progress: queued={queue.qsize()}, active_workers={len(tasks)}',flush=True)

def export(args):
    from rgrpo.math_score_light import compute_math_score
    candidates = read(args.output/'candidates.jsonl')
    perfect = read(args.output/'perfect.jsonl')
    report = {}
    for model in MODELS:
        accepted, review = [], []
        for row in (r for r in candidates if r['student_model']==model):
            paths = [args.output/'calls'/(row['uid']+'.'+s+'.json') for s in ('diagnosis','continuation')]
            if not all(p.exists() for p in paths):
                review.append(dict(**row,review_reason='incomplete_stages'))
                continue
            d,c = [json.loads(p.read_text()) for p in paths]
            if d['status']!='ok' or c['status']!='ok':
                review.append(dict(**row,review_reason='stage_not_ok',diagnosis=d,continuation=c))
                continue
            try:
                score = compute_math_score(generated=c['content'],ground_truth=row['ground_truth'])
            except Exception:
                score = None
            if score != 1.0:
                review.append(dict(**row,review_reason='answer_mismatch_or_unverified',diagnosis=d,continuation=c))
                continue
            p = d['annotation']
            accepted.append(dict(**row,cut_point=row['model_response'] if args.phase1_prefix=='full' else p['cut_point'],
                                 diagnostic_cut_point=p['cut_point'],reflection=p['reflection'],
                                 reflection_text=reflection_text(p),reasoning=c['content'],
                                 reference_score=score,semantic_reviewed=False))
        out = args.output/model
        out.mkdir(exist_ok=True)
        jsonl(out/'reasoning_outputs.jsonl',accepted)
        jsonl(out/'review.jsonl',review)
        goods = [dict(**r,full_reasoning=r['model_response']) for r in perfect if r['student_model']==model]
        jsonl(out/'reflection_data_prepared_perfect.jsonl',goods)
        mixed = [dict(r, type='reflection') for r in accepted] + [dict(r, type='perfect') for r in goods]
        random.Random(42).shuffle(mixed)
        jsonl(out/'phase1_mixed.jsonl',mixed)
        report[model]=dict(auto_checked_reflection=len(accepted),review=len(review),perfect=len(goods),mixed=len(mixed),phase1_prefix=args.phase1_prefix,semantic_reviewed=False)
    dump(args.output/'export_summary.json',report)
    print(json.dumps(report,indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','export'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/mnt/data/reflection_grpo/data/sft'))
    p.add_argument('--teacher',default='DeepSeek-V4-Pro',choices=('DeepSeek-V4-Pro','MiniMax-M3'))
    p.add_argument('--workers',type=int,default=64)
    p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--max-tokens',type=int,default=32768)
    p.add_argument('--errors-only',action='store_true',help='Prepare only incorrect candidates; do not export perfect samples')
    p.add_argument('--thinking',choices=('enabled','disabled'),default='enabled')
    p.add_argument('--limit',type=int,default=0,help='Per backbone; 0 means all candidates')
    p.add_argument('--total-limit',type=int,default=0,help='Total candidates, interleaved across backbones')
    p.add_argument('--prompt-api-key',action='store_true',help='Read key from hidden terminal prompt')
    p.add_argument('--phase1-prefix',choices=('full','cut'),default='cut')
    args = p.parse_args()
    if min(args.workers,args.attempts,args.max_tokens)<=0 or args.limit<0 or args.total_limit<0:
        p.error('Invalid limits')
    if args.action=='prepare':
        prepare(args)
        return
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((args.output/'manifest.json').read_text())
        for name in ('candidates','perfect'):
            if file_hash(args.output/(name+'.jsonl'))!=manifest[name+'_sha256']:
                raise ValueError('Prepared input modified')
        if args.action=='run':
            asyncio.run(run(args,manifest))
        else:
            if manifest.get('phase1_prefix', args.phase1_prefix) != args.phase1_prefix:
                raise ValueError('Export prefix differs from prepared data contract')
            export(args)

if __name__=='__main__':
    main()
