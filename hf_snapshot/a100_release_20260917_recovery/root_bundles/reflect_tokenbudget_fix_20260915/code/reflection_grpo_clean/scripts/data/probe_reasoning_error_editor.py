import asyncio
import getpass
import json
import time
from pathlib import Path
import httpx

PROMPT = '''You are a reasoning error editor.

Input:
{problem, ground_truth, model_response}

Task:
Find the FIRST incorrect or redundant step in model_response.

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
Must copy verbatim from the first character of model_response, character by character, up to and including the first error or redundant step.No omission, paraphrase, summary or skipping is allowed in between; the original text must be kept exactly identical.

verified_state
The correct section should enumerate as many correct intermediate conclusions as possible (e.g., variable definitions, equation forms, correct case distinctions, etc.) so that subsequent reasoning can reuse them directly.

error_step
Copy the exact incorrect or redundant step verbatim.

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

OUT = Path('/root/reflection_grpo_clean/outputs/teacher_error_editor_pro_probe3_20260912')
SOURCE = Path('/mnt/data/reflection_grpo/data/sft/teacher_sft_firstperson_nogold_probe10_deepseek_v4_pro_20260910/samples.json')

def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

async def main():
    rows = json.loads(SOURCE.read_text())
    ids = {'qwen3-0.6b_k2_03', 'qwen3-4b_k2_01', 'qwen3-4b_k1_02'}
    rows = [r for r in rows if r['probe_id'] in ids]
    assert len(rows) == 3
    key = getpass.getpass('Paratera API key (hidden): ').strip()
    OUT.mkdir(parents=True, exist_ok=False)
    save('manifest.json', dict(prompt=PROMPT, source=str(SOURCE), samples=rows, retries=0, concurrency=3))
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30), follow_redirects=False) as client:
        async def one(row):
            pid = row['probe_id']
            body = dict(model='DeepSeek-V4-Pro', temperature=0.1, top_p=0.95,
                        max_tokens=16384, thinking={'type':'enabled'}, reasoning_effort='high',
                        messages=[{'role':'user', 'content':PROMPT + '\nInput:\n' + json.dumps(dict(problem=row['problem'], ground_truth=row['answer'], model_response=row['completion']), ensure_ascii=False)}])
            save(pid + '.request.json', body)
            start = time.monotonic()
            result = dict(probe_id=pid)
            print('REQUEST ' + pid, flush=True)
            try:
                response = await client.post('https://llmapi.paratera.com/v1/chat/completions', json=body, headers={'Authorization':'Bearer '+key})
                result['http_status'] = response.status_code
                raw = json.loads(response.text.replace(key, '[REDACTED]'))
                save(pid + '.response.json', raw)
                response.raise_for_status()
                choice = raw['choices'][0]
                msg = choice['message']
                content = msg.get('content') or ''
                result.update(model=raw.get('model'), finish_reason=choice.get('finish_reason'), thinking_returned=bool(msg.get('reasoning_content') or msg.get('reasoning_details')), usage=raw.get('usage'), content=content)
                if content == 'perfect':
                    result['perfect'] = True
                else:
                    p = json.loads(content)
                    ref = p['reflection']
                    result.update(valid_structure=set(p)=={'cut_point','reflection'} and set(ref)=={'verified_state','error_step','error_reason','new_direction'} and isinstance(ref['verified_state'],list) and all(isinstance(s,str) for s in ref['verified_state']) and all(isinstance(ref[k],str) for k in ('error_step','error_reason','new_direction')) and isinstance(p['cut_point'],str),
                                  exact_prefix=bool(p['cut_point']) and row['completion'].startswith(p['cut_point']),
                                  exact_error_step=bool(ref['error_step']) and ref['error_step'] in row['completion'],
                                  cut_ends_with_error=p['cut_point'].endswith(ref['error_step']))
            except Exception as exc:
                result['error_type'] = type(exc).__name__
            result['elapsed_seconds'] = round(time.monotonic()-start, 2)
            save(pid + '.result.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result
        results = await asyncio.gather(*(one(r) for r in rows))
    save('results.json', results)

if __name__ == '__main__':
    asyncio.run(main())
