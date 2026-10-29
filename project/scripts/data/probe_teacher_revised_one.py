"""One paid request, no retries, using the revised prompt and a frozen regression case."""
import getpass
import inspect
import json
import sys
import time
from pathlib import Path

import httpx
from prepare_sft_data import _teacher_prompt
from probe_teacher_paratera_ten import ENDPOINT, request_body, validate
from probe_teacher_v1 import save, digest, strict_json


def main():
    out = Path('/mnt/data/reflection_grpo/data/sft/teacher_revised_probe1_deepseek_v4_pro_20260910')
    if out.exists():
        raise RuntimeError('Refusing overwrite or duplicate paid run')
    prior = Path('/mnt/data/reflection_grpo/data/sft/teacher_legacy_probe10_deepseek_v4_pro_20260910')
    manifest = json.loads((prior / 'manifest.json').read_text())
    assert digest(prior / 'samples.json') == manifest['samples_sha256']
    row = next(r for r in json.loads((prior / 'samples.json').read_text()) if r['probe_id'] == 'qwen3-0.6b_k3_01')
    assert sys.stdin.isatty()
    key = getpass.getpass('Paratera API key (hidden, not saved): ').strip()
    if not key:
        raise RuntimeError('Missing key; no request sent')
    out.mkdir(exist_ok=False)
    body = request_body(row)
    save(out / 'sample.json', row)
    save(out / 'request.json', body)
    save(out / 'manifest.json', {'prompt_function_snapshot': inspect.getsource(_teacher_prompt), 'endpoint': ENDPOINT, 'max_requests': 1, 'automatic_retries': 0, 'reference_probe': str(prior)})
    print('REQUEST 1/1', flush=True)
    started = time.monotonic()
    result = {'actual_requests': 1, 'perfect_api_requests': 0}
    try:
        with httpx.Client(timeout=httpx.Timeout(600, connect=30), follow_redirects=False) as client:
            response = client.post(ENDPOINT, json=body, headers={'Authorization': 'Bearer ' + key})
        result['http_status'] = response.status_code
        raw = json.loads(response.text.replace(key, '[REDACTED]'))
        save(out / 'response.json', raw)
        response.raise_for_status()
        choice = raw['choices'][0]
        message = choice['message']
        result.update(response_model=raw.get('model'), finish_reason=choice.get('finish_reason'), thinking_returned=bool(message.get('reasoning_content') or message.get('reasoning_details')), usage=raw.get('usage'))
        payload = strict_json(message.get('content') or '')
        validate(payload)
        result.update(structure_valid=True, parsed_payload=payload)
        if payload['verdict'] == 'unusable':
            assert payload['error_reason'].strip()
            assert all(payload[k] == '' for k in ('verified_states', 'new_direction', 'reasoning'))
    except Exception as exc:
        result['error'] = type(exc).__name__
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    save(out / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
