"""Freeze five distinct questions from the restored-prompt task; no API calls."""
import json
from pathlib import Path
from build_two_stage_sft import read, jsonl, dump, file_hash

BASE = Path('/root/reflection_grpo_clean/outputs')
source = BASE / 'two_stage_sft_original_prompts_20260912'
out = BASE / 'two_stage_sft_original_prompt_probe5_20260912'
manifest = json.loads((source/'manifest.json').read_text())
assert file_hash(source/'candidates.jsonl') == manifest['candidates_sha256']
rows = read(source/'candidates.jsonl')
selected, seen = [], set()
for model in ['qwen3-0.6b', 'qwen3-4b', 'qwen3-0.6b', 'qwen3-4b', 'qwen3-0.6b']:
    row = next(r for r in rows if r['student_model']==model and r['problem_id'] not in seen)
    selected.append(row)
    seen.add(row['problem_id'])
out.mkdir(parents=True, exist_ok=False)
jsonl(out/'candidates.jsonl', selected)
jsonl(out/'perfect.jsonl', [])
manifest.update(candidates_sha256=file_hash(out/'candidates.jsonl'), perfect_sha256=file_hash(out/'perfect.jsonl'),
                reflection_counts={'qwen3-0.6b':3,'qwen3-4b':2},perfect_counts={},
                probe_selection='Five distinct problem IDs; alternating backbone, first eligible candidate in frozen source order.',
                reference_manifest=str(source/'manifest.json'))
dump(out/'manifest.json',manifest)
for r in selected:
    print(r['student_model'],r['uid'],r['problem'], 'GOLD:',r['ground_truth'])
