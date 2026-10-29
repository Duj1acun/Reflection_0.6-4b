"""Copy the frozen five-sample experiment without changing prompts or samples."""
import json
from pathlib import Path
from build_two_stage_sft import read, jsonl, dump, file_hash

base=Path('/root/reflection_grpo_clean/outputs')
source=base/'two_stage_sft_original_prompt_probe5_20260912'
out=base/'two_stage_sft_original_prompt_probe5_nothinking_20260912'
manifest=json.loads((source/'manifest.json').read_text())
for name in ('candidates','perfect'):
    assert file_hash(source/(name+'.jsonl'))==manifest[name+'_sha256']
out.mkdir(exist_ok=False)
for name in ('candidates','perfect'):
    jsonl(out/(name+'.jsonl'),read(source/(name+'.jsonl')))
    assert file_hash(out/(name+'.jsonl'))==manifest[name+'_sha256']
manifest['thinking_baseline']=str(source)
dump(out/'manifest.json',manifest)
print(out)
