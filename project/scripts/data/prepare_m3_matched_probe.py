"""Freeze exactly the same five samples and prompts for MiniMax M3."""
import json
import argparse
from pathlib import Path
from build_two_stage_sft import read, jsonl, dump, file_hash
base=Path('/root/reflection_grpo_clean/outputs')
source=base/'two_stage_sft_original_prompt_probe5_20260912'
parser=argparse.ArgumentParser()
parser.add_argument('--output-name',default='two_stage_sft_original_prompt_probe5_m3_nothinking_20260912')
args=parser.parse_args()
if Path(args.output_name).name != args.output_name:
    raise ValueError('Output name must be a directory basename')
out=base/args.output_name
manifest=json.loads((source/'manifest.json').read_text())
for name in ('candidates','perfect'):
    assert file_hash(source/(name+'.jsonl'))==manifest[name+'_sha256']
out.mkdir(exist_ok=False)
for name in ('candidates','perfect'):
    jsonl(out/(name+'.jsonl'),read(source/(name+'.jsonl')))
    assert file_hash(out/(name+'.jsonl'))==manifest[name+'_sha256']
manifest['comparison_baseline']=str(source)
dump(out/'manifest.json',manifest)
print(out)
