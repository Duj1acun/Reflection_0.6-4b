from pathlib import Path
import hashlib
import json
import shutil

b = Path('/root/score_math7500_0.6b_gpu0_20260915')
assert not b.exists(), 'Refusing to overwrite an existing SCORE bundle'
shared = Path('/root/reflect_math7500_0.6b_20260914')
stable = Path('/root/reflect_branch_stable_20260915')
ignore = shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache')
r = b/'code/reflection_grpo_clean'
p = b/'code/Policy-As-GenVerifier'
shutil.copytree(stable/'code/reflection_grpo_clean', r, ignore=ignore)
shutil.copytree(shared/'code/Policy-As-GenVerifier', p, ignore=ignore)
c = json.loads((stable/'config.initial-save5.json').read_text())
c.update(experiment=b.name, output='/data/reflection_grpo/experiments/'+b.name,
         reflection_repo=str(r), pag_repo=str(p), gpus='0', tokens_per_gpu=16384,
         val_before_train=False, save_frequency=50, ray_num_cpus=12,
         ray_tmp_dir='/tmp/sc-'+hashlib.sha256(b.name.encode()).hexdigest()[:10])
c.pop('reflection_branching', None)
c['probe_train_batch_size'] = c['train_batch_size']
c['probe_train_n'] = c['train_n']
raw = json.dumps(c, indent=2)+'\n'
for f in [b/'config.json', r/'experiments/four_methods/config.json', r/'experiments/four_methods/config.math7500.json']:
    f.write_text(raw)
out = Path(c['output']); out.mkdir(parents=True, exist_ok=False)
(out/'initialization').mkdir()
(out/'initialization/reflect_merged').symlink_to(shared/'models/reflect_merged')
(b/'results').symlink_to(out)

def replace(path, old, new, count=1):
    s=path.read_text()
    assert s.count(old)==count, (str(path), old, s.count(old))
    path.write_text(s.replace(old,new))

replace(r/'fourway/configuration.py', '"trainer.val_before_train": not probe,',
        '"trainer.val_before_train": c.get("val_before_train", True) and not probe,')
replace(p/'verl/trainer/main_ppo.py',
        '        ray.init(**({"num_cpus": 16, "object_store_memory": 2147483648, "include_dashboard": False, "_temp_dir": "/mnt/data/tmp/fw"} if os.environ.get("FOURWAY_CONFIG") else {}), runtime_env={',
        '        ray_options = {}\n'
        '        if os.environ.get("FOURWAY_CONFIG"):\n'
        '            from fourway.contracts import config as experiment_config, gpu_count\n'
        '            settings = experiment_config()\n'
        '            ray_options = {"num_cpus": settings.get("ray_num_cpus", 12), "num_gpus": gpu_count(settings),\n'
        '                           "object_store_memory": 2147483648, "include_dashboard": False, "_temp_dir": settings["ray_tmp_dir"]}\n'
        '        ray.init(**ray_options, runtime_env={')
manager=p/'verl/workers/sharding_manager/fsdp_vllm.py'
replace(manager, 'import torch\n', 'import torch\nfrom rgrpo.vllm_lifecycle import synchronized_transition\n')
replace(manager, 'self.inference_engine.wake_up()', 'synchronized_transition(self.inference_engine.wake_up)')
replace(manager, 'self.inference_engine.sleep(level=1)', 'synchronized_transition(self.inference_engine.sleep, level=1)')
replace(manager, '            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")',
        '            torch.cuda.synchronize()\n            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")')
rollout=p/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
replace(rollout, 'import torch\n', 'import torch\nfrom rgrpo.vllm_lifecycle import synchronized_transition\n')
replace(rollout, 'self.inference_engine.sleep(level=1)', 'synchronized_transition(self.inference_engine.sleep, level=1)')

(r/'experiments/four_methods/run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec /data/reflection_grpo/envs/reflect_math7500/bin/python -m fourway.cli "$@"
''')
for alias,target in [(Path('/root/reflection_grpo_clean'),r),(Path('/root/Policy-As-GenVerifier'),p)]:
    assert not alias.exists() and not alias.is_symlink(), str(alias)
    alias.symlink_to(target)
files=sorted(x for x in (b/'code').rglob('*') if x.is_file())
(b/'CODE_SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+str(x.relative_to(b))+'\n' for x in files))
print(json.dumps({'bundle':str(b),'gpu':c['gpus'],'steps':c['steps'],'stage1_steps':c['score_stage1_steps'],
                  'config_sha256':hashlib.sha256(raw.encode()).hexdigest(),'code_files':len(files)},indent=2))
