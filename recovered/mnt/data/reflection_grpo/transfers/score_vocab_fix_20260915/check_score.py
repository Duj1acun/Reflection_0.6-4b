from pathlib import Path
import json
import subprocess

script=r'''
from pathlib import Path
import json,subprocess,datetime,re,os
b=Path('/root/score_math7500_0.6b_gpu0_vocabfix_20260915');c=json.loads((b/'config.json').read_text());out=Path(c['output'])
s=json.loads((b/'SERVER_STATUS.json').read_text())
s['time']=datetime.datetime.now().isoformat()
try:s['process_sid']=os.getsid(s['launcher_pid'])
except ProcessLookupError:s['process_sid']=None
logs=['gpu_vocab_regression.log','prepare.log','regression_tests.log','gpu_preflight.log','launcher.log','probes/score/stage1/stage1.log','probes/score/stage2/stage2.log','runs/score/stage1.log','runs/score/stage2.log']
s['logs']={}
for rel in logs:
 p=out/rel
 if not p.exists():continue
 t=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace'))
 info={'bytes':p.stat().st_size,'tail':t[-1400:],'traceback':'Traceback (most recent call last)' in t,'backward_proofs':t.count('FOURWAY_NO_UPDATE_PROBE')}
 steps=re.findall(r'step:\d+ - [^\n]+',t)
 if steps:info['last_step']=steps[-1]
 s['logs'][rel]=info
s['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
s['memory']=subprocess.check_output(['free','-m'],text=True)
s['ray_workers']=[]
for p in (Path(c['ray_tmp_dir'])/'session_latest/logs').glob('worker-*'):
 if p.suffix not in ['.out','.err']:continue
 t=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace'))
 if 'TaskRunner' not in t and 'WorkerDict' not in t:continue
 info={'file':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'traceback':'Traceback (most recent call last)' in t,'backward_proofs':t.count('FOURWAY_NO_UPDATE_PROBE')}
 info['probe']=str(out/'probes') in t
 steps=re.findall(r'step:\d+ - [^\n]+',t)
 if steps:info['last_step']=steps[-1]
 if info['traceback']:info['tail']=t[-3000:]
 s['ray_workers'].append(info)
reflect_status_path=Path('/root/reflect_tokenbudget_fix_20260915/SERVER_STATUS.json')
if not reflect_status_path.exists():reflect_status_path=Path('/root/reflect_branch_stable_20260915/SERVER_STATUS.json')
s['reflect_status']=json.loads(reflect_status_path.read_text())
print(json.dumps(s,indent=2))
'''
r=subprocess.run(['ssh','-F','/dev/null','-S','/tmp/reflect-transfer-telecom.sock','-p','33316','-o','BatchMode=yes','root@180.127.11.169','python3 -'],input=script,text=True,capture_output=True)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
(Path(__file__).parent/'latest_status.json').write_text(r.stdout)
d=json.loads(r.stdout)
print(json.dumps({k:d.get(k) for k in ['time','stage','launcher_pid','process_sid','error']},ensure_ascii=False));print(d['gpus']);print(d['memory'])
for p,v in d['logs'].items():
 if d['stage']=='failed' or p.startswith(('probes/','runs/')) or (d['stage']=='native_regression_tests' and p=='regression_tests.log'):
  print(p,json.dumps(v,ensure_ascii=False))
