from pathlib import Path
import json
import subprocess
import sys

SCRIPT = r'''
from pathlib import Path
import datetime,json,re,subprocess,os
b=Path('/root/reflect_tokenbudget_fix_20260915')
c=json.loads((b/'config.json').read_text());s=json.loads((b/'SERVER_STATUS.json').read_text())
s['checked_at']=datetime.datetime.now().astimezone().isoformat()
try:s['process_sid']=os.getsid(s['launcher_pid'])
except ProcessLookupError:s['process_sid']=None
s['logs']={}
for name in ['launcher.log','cpu_preflight.log','gpu_preflight.log']:
 p=b/name
 if p.exists():
  t=p.read_text(errors='replace');s['logs'][name]={'bytes':p.stat().st_size,'tail':t[-1800:]}
logs=Path(c['ray_tmp_dir'])/'session_latest/logs'
s['workers']=[]
for p in logs.glob('worker-*'):
 if p.suffix not in ['.out','.err']:continue
 t=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace'))
 steps=re.findall(r'step:(\d+) - ([^\n]+)',t)
 proofs=[l for l in t.splitlines() if any(m in l for m in ['TOKEN_BUDGET_LONG_GPU_PASSED','TOKEN_BUDGET_OLD_FAILURE_REPRODUCED','FOURWAY_NO_UPDATE_PROBE'])]
 entry={'file':str(p),'bytes':p.stat().st_size,'modified':datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat()}
 if steps:
  n,line=steps[-1];entry['step']=int(n)
  keys=['actor/grad_norm','branch/optimizer_steps','timing_s/step','timing_s/update_actor','fourway/generated_tokens','perf/max_memory_allocated_gb','perf/max_memory_reserved_gb']
  entry['metrics']={k:v for k,v in re.findall(r'([^\s:]+):([^ ]+)',line) if k in keys}
 if proofs:entry['proofs']=proofs
 if 'Traceback (most recent call last)' in t:entry['error_tail']=t[-4500:]
 elif p.suffix=='.err':entry['tail']=t[-350:]
 s['workers'].append(entry)
s['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
s['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv'],text=True)
print(json.dumps(s,ensure_ascii=False,indent=2))
'''

r = subprocess.run(['ssh','-F','/dev/null','-S','/tmp/reflect-transfer-telecom.sock',
    '-p','33316','-o','BatchMode=yes','root@180.127.11.169','python3 -'],
    input=SCRIPT, text=True, capture_output=True, check=True)
p=Path(__file__).parent/'latest_status.json'
p.write_text(r.stdout)
if '--full' in sys.argv:
    print(r.stdout)
else:
    d=json.loads(r.stdout)
    print(d['checked_at'], d['stage'], 'PID', d['launcher_pid'], 'SID', d['process_sid'], 'error', d.get('error'))
    print(d['gpu'].strip())
    for w in d['workers']:
        if 'step' in w: print('step', w['step'], w['metrics'])
        for proof in w.get('proofs', []): print(proof)
        if 'error_tail' in w: print(w['error_tail'])
    if d['stage']=='failed':
        for name, item in d['logs'].items(): print(name, item['tail'])
    elif d['stage']=='gpu_preflight':
        print(d['logs'].get('gpu_preflight.log', {}).get('tail', '')[-500:])
