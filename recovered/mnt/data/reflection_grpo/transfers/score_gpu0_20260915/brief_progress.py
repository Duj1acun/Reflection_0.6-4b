from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess, json, re

b=Path(__file__).parent
r=b.parent/'reflect_diagnosis_20260915'
with ThreadPoolExecutor(2) as pool:
    for x in pool.map(lambda p:subprocess.run(['python3',str(p)],stdout=subprocess.DEVNULL,check=True),[b/'check_score.py',r/'check_remote.py']):
        pass
s=json.loads((b/'latest_status.json').read_text())
print(s['time'], 'SCORE', s['stage'], 'PID',s['launcher_pid'],'SID',s['process_sid'],'error',s.get('error'))
print(s['gpus'].strip())
for p,v in s['logs'].items():
    if p.startswith(('probes/','runs/')):
        print(p,'backward proofs',v['backward_proofs'],'traceback',v['traceback'],'bytes',v['bytes'])
for w in s.get('ray_workers',[]):
    if 'last_step' in w:
        line=w['last_step'];print('SCORE probe' if w.get('probe') else 'SCORE training','Ray step',re.search(r'step:(\d+)',line).group(1))
        print({k:v for k,v in re.findall(r'([^\s:]+):([^ ]+)',line) if k in ['actor/grad_norm','critic/grad_norm','timing_s/step','fourway/generated_tokens','critic/score/mean','actor/lr']})
    if w['traceback']:print('ERROR',w.get('tail'))
for j in json.loads((r/'latest_status.json').read_text())['jobs']:
    t=j.get('train',{});print('REFLECT',j['stage'],'PID',j['launcher_pid'],'process',j.get('process_stat'),'completed',t.get('step'),'error',j.get('error'),'traceback',t.get('traceback'),'grad',t.get('metrics',{}).get('actor/grad_norm'))
