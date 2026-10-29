from pathlib import Path
import subprocess,json,sys
b=Path(__file__).resolve().parent
r=subprocess.run(['ssh','-F','/dev/null','-S','/tmp/reflect-transfer-telecom.sock','-p','33316','-o','BatchMode=yes','root@180.127.11.169','python3 -'],input=(b/'remote_status.py').read_text(),text=True,capture_output=True)
r.check_returncode();(b/'latest_status.json').write_text(r.stdout)
if '--brief' in sys.argv:
 d=json.loads(r.stdout);print(d['time']);print(d['gpus'])
 for j in d['jobs']:
  print(json.dumps({k:j.get(k) for k in ['gpu','launcher_pid','stage','process_stat','error']},ensure_ascii=False))
  t=j.get('train',{});print('train',json.dumps({k:t.get(k) for k in ['step','metrics','validation_progress','traceback','last_modified']},ensure_ascii=False))
  if j['stage']!='formal_training':print('probe',json.dumps(j.get('probe',{}),ensure_ascii=False))
else:print(r.stdout)
if r.stderr: print(r.stderr)
