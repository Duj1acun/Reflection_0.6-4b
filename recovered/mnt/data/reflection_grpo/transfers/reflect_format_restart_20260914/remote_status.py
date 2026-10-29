from pathlib import Path
import json,os,re,subprocess,datetime
report={'time':datetime.datetime.now().isoformat(),'jobs':[]}
for name in ['reflect_main_sftprompt_20260914','reflect_branch_sftprompt_20260914']:
 b=Path('/root')/name;c=json.loads((b/'config.json').read_text());o=Path(c['output']);s=json.loads((b/'SERVER_STATUS.json').read_text())
 try:
  pid=s['launcher_pid'];s['process_sid']=os.getsid(pid);s['process_stat']=Path('/proc')/str(pid)/'stat';s['process_stat']=s['process_stat'].read_text().split(')')[-1].split()[0]
 except Exception:s['process_stat']='missing'
 for label,path in [('probe',o/'probes/reflect/stage1/stage1.log'),('train',o/'runs/reflect/stage1.log')]:
  if not path.exists():continue
  t=path.read_text(errors='replace');t=re.sub(r'\x1b\[[0-9;]*m','',t)
  if label=='train':
   for worker_file in (Path(c['ray_tmp_dir'])/'session_latest/logs').glob('worker-*.out'):
    direct=worker_file.read_text(errors='replace')
    if 'TaskRunner' in direct[:500]:t += '\n'+direct
  steps=[l for l in t.splitlines() if re.search(r'step:\d+ - ',l)]
  entry={'bytes':path.stat().st_size,'last_modified':datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),'probe_completed':'FOURWAY_PROBE_COMPLETED' in t,'traceback':'Traceback (most recent call last)' in t,'tail':t[-450:]}
  progress=re.findall(r'FOURWAY_VALIDATION reflect step=(\d+) samples=(\d+)/(\d+)',t)
  if progress:entry['validation_progress']=progress[-1]
  if steps:
   last=steps[-1];entry['step']=int(re.search(r'step:(\d+)',last).group(1));metrics=dict(re.findall(r'([\w/]+):(-?[\d.]+)',last))
   entry['metrics']={k:v for k,v in metrics.items() if k in ['timing_s/step','actor/grad_norm','branch/root_count','branch/leaf_count','branch/padding_count','branch/reward','branch/accuracy','branch/optimizer_steps','fourway/generated_tokens','perf/max_memory_reserved_gb','reflection_grpo/format/gate_zero_rate','reflection_grpo/format/valid_rate_given_reflection','reflection_grpo/n_correct_reflection','reflection_grpo/reward/depth_1']}
  v=o/('probes/reflect/stage1/validation' if label=='probe' else 'runs/reflect/validation')
  entry['validation_files']=[{'file':p.name,'bytes':p.stat().st_size,'lines':sum(1 for _ in p.open()) if p.suffix=='.jsonl' else None} for p in v.glob('*') if p.is_file()]
  s[label]=entry
 report['jobs'].append(s)
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
print(json.dumps(report,indent=2))
