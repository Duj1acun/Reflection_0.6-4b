from pathlib import Path
import json,os,re,subprocess,datetime
report={'time':datetime.datetime.now().isoformat(),'jobs':[]}
for name in [('reflect_tokenbudget_fix_20260915' if Path('/root/reflect_tokenbudget_fix_20260915/SERVER_STATUS.json').exists() else 'reflect_branch_stable_20260915')]:
 b=Path('/root')/name;c=json.loads((b/'config.json').read_text());o=Path(c['output']);s=json.loads((b/'SERVER_STATUS.json').read_text())
 try:
  pid=s['launcher_pid'];s['process_sid']=os.getsid(pid);s['process_stat']=Path('/proc')/str(pid)/'stat';s['process_stat']=s['process_stat'].read_text().split(')')[-1].split()[0]
 except Exception:s['process_stat']='missing'
 for label,path in [('probe',o/'probes/reflect/stage1/stage1.log'),('train',o/'runs/reflect/stage1.log')]:
  if not path.exists():continue
  t=path.read_text(errors='replace');t=re.sub(r'\x1b\[[0-9;]*m','',t)
  if label=='train':
   for worker_file in (Path(c['ray_tmp_dir'])/'session_latest/logs').glob('worker-*.out'):
    if s.get('resumed_from_step') and worker_file.stat().st_mtime < datetime.datetime.fromisoformat(s['updated_utc']).timestamp():continue
    direct=worker_file.read_text(errors='replace')
    if 'TaskRunner' in direct[:500]:t += '\n'+direct
  steps=[l for l in t.splitlines() if re.search(r'step:\d+ - ',l)]
  entry={'bytes':path.stat().st_size,'last_modified':datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),'probe_completed':'FOURWAY_PROBE_COMPLETED' in t,'traceback':'Traceback (most recent call last)' in t,'tail':t[-450:]}
  if label=='train':
   current_text=t.rsplit('USER_REQUESTED_SAVE_FREQUENCY_50:',1)[-1]
   entry['traceback']='Traceback (most recent call last)' in current_text if s.get('stage')!='reconfiguring' else False
   entry['expected_reconfiguration']=s.get('stage')=='reconfiguring'
  progress=re.findall(r'FOURWAY_VALIDATION reflect step=(\d+) samples=(\d+)/(\d+)',t)
  if progress:entry['validation_progress']=progress[-1]
  if steps:
   unique_steps={int(re.search(r'step:(\d+)',line).group(1)):line for line in steps}
   entry['completed_update_steps']=[number for number,line in sorted(unique_steps.items()) if re.search(r'branch/optimizer_steps:1(?:\.0+)?(?:\s|$)',line)]
   last=steps[-1];entry['step']=int(re.search(r'step:(\d+)',last).group(1));metrics=dict(re.findall(r'([\w/]+):(-?[\d.]+)',last))
   entry['metrics']={k:v for k,v in metrics.items() if k in ['timing_s/step','actor/grad_norm','branch/root_count','branch/leaf_count','branch/padding_count','branch/reward','branch/accuracy','branch/reflections','branch/optimizer_steps','fourway/generated_tokens','perf/max_memory_reserved_gb','reflection_grpo/trigger_rate','reflection_grpo/n_reflect_samples','reflection_grpo/n_direct_samples','reflection_grpo/format/gate_zero_rate','reflection_grpo/format/valid_rate_given_reflection','reflection_grpo/n_correct_reflection','reflection_grpo/reward/depth_1']}
   required=['branch/root_count','branch/leaf_count','branch/padding_count','reflection_grpo/n_reflect_samples','reflection_grpo/n_direct_samples']
   if all(k in metrics for k in required):
    roots,leaves,padding,reflected,direct=[int(float(metrics[k])) for k in required]
    if reflected+direct!=leaves+padding:raise ValueError('Cannot reconstruct root reflection rate: inconsistent row counts')
    reflected_roots=roots-(direct-padding)
    if not 0<=reflected_roots<=roots:raise ValueError('Invalid reconstructed reflection root count')
    entry['root_reflection']={'reflected_roots':reflected_roots,'total_roots':roots,'rate':reflected_roots/roots,'source':'derived from unbranched direct leaves excluding zero-weight padding'}
  v=o/('probes/reflect/stage1/validation' if label=='probe' else 'runs/reflect/validation')
  entry['validation_files']=[{'file':p.name,'bytes':p.stat().st_size,'lines':sum(1 for _ in p.open()) if p.suffix=='.jsonl' else None} for p in v.glob('*') if p.is_file()]
  s[label]=entry
 run=o/'runs/reflect'
 s['checkpoints']=[{'directory':p.name,'files':[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file()]} for p in sorted(run.glob('global_step_*')) if p.is_dir()]
 for marker in ['latest.json','best.json']:
  path=run/marker
  if path.exists():
   try:s[marker]=json.loads(path.read_text())
   except (ValueError,OSError):s[marker]={'unreadable':True}
 change=b/'SAVE_FREQUENCY_CHANGE.json'
 if change.exists():s['save_frequency_change']=json.loads(change.read_text())
 report['jobs'].append(s)
report['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
print(json.dumps(report,indent=2))
