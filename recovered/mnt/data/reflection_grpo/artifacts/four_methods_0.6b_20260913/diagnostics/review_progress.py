import json,re,collections,statistics
from pathlib import Path
import numpy as np
root=Path('/mnt/data/reflection_grpo/artifacts/four_methods_0.6b_20260913')
s=(root/'runs/grpo/stage1.log').read_text()
metrics=[]
for line in s.splitlines():
 m=re.search(r'\bstep:(\d+) -',line)
 if m and 'timing_s/step:' in line:
  row={k:float(v) for k,v in re.findall(r'([A-Za-z0-9_/]+):(-?[0-9.]+)',line)}
  row['step']=int(m.group(1));metrics.append(row)
train={}
for label,rows in [('all',metrics),('first10',metrics[:10]),('last10',metrics[-10:])]:
 keys=['critic/score/mean','rgrpo/signal/zero_adv_ratio','response_length/mean','response_length/clip_ratio','actor/grad_norm','actor/pg_clipfrac','training/rollout_probs_diff_mean','actor/entropy_loss']
 train[label]={'steps':len(rows),**{k:statistics.mean(x[k] for x in rows) for k in keys}}
 train[label]['tokens']=sum(x['fourway/generated_tokens'] for x in rows)
 train[label]['zero_adv_token_fraction_weighted']=sum(x['fourway/generated_tokens']*x['rgrpo/signal/zero_adv_ratio'] for x in rows)/train[label]['tokens']
 train[label]['cap_hits_reconstructed_from_rounded_ratio']=sum(round(x['response_length/clip_ratio']*512) for x in rows)
 train[label]['responses']=len(rows)*512
train['max_step_clip_ratio']=max(x['response_length/clip_ratio'] for x in metrics)
train['zero_prompt_clip']=all(x['prompt_length/clip_ratio']==0 for x in metrics)
train['seconds']={k:sum(x.get('timing_s/'+k,0) for x in metrics) for k in ['gen','reward','old_log_prob','update_actor','testing','step']}
train['latest_step']=metrics[-1]['step']
train['exception_batches']=[]
for m in re.finditer("list\\(reward_extra_infos_dict.keys\\(\\)\\)=\\['score', 'reward_score_exception'\\]",s):
 n=re.search(r'\bstep:(\d+) -',s[m.end():]);train['exception_batches'].append(int(n.group(1)) if n else None)
validation={};groups={}
for step in [0,50]:
 rows=[json.loads(x) for x in (root/f'runs/grpo/validation/step_{step:04d}.samples.jsonl').read_text().splitlines()]
 validation[step]={};groups[step]={}
 for ds in ['math500','amc23']:
  rr=[r for r in rows if r['dataset']==ds]; gg=collections.defaultdict(list)
  for r in rr:gg[r['uid']].append(r)
  groups[step][ds]=gg
  trunc=[r for r in rr if r['truncated']]
  lengths=np.array([r['generated_tokens'] for r in rr])
  validation[step][ds]={'samples':len(rr),'questions':len(gg),'pass4':sum(any(r['correct'] for r in g) for g in gg.values())/len(gg),'mean4':sum(r['correct'] for r in rr)/len(rr),'truncated':len(trunc),'truncated_at_8192':sum(r['generated_tokens']==8192 for r in trunc),'truncated_correct':sum(r['correct'] for r in trunc),'truncated_question_ids':[uid for uid,g in gg.items() if any(r['truncated'] for r in g)],'failed_questions_with_truncation':sum(not any(r['correct'] for r in g) and any(r['truncated'] for r in g) for g in gg.values()),'tokens_mean':float(lengths.mean()),'tokens_quantiles':dict(zip(['p50','p90','p95','p99','max'],np.quantile(lengths,[.5,.9,.95,.99,1]).tolist())),'score_status':dict(collections.Counter(r['score_status'] for r in rr)),'extraction_method':dict(collections.Counter(r['extraction_method'] for r in rr)),'unique_outputs_per_question':dict(collections.Counter(len({r['final_answer_text'] for r in g}) for g in gg.values()))}
changes={};rng=np.random.default_rng(1234)
for ds in groups[0]:
 uids=list(groups[0][ds]);a=groups[0][ds];b=groups[50][ds]
 pa=np.array([any(r['correct'] for r in a[u]) for u in uids],int);pb=np.array([any(r['correct'] for r in b[u]) for u in uids],int)
 ma=np.array([sum(r['correct'] for r in a[u])/4 for u in uids]);mb=np.array([sum(r['correct'] for r in b[u])/4 for u in uids])
 indices=rng.integers(0,len(uids),size=(20000,len(uids)))
 changes[ds]={'gained_pass_questions':int(sum((pa==0)&(pb==1))),'lost_pass_questions':int(sum((pa==1)&(pb==0))),'pass_delta':float((pb-pa).mean()),'pass_delta_question_bootstrap_95ci':np.quantile((pb-pa)[indices].mean(1),[.025,.975]).tolist(),'mean_delta':float((mb-ma).mean()),'mean_delta_question_bootstrap_95ci':np.quantile((mb-ma)[indices].mean(1),[.025,.975]).tolist()}
report={'training':train,'validation':validation,'paired_changes':changes}
(root/'diagnostics/progress_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
