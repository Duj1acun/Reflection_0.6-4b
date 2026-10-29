"""Rejudge completed saved answers, preserving generations and an audit backup."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluation.evaluators.paired_scoring import SCORER_VERSION,score_sample,summarize

OUT=Path('/mnt/data/reflection_grpo/artifacts/paired_eval_pass4_20260912')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['math500','gpqa'],default='math500')
    ap.add_argument('--size',choices=['0.6b','4b'],default='0.6b')
    ap.add_argument('--wait',action='store_true')
    args=ap.parse_args()
    ds=args.dataset
    if args.wait:
        deadline=time.time()+7200
        while not all((OUT/f'qwen3-{args.size}-{v}__{ds}.summary.json').exists() for v in ['base','phase2']):
            if time.time()>deadline:raise TimeoutError('Completed summaries did not appear in two hours')
            time.sleep(5)
    audit=OUT/'reliability_audit'
    backup=audit/time.strftime(f'before_{args.size}_{ds}_rescore_%Y%m%d_%H%M%S')
    backup.mkdir(parents=True)
    changes=[]
    prepared=[]
    with ProcessPoolExecutor(4) as pool:
        for variant in ['base','phase2']:
            tag=f'qwen3-{args.size}-{variant}'
            sample_path=OUT/f'{tag}__{ds}.samples.jsonl'
            summary_path=OUT/f'{tag}__{ds}.summary.json'
            rows=[json.loads(l) for l in sample_path.open()]
            original=json.loads(summary_path.read_text())
            nq=original['n_questions']
            assert len(rows)==nq*4 and summary_path.exists()
            jobs=[(rd['text'],r['gold'],ds) for r in rows for rd in r['rounds']]
            results=iter(pool.map(score_sample,jobs,chunksize=16))
            for r in rows:
                old=r['correct']
                for rd in r['rounds']:
                    rd.pop('score_error',None)
                    rd.update(next(results))
                r.pop('score_error',None)
                r.update({k:r['rounds'][-1][k] for k in ['correct','extracted','extraction_method','score_status','scorer_version']})
                if old!=r['correct']:
                    changes.append({'model':tag,'uid':r['uid'],'sample_index':r['sample_index'],
                                    'gold':r['gold'],'extracted':r['extracted'],'old':old,'new':r['correct']})
            metrics=summarize(rows,nq,4)
            if any(rd['score_status'] in {'exception','timeout'} for r in rows for rd in r['rounds']):
                raise ValueError('Judge errors remain; no output files replaced')
            first={}
            for r in rows:first.setdefault(r['uid'],[]).append(r['rounds'][0]['correct'])
            original.update(metrics,scorer_version=SCORER_VERSION,rescored_at=time.time(),
                initial_pass_at_n=sum(any(v) for v in first.values())/nq,
                wrong_to_correct=sum(not r['rounds'][0]['correct'] and r['correct'] for r in rows),
                correct_to_wrong=sum(r['rounds'][0]['correct'] and not r['correct'] for r in rows),
                intermediate_judge_errors=0)
            prepared.append((sample_path,summary_path,rows,original))
    # Replace only after both models pass validation. Generated text is untouched.
    for sp,mp,rows,metrics in prepared:
        for p in [sp,mp]:shutil.copy2(p,backup/p.name)
        tmp=sp.with_suffix('.rescore.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        tmp.replace(sp)
        tmp=mp.with_suffix('.rescore.tmp');tmp.write_text(json.dumps(metrics,indent=2));tmp.replace(mp)
        print(mp.name,metrics['pass_at_n'],metrics['mean_at_n'],flush=True)
    (audit/f'{args.size}_{ds}_rescore_changes.json').write_text(json.dumps({'version':SCORER_VERSION,
        'backup':str(backup),'changed_sample_count':len(changes),'changes':changes},indent=2,ensure_ascii=False))
    print('Changed samples',len(changes),flush=True)


if __name__=='__main__':main()
