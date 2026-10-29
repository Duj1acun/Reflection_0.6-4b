"""Detached local supervisor; run only after CPU/GPU preflight succeeds."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/mnt/data/reflection_grpo/artifacts/paired_eval_pass4_20260912')


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def report():
    rows = []
    lines = ['# Qwen3 Base / Phase 2 pass@4', '',
             'Seed 1234；thinking=false；最多两轮；Base/SFT 均要求 boxed 答案；仅完整反思触发续写；未完成项标记为 pending。', '',
             '| Model | Dataset | N | pass@4 | 首轮 pass@4 | avg@4 | 反思触发率 | 末轮截断率 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for size in ['0.6b','4b']:
        for variant in ['base','phase2']:
            model=f'qwen3-{size}-{variant}'
            for ds in ['math500','amc23','aime24','aime25','gpqa']:
                p=OUT/f'{model}__{ds}.summary.json'
                if p.exists():
                    r=json.loads(p.read_text()); rows.append(r)
                    metrics = [f'{r[k]*100:.2f}%' for k in ['pass_at_n','initial_pass_at_n','mean_at_n','reflection_trigger_rate','truncated_rate']]
                    lines.append(f'| {model} | {ds} | {r["n_questions"]} | ' + ' | '.join(metrics) + ' |')
                else:
                    state='scoring pending' if (OUT/f'{model}__{ds}.scoring_pending.json').exists() else 'pending'
                    lines.append(f'| {model} | {ds} | {state} | — | — | — | — | — |')
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'results.json', {'completed_comparisons':len(rows),'expected_comparisons':20,'results':rows})


def main():
    statuses={f'qwen3-{size}-{variant}': {'gpu':gpu,'status':'queued'}
              for size in ['0.6b','4b'] for variant,gpu in [('base',0),('phase2',1)]}
    # Precheck all paths before starting any worker.
    for size in ['0.6b','4b']:
        smoke=OUT/f'qwen3-{size}.generation_smoke.json'
        if not smoke.exists():
            raise RuntimeError(f'GPU smoke missing: {smoke}')
        for variant in ['base','phase2']:
            if (OUT/f'qwen3-{size}-{variant}.run.log').exists():
                raise FileExistsError('A run log already exists; inspect before restarting')
    for size in ['0.6b','4b']:
        children={}
        for variant,gpu in [('base',0),('phase2',1)]:
            key=f'qwen3-{size}-{variant}'
            with (OUT/f'{key}.run.log').open('x') as log:
                p=subprocess.Popen(['bash',str(ROOT/'evaluation/launchers/evaluate_paired_qwen3.sh'),
                                    size,'run',str(gpu),variant], cwd=ROOT,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT)
            children[key]=p
            statuses[key].update(pid=p.pid,status='running',started=time.time())
        while True:
            for key,p in children.items():
                code=p.poll()
                if code is not None and statuses[key]['status']=='running':
                    statuses[key].update(status='completed' if code==0 else 'failed',exit_code=code,ended=time.time())
                    marker=OUT/f'{key}.completed.json'
                    if code==0 and marker.exists():
                        detail=json.loads(marker.read_text())
                        if detail.get('pending_scoring'):
                            statuses[key].update(status='generated_pending_scoring',pending_scoring=detail['pending_scoring'])
            write(OUT/'status.json',statuses)
            report()
            if all(p.poll() is not None for p in children.values()):
                break
            time.sleep(10)
        if any(p.returncode != 0 for p in children.values()):
            for value in statuses.values():
                if value['status']=='queued':
                    value['status']='blocked_by_previous_stage_failure'
            write(OUT/'status.json',statuses)
            raise RuntimeError(f'{size} evaluation failed; next size not launched')


if __name__=='__main__':
    main()
