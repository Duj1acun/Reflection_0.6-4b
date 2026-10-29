"""Render saved final outputs, never hidden thinking, for a bounded probe."""
import argparse
from collections import Counter
import json
from pathlib import Path
from build_two_stage_sft import read, jsonl, dump, reflection_text

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    rows=read(args.output/'candidates.jsonl')
    records=[]
    usage=Counter()
    for row in rows:
        base=args.output/'calls'/row['uid']
        dp=Path(str(base)+'.diagnosis.json')
        if not dp.exists():
            continue
        d=json.loads(dp.read_text())
        cp=Path(str(base)+'.continuation.json')
        c=json.loads(cp.read_text()) if cp.exists() else {}
        for result in (d,c):
            usage.update({k:v for k,v in (result.get('usage') or {}).items() if isinstance(v,(int,float))})
        records.append(dict(**row,diagnosis=d,continuation=c))
    jsonl(args.output/'probe_results.jsonl',records)
    lines=['# 两阶段 API 试跑：最终原文','', '以下为 API 最终输出原文，未翻译或修正，不含内部 thinking。','']
    for i,r in enumerate(records,1):
        d,c=r['diagnosis'],r['continuation']
        lines.extend([f"## {i}. {r['student_model']} / {r['trajectory_id']}",'', '### 题目','',r['problem'],'',f"参考答案：{r['ground_truth']}",'','### 第一阶段：结构化反思（原始 JSON）','','```json',d.get('content',json.dumps(d,ensure_ascii=False)),'```',''])
        if d.get('annotation'):
            lines.extend(['### 实际传给第二阶段的反思','',reflection_text(d['annotation']),''])
        lines.extend(['### 第二阶段：最终续推原文','',c.get('content') or ('空输出；状态：'+c.get('status','unknown')+'；结束原因：'+str(c.get('finish_reason'))) if c else '未进入第二阶段。','','### 原始学生回答','',r['model_response'],''])
        lines.extend(['### 接口状态','',json.dumps({stage:{k:value.get(k) for k in ('status','http_status','finish_reason','thinking_returned')} for stage,value in [('diagnosis',d),('continuation',c)]},ensure_ascii=False),''])
    # Generated report, not an edit to source data.
    (args.output/'GENERATED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    summary=dict(records=len(records),unique_problems=len({r['problem_id'] for r in records}),diagnosis_ok=sum(r['diagnosis'].get('status')=='ok' for r in records),continuation_ok=sum(r['continuation'].get('status')=='ok' for r in records),usage=dict(usage))
    dump(args.output/'probe_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
