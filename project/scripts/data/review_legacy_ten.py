"""Offline, read-only checks of inputs; creates review artifacts once, no API."""
import json
from pathlib import Path
from probe_teacher_v1 import save, digest
from rgrpo.math_score_light import compute_math_score_payload, extract_answer

out = Path('/mnt/data/reflection_grpo/data/sft/teacher_legacy_probe10_deepseek_v4_pro_20260910')
samples = json.loads((out / 'samples.json').read_text())
results = {r['probe_id']: r for r in json.loads((out / 'results.json').read_text())}
manifest = json.loads((out / 'manifest.json').read_text())
assert all(digest(Path(s['path'])) == s['sha256'] for s in manifest['sources'].values())
checks = []
parts = ['# 指定旧 prompt：DeepSeek-V4-Pro 10 条试验', '', 'Prompt 与 teacher_legacy_probe4_m3_t03_20260910 保存的请求一致，仅替换题目、参考答案、学生回答。每条一次 API，thinking enabled / high；temperature 0.3 为请求值，不代表 thinking 模式一定采纳。', '', '以下是原始最终 JSON 的标准序列化展示，未翻译或修改；不展示内部 thinking。参考答案匹配不等于数学证明或诊断正确。', '']
for i, row in enumerate(samples, 1):
    result = results[row['probe_id']]
    payload = result.get('parsed_payload', {})
    reasoning = payload.get('reasoning', '')
    score = compute_math_score_payload({'solution_str': reasoning, 'ground_truth': str(row['answer'])})
    checks.append({'probe_id': row['probe_id'], 'gold': row['answer'], 'extracted': extract_answer(reasoning), 'score': score, 'boxed': '\\boxed{' in reasoning})
    parts += [f"## {i}. {row['probe_id']}", '', '题目：' + row['problem'], '', '数据集参考答案：' + str(row['answer']), '', '### API 最终标注', '', '```json', json.dumps(payload, ensure_ascii=False, indent=2), '```', '', '### 学生原回答', '', row['completion'], '']
save(out / 'automatic_checks.json', checks)
with (out / 'SAMPLES_REVIEWED.md').open('x') as f:
    f.write('\n'.join(parts))
print(json.dumps(checks, ensure_ascii=False))
