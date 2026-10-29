"""Offline, read-only checks of inputs; creates review artifacts once, no API."""
import json
from pathlib import Path
from probe_teacher_v1 import save, digest
from rgrpo.math_score_light import compute_math_score_payload, extract_answer

out = Path('/mnt/data/reflection_grpo/data/sft/teacher_revised_probe20_thinking_t01_deepseek_v4_pro_20260910')
samples = json.loads((out / 'samples.json').read_text())
results = {r['probe_id']: r for r in json.loads((out / 'results.json').read_text())}
manifest = json.loads((out / 'manifest.json').read_text())
assert all(digest(Path(s['path'])) == s['sha256'] for s in manifest['sources'].values())
checks = []
parts = ['# 修改后 prompt：DeepSeek-V4-Pro 开启 thinking 的 20 条试验', '', '使用修改后的逐步验证 prompt，同一组 20 条题目；每条一次 API，thinking enabled，reasoning_effort high，temperature 请求值 0.1（不保证服务端采纳），top_p 0.95。', '', '以下是原始最终 JSON 的标准序列化展示，未翻译或修改；不展示内部 thinking。参考答案匹配不等于数学证明或诊断正确。', '']
for i, row in enumerate(samples, 1):
    result = results[row['probe_id']]
    payload = result.get('parsed_payload', {})
    reasoning = payload.get('reasoning', '')
    score = compute_math_score_payload({'solution_str': reasoning, 'ground_truth': str(row['answer'])}) if payload.get('verdict') == 'incorrect' else {'score': None, 'not_applicable': True}
    checks.append({'probe_id': row['probe_id'], 'gold': row['answer'], 'extracted': extract_answer(reasoning), 'score': score, 'boxed': '\\boxed{' in reasoning})
    parts += [f"## {i}. {row['probe_id']}", '', '题目：' + row['problem'], '', '数据集参考答案：' + str(row['answer']), '', '### API 最终标注', '', '```json', json.dumps(payload, ensure_ascii=False, indent=2), '```', '', '### 学生原回答', '', row['completion'], '']
save(out / 'automatic_checks.json', checks)
with (out / 'SAMPLES_REVIEWED.md').open('x') as f:
    f.write('\n'.join(parts))
print(json.dumps(checks, ensure_ascii=False))


