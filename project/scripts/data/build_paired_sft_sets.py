#!/usr/bin/env python3
"""Offline paired reflection/perfect selection; never calls a teacher API."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgrpo.math_score_light import compute_math_score
from rgrpo.sft_data import SYSTEM_PROMPT
from rgrpo.chat_template import apply_chat_template


def read(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question(row):
    return ' '.join(row['problem'].split())


def quality_flags(text):
    flags = []
    if re.search(r'<\s*/?\s*(?:think|reflection)\b', text, re.I):
        flags.append('embedded_think_or_reflection_markup')
    if re.search(r'\b(?:I|we)\s+(?:made|have made)\s+(?:a|an)\s+(?:mistake|error)\b|'
                 r'\b(?:I|we)\s+(?:was|were)\s+(?:wrong|mistaken)\b|'
                 r'\blet (?:me|us) correct\b|\bmy (?:previous|earlier) (?:answer|calculation) was\b', text, re.I):
        flags.append('explicit_self_correction_language')
    paragraphs = [' '.join(p.split()) for p in re.split(r'\n\s*\n', text)]
    counts = Counter(p for p in paragraphs if len(p) >= 120)
    if any(n >= 3 for n in counts.values()):
        flags.append('long_paragraph_repeated_three_times')
    return flags


def stratified_sample(rows, target, seed):
    # Preserve source/domain diversity; proportional quotas with deterministic ties.
    groups = defaultdict(list)
    for row in rows:
        groups[json.dumps([row['source'], row['domain']], sort_keys=True)].append(row)
    target = min(target, len(rows))
    if not rows:
        return []
    quota = {k: len(v) * target // len(rows) for k, v in groups.items()}
    order = sorted(groups, key=lambda k: (-(len(groups[k]) * target % len(rows)), k))
    for k in order[:target - sum(quota.values())]:
        quota[k] += 1
    selected = []
    for k in sorted(groups):
        pool = sorted(groups[k], key=lambda r: r['uid'])
        random.Random(f'{seed}:{k}').shuffle(pool)
        selected.extend(pool[:quota[k]])
    return selected


def length_audit(rows, tokenizer):
    totals = defaultdict(Counter)
    details = []
    for r in rows:
        header = apply_chat_template(tokenizer, [dict(role='system', content=SYSTEM_PROMPT),
                    dict(role='user', content=r['problem'])], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        if r['type'] == 'perfect':
            variants = [('perfect', header, r['full_reasoning'])]
        else:
            variants = [('phase1_reflection', header + r['cut_point'].rstrip() + '\n', r['reflection_text']),
                        ('phase2_continuation', header + r['reflection_text'].rstrip() + '\n', r['reasoning'])]
        for kind, prefix, target in variants:
            plen = len(tokenizer(prefix, add_special_tokens=False)['input_ids'])
            flen = len(tokenizer(prefix + target + tokenizer.eos_token, add_special_tokens=False)['input_ids'])
            label_tokens = max(0, flen - plen)
            c = totals[r['split'] + ':' + kind]
            c['rows'] += 1
            c['untruncated_label_tokens'] += label_tokens
            c['max_full_tokens'] = max(c['max_full_tokens'], flen)
            for limit in (2048, 4096):
                c[f'over_{limit}'] += flen > limit
                c[f'zero_labels_{limit}'] += plen >= limit
                c[f'label_tokens_{limit}'] += max(0, min(flen, limit) - min(plen, limit))
            details.append(dict(uid=r['uid'], split=r['split'], kind=kind,
                                prefix_tokens=plen, full_tokens=flen, label_tokens=label_tokens))
    return {k: dict(v) for k, v in totals.items()}, details


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    args.output.mkdir(parents=True)
    parent = json.loads((args.input / 'manifest.json').read_text())
    manifest = dict(parent=str(args.input.resolve()), parent_manifest_sha256=digest(args.input/'manifest.json'),
                    selection='same-question correct response first; then all-four-correct; one perfect per question',
                    quality='local answer recheck and heuristic exclusions only; no complete semantic proof review',
                    seed=42, models={})
    for model, model_name in [('qwen3-0.6b', 'Qwen3-0.6B'), ('qwen3-4b', 'Qwen3-4B')]:
        src = args.input / model
        reflections = read(src / 'reasoning_outputs.jsonl')
        originals = read(src / 'reflection_data_prepared_perfect.jsonl')
        source = Path(parent['sources'][model]['path'])
        assert digest(source) == parent['sources'][model]['sha256']
        groups = defaultdict(list)
        for r in read(source):
            groups[r['problem_id']].append(r)
        paired_questions = {question(r): r['split'] for r in reflections}
        selected = []
        stable = defaultdict(list)
        audit = []
        for base in originals:
            q = question(base)
            kind = 'paired_correct' if q in paired_questions else 'all_four_correct'
            if kind == 'all_four_correct' and base['correct_out_of_four'] != 4:
                continue
            if q in paired_questions:
                assert base['split'] == paired_questions[q]
            pool = [r for r in groups[base['problem_id']] if r['score_status'] == 'correct'
                    and r.get('finish_reason') == 'stop' and r['completion'].strip()]
            # A concise candidate is preferred only among already answer-correct responses.
            pool.sort(key=lambda r: (len(r['completion']), r['sample_index']))
            chosen = None
            for r in pool:
                assert question(r) == q and str(r['answer']) == str(base['ground_truth'])
                flags = quality_flags(r['completion'])
                score = None
                if not flags:
                    score = compute_math_score(generated=r['completion'], ground_truth=r['answer'])
                    if score != 1.0:
                        flags.append('local_answer_recheck_failed')
                audit.append(dict(student_model=model, problem_id=base['problem_id'],
                    trajectory_id=r['trajectory_id'], split=base['split'], selection_group=kind,
                    flags=flags, reference_score=score, passed_candidate_screen=not flags))
                if flags:
                    continue
                uid = hashlib.sha256((model + ':' + str(r['trajectory_id'])).encode()).hexdigest()
                chosen = dict(base, uid=uid, index=uid, trajectory_id=r['trajectory_id'],
                    model_response=r['completion'], full_reasoning=r['completion'], type='perfect',
                    selection_group=kind, reference_score=score, semantic_reviewed=False,
                    quality_checks='source_correct_stop + local_answer_match + heuristic_screen',
                    source=r.get('source'), domain=r.get('domain'),
                    original_exported_trajectory_id=base['trajectory_id'])
                break
            if chosen:
                if kind == 'paired_correct':
                    selected.append(chosen)
                else:
                    stable[base['split']].append(chosen)
        for split in ['train', 'eval']:
            target = len(stable[split]) if model == 'qwen3-0.6b' else (200 if split == 'train' else 22)
            selected.extend(stratified_sample(stable[split], target, f'42:{model}:{split}'))
        selected.sort(key=lambda r: (r['split'], r['uid']))
        refl = [dict(r, type='reflection') for r in reflections]
        assert len({question(r) for r in selected}) == len(selected)
        assert all(r['score_status']=='correct' and r['finish_reason']=='stop' for r in selected)
        mixed = refl + selected
        train_q = {question(r) for r in mixed if r['split']=='train'}
        eval_q = {question(r) for r in mixed if r['split']=='eval'}
        assert not train_q & eval_q
        random.Random(42).shuffle(mixed)
        out = args.output / model
        out.mkdir()
        write(out/'reasoning_outputs.jsonl', refl)
        write(out/'reflection_data_prepared_perfect.jsonl', selected)
        write(out/'phase1_mixed.jsonl', mixed)
        for split in ['train','eval']:
            write(out/f'reflection_{split}.jsonl', [r for r in refl if r['split']==split])
            write(out/f'perfect_{split}.jsonl', [r for r in selected if r['split']==split])
        write(out/'perfect_selection_audit.jsonl', audit)
        paired_actual = {question(r) for r in selected if r['selection_group']=='paired_correct'}
        write(out/'unpaired_reflection_questions.jsonl', [dict(problem=q, split=split)
              for q,split in paired_questions.items() if q not in paired_actual])
        print(model, 'selection done; auditing token lengths', flush=True)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/'+model_name, local_files_only=True)
        lengths, details = length_audit(mixed, tokenizer)
        write(out/'length_audit.jsonl', details)
        summary = dict(reflection=dict(Counter(r['split'] for r in refl)),
            perfect=dict(Counter(r['split'] for r in selected)),
            perfect_groups=dict(Counter(r['split']+':'+r['selection_group'] for r in selected)),
            unpaired_reflection_questions=len(paired_questions.keys()-paired_actual),
            excluded_candidate_reasons=dict(Counter(flag for a in audit for flag in a['flags'])),
            alternative_trajectories_selected=sum(r['trajectory_id']!=r['original_exported_trajectory_id'] for r in selected),
            lengths=lengths)
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        manifest['models'][model]=dict(summary=summary,
            input_sha256={n:digest(src/n) for n in ['reasoning_outputs.jsonl','reflection_data_prepared_perfect.jsonl']},
            source_sha256=digest(source), output_sha256={f.name:digest(f) for f in out.glob('*.jsonl')})
        print(model, json.dumps({k:v for k,v in summary.items() if k!='lengths'}), flush=True)
    manifest['builder_sha256']=digest(Path(__file__))
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
