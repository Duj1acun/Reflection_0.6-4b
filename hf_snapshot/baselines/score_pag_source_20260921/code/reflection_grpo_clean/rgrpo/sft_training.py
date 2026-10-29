"""Shared validation, causal metrics, and encoding for the two SFT stages."""
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import TrainerCallback


class EvaluateLastStep(TrainerCallback):
    """Include the final update even when max_steps is not an eval interval."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def save_checkpoint_selection(trainer, output_dir):
    trainer.save_state()
    Path(output_dir, 'checkpoint_selection.json').write_text(json.dumps({
        'criterion': 'minimum eval_loss on the fixed validation split',
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_eval_loss': trainer.state.best_metric,
        'last_global_step': trainer.state.global_step,
        'final_adapter_contains': 'best checkpoint weights reloaded by Trainer',
    }, indent=2))


def encode_supervised(tokenizer, prefix, target, max_len):
    """Mask the prompt by character offsets; reject silent target truncation."""
    if not target.strip():
        raise ValueError('Empty SFT target')
    text = prefix + target + tokenizer.eos_token
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) > max_len:
        raise ValueError(f'SFT sequence has {len(enc["input_ids"])} tokens, exceeds MAX_LEN={max_len}')
    labels = [token if end > len(prefix) else -100
              for token, (_, end) in zip(enc['input_ids'], enc['offset_mapping'])]
    # The terminal EOS is a real supervised token, even if a tokenizer reports
    # a special-token offset of (0, 0). Padding EOS tokens remain masked below.
    if enc['input_ids'][-1] != tokenizer.eos_token_id:
        raise ValueError('SFT sequence does not end with tokenizer EOS')
    labels[-1] = enc['input_ids'][-1]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError('SFT sample has no causal target labels')
    pad = max_len - len(labels)
    return dict(input_ids=torch.tensor(enc['input_ids'] + [tokenizer.pad_token_id] * pad),
                attention_mask=torch.tensor([1] * len(labels) + [0] * pad),
                labels=torch.tensor(labels + [-100] * pad))


def causal_token_metrics(eval_pred):
    predictions, labels = eval_pred
    # logits[t] predicts token[t+1], matching ForCausalLMLoss.
    predictions, labels = predictions[..., :-1], labels[..., 1:]
    mask = labels != -100
    count = int(mask.sum())
    return {'token_accuracy': float(((predictions == labels) & mask).sum() / count) if count else 0.0}


def preprocess_token_logits(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def validate_split_groups(train, evaluation):
    if not train or not evaluation:
        raise ValueError('Both training and evaluation sets must be nonempty')
    train_questions = {' '.join(r['problem'].split()) for r in train}
    eval_questions = {' '.join(r['problem'].split()) for r in evaluation}
    if train_questions & eval_questions:
        raise ValueError('Same question appears in both train and eval; use explicit question-level splits')


def audit_datasets(datasets, output_path):
    report = {}
    for split, dataset in datasets.items():
        by_type = {}
        seen = {}
        for i, row in enumerate(dataset.records):
            key = (row.get('uid', i), row['type'])
            if key not in seen:
                item = dataset[i]
                mask = item['labels'][1:] != -100
                seen[key] = (int(item['attention_mask'].sum()), int(mask.sum()))
                if not seen[key][1]:
                    raise ValueError(f'Zero supervised tokens: {split} {key}')
                if torch.any(item['labels'][item['attention_mask'] == 0] != -100):
                    raise ValueError('Padding is supervised')
            length, supervised = seen[key]
            count = by_type.setdefault(row['type'], Counter())
            count['rows'] += 1
            count['label_tokens'] += supervised
            count['max_sequence_tokens'] = max(count['max_sequence_tokens'], length)
        report[split] = {k: dict(v) for k, v in by_type.items()}
    Path(output_path).write_text(json.dumps(report, indent=2))
    print('[Preflight] All train/eval samples have valid masks and fit MAX_LEN.')
    print(json.dumps(report, indent=2))
    return report


def load_single_device_model(model_id, attention):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this BF16 training entry')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES; this entry is single-process')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU does not support BF16')
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation=attention, local_files_only=True)
