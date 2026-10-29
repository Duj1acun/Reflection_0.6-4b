"""GPU forward/backward and adapter save/reload checks; no optimizer steps."""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoTokenizer, set_seed
from rgrpo.sft_training import load_single_device_model
from train_phase1 import Phase1Dataset
from train_phase2 import Phase2Dataset


def check(model, dataset, record, name):
    model.zero_grad(set_to_none=True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    batch = {k:v.unsqueeze(0).to(model.device) for k,v in dataset([record], tokenizer, max_len=4096)[0].items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss: '+name)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().max()>0 for g in grads)
    assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    result = dict(loss=float(loss.detach()), supervised_tokens=int((batch['labels']!=-100).sum()),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    model.zero_grad(set_to_none=True)
    del batch, loss, grads
    print(name, result, flush=True)
    return result


if __name__ == '__main__':
    set_seed(42)
    model_id=os.environ['MODEL_ID']
    tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    refl=[json.loads(s) for s in Path(os.environ['DATA_PATH']).open()]
    perfect=[json.loads(s) for s in Path(os.environ['PERFECT_PATH']).open()]
    r=dict(max(refl,key=lambda r:len(r['cut_point'])+len(r['reflection_text'])),type='reflection')
    p=dict(max(perfect,key=lambda r:len(r['full_reasoning'])),type='perfect')
    model=load_single_device_model(model_id,os.environ['ATTN_IMPLEMENTATION'])
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
        lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.enable_input_require_grads()
    report={}
    report['phase1_reflection']=check(model,Phase1Dataset,r,'phase1_reflection')
    report['phase1_perfect']=check(model,Phase1Dataset,p,'phase1_perfect')
    with tempfile.TemporaryDirectory(prefix='adapter_smoke_',dir=os.environ['OUTPUT_ROOT']) as tmp:
        model.save_pretrained(tmp)
        saved={k:v.detach().cpu().clone() for k,v in model.named_parameters() if 'lora_' in k}
        del model
        gc.collect();torch.cuda.empty_cache()
        base=load_single_device_model(model_id,os.environ['ATTN_IMPLEMENTATION'])
        base.config.use_cache=False
        model=PeftModel.from_pretrained(base,tmp,is_trainable=True)
        model.enable_input_require_grads()
        for name,param in model.named_parameters():
            if name in saved:assert torch.equal(param.detach().cpu(),saved[name])
        report['adapter_roundtrip']='exact; trainable'
        report['phase2_continuation']=check(model,Phase2Dataset,r,'phase2_continuation')
        report['phase2_replay']=check(model,Phase2Dataset,dict(r,type='phase1_replay'),'phase2_replay')
        report['phase2_perfect']=check(model,Phase2Dataset,p,'phase2_perfect')
    report['optimizer_steps']=0
    (Path(os.environ['OUTPUT_ROOT'])/'gpu_smoke.json').write_text(json.dumps(report,indent=2))
