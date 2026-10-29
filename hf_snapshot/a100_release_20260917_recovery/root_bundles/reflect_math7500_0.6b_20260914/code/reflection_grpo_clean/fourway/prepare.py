"""Auditable, idempotent preparation; never starts training."""
import collections
import json
from pathlib import Path

from .contracts import (FORMATTING_OVERLAP_IDS, METHODS, atomic_json, file_sha,
                        messages, normalize, problem_id)

def clean(train, validation, verified_overlaps=None):
    groups = collections.defaultdict(list)
    for row in train:
        groups[normalize(row["problem"])].append(row)
    val_keys = {normalize(row["problem"]): row["uid"] for row in validation}
    retained, excluded = [], []
    verified_overlaps = FORMATTING_OVERLAP_IDS if verified_overlaps is None else verified_overlaps
    for key, rows in groups.items():
        reason = None
        if not key or any(not normalize(r["gold"]) for r in rows):
            reason = "empty_question_or_label"
        elif len({normalize(r["gold"]) for r in rows}) != 1:
            reason = "conflicting_labels"
        elif key in val_keys:
            reason = "validation_exact_overlap"
        elif any(r["source_id"] in verified_overlaps for r in rows):
            reason = "validation_verified_formatting_overlap"
        if reason:
            excluded.extend(dict(r, exclusion_reason=reason) for r in rows)
        else:
            retained.append(rows[0])
            excluded.extend(dict(r, exclusion_reason="duplicate_same_label") for r in rows[1:])
    return retained, excluded

def prepare(c):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    output = Path(c["output"]) / "data"
    output.mkdir(parents=True, exist_ok=True)
    if c.get("train_source_sha256") and file_sha(c["train_source"]) != c["train_source_sha256"]:
        raise ValueError("Training source changed; re-audit the source and exclusion list")
    def read_source(path, dataset):
        out = []
        for i, r in enumerate(pq.read_table(path).to_pylist()):
            problem = "\n".join(m["content"] for m in r["prompt"] if m["role"] == "user")
            source_id = str(r.get("unique_id", r["extra_info"]["index"]))
            out.append(dict(uid=problem_id(problem), source_id=source_id, source_row=i,
                            problem=problem, gold=str(r["reward_model"]["ground_truth"]), dataset=dataset))
        return out
    dataset = c.get("train_dataset", "dapo")
    train = read_source(c["train_source"], dataset)
    math = read_source(c["math500_source"], "math500")
    amc = [dict(r, source_id=r["uid"], source_row=i, uid=problem_id(r["problem"]), dataset="amc23")
           for i, r in enumerate(json.loads(x) for x in Path(c["amc23_source"]).read_text().splitlines() if x.strip())]
    val = math + amc
    if len(math) != 500 or len(amc) != 40 or len({r["uid"] for r in val}) != 540:
        raise ValueError("Unexpected validation counts / duplicate questions")
    verified = c.get("verified_overlap_source_ids", FORMATTING_OVERLAP_IDS if dataset == "dapo" else {})
    retained, excluded = clean(train, val, verified)
    expected = c.get("expected_train_questions", 17133 if dataset == "dapo" else None)
    if expected is None or len(retained) != expected:
        raise ValueError(f"Source changed: expected {expected} cleaned questions, got {len(retained)}")
    sources = {k: file_sha(c[k]) for k in ("train_source", "math500_source", "amc23_source")}
    old = output / "manifest.json"
    if old.exists() and json.loads(old.read_text())["source_sha256"] != sources:
        raise ValueError("Source hashes changed; use a new experiment output")
    tok = AutoTokenizer.from_pretrained(c["base_model"], local_files_only=True)
    manifest = {"source_sha256": sources, "raw_train_rows": len(train), "train_questions": len(retained),
                "validation_questions": {"math500": len(math), "amc23": len(amc)},
                "excluded": dict(collections.Counter(r["exclusion_reason"] for r in excluded)), "methods": {}}
    for method in METHODS:
        manifest["methods"][method] = {}
        for split, records in (("train", retained), ("validation", val)):
            rows = [dict(prompt=messages(r["problem"], method),
                         reward_model={"ground_truth": r["gold"], "style": "rule"},
                         data_source=r["dataset"], ability="math",
                         extra_info={"index": r["uid"], "source_id": r["source_id"], "source_row": r["source_row"], "split": split})
                    for r in records]
            texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
            lengths = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
            if max(lengths) > c["max_prompt_length"]:
                raise ValueError(f"Overlong {method}/{split}: {max(lengths)}")
            path = output / f"{method}.{split}.parquet"
            table = pa.Table.from_pylist(rows)
            if path.exists():
                if pq.read_table(path).to_pylist() != rows:
                    raise ValueError(f"Refusing to overwrite different data: {path}")
            else:
                pq.write_table(table, path)
            manifest["methods"][method][split] = {"path": str(path), "sha256": file_sha(path),
                                                        "questions": len(rows), "max_prompt_tokens": max(lengths)}
    for name, rows in (("train.canonical", retained), ("validation.canonical", val), ("excluded", excluded)):
        path = output / f"{name}.jsonl"
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite changed canonical data: {path}")
        path.write_text(content)
    # All backends use the same torch RandomSampler and full-batch/drop-last policy.
    import torch
    gen = torch.Generator().manual_seed(c["seed"])
    sampler = torch.utils.data.RandomSampler(retained, generator=gen)
    schedule = []
    while len(schedule) < c["steps"]:
        order = list(sampler)
        for start in range(0, len(order) - c["train_batch_size"] + 1, c["train_batch_size"]):
            schedule.append([retained[i]["uid"] for i in order[start:start+c["train_batch_size"]]])
            if len(schedule) == c["steps"]:
                break
    atomic_json(output / "batch_schedule.json", schedule)
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

def merge(c):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    out = Path(c["output"]) / "initialization/reflect_merged"
    adapter = Path(c["sft_adapter"])
    ac = json.loads((adapter / "adapter_config.json").read_text())
    if Path(ac["base_model_name_or_path"]).resolve() != Path(c["base_model"]).resolve():
        raise ValueError("SFT adapter/base mismatch")
    fingerprint = {"base_config": file_sha(Path(c["base_model"]) / "config.json"),
                   "adapter": file_sha(adapter / "adapter_model.safetensors"),
                   "base_weights": {p.name: file_sha(p) for p in Path(c["base_model"]).glob("*.safetensors")}}
    if (out / "merge_manifest.json").exists():
        previous = json.loads((out / "merge_manifest.json").read_text())
        if previous["inputs"] != fingerprint or any(file_sha(out / k) != v for k,v in previous["outputs"].items()):
            raise ValueError("Merged initialization fingerprint mismatch")
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Incomplete merge exists; inspect {out}")
    tok = AutoTokenizer.from_pretrained(c["base_model"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(c["base_model"], torch_dtype=torch.float32,
                                                device_map="cpu", local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter).eval()
    ids = tok.apply_chat_template(messages("What is 2+3?", "reflect"), enable_thinking=False,
                                  add_generation_prompt=True, return_tensors="pt")
    torch.set_num_threads(4)
    with torch.no_grad():
        before = model(ids).logits[:, -1].float()
        model = model.merge_and_unload().eval()
        after = model(ids).logits[:, -1].float()
    if not torch.allclose(before, after, atol=2e-4, rtol=2e-4):
        raise ValueError("Merged logits do not match LoRA initialization")
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    atomic_json(out / "merge_manifest.json", {"inputs": fingerprint,
                "fp32_merge_logit_max_error": (before-after).abs().max().item(),
                "outputs": {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}})
    print(f"Merged and verified SFT initialization: {out}")
    return out
