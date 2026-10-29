"""Read-only schema/context check before starting an aligned PAG/SCoRe run."""
import ast
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer


def template_overhead(tokenizer, method):
    # Read the actual rollout templates without importing CUDA/vLLM workers.
    root = Path(__file__).resolve().parents[1]
    filename = "vllm_pag_rollout_spmd.py" if method == "pag" else "vllm_multiturn_rollout_spmd.py"
    tree = ast.parse((root / "verl/workers/rollout/vllm_rollout" / filename).read_text())
    attribute = "prompt_templates" if method == "pag" else "feedback_options"
    templates = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == attribute for target in node.targets
        ):
            templates = ast.literal_eval(node.value)
            break
    if templates is None:
        raise ValueError(f"Cannot locate rollout templates: {attribute}")
    keys = ["verify", "regenerate"] if method == "pag" else ["default"]
    total = 0
    for key in keys:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": templates[key]}], tokenize=False, add_generation_prompt=True
        )
        if rendered.startswith("<|im_start|>system"):
            rendered = rendered[rendered.find("<|im_end|>") + len("<|im_end|>"):].lstrip()
        total += len(tokenizer.encode("\n\n" + rendered, add_special_tokens=False))
    return total


def main():
    method, model, train, val, prompt_cap, response_cap, context_cap = sys.argv[1:]
    prompt_cap, response_cap, context_cap = map(int, (prompt_cap, response_cap, context_cap))
    for path in (Path(model) / "config.json", Path(train), Path(val)):
        if not path.is_file():
            raise SystemExit(
                f"Missing aligned input: {path}\n"
                "Set MODEL_PATH/TRAIN_FILE/TEST_FILE to the comparison inputs. "
                "Use quick_start/prepare_aligned_data.py to convert reference JSONL files."
            )
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if context_cap > config.max_position_embeddings:
        raise SystemExit(f"MAX_MODEL_LEN={context_cap} exceeds model context={config.max_position_embeddings}")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    overhead = template_overhead(tokenizer, method)
    segments = 3 if method == "pag" else 2
    for filename in (train, val):
        table = pq.read_table(filename)
        if not {"prompt", "reward_model", "data_source"} <= set(table.column_names):
            raise SystemExit(f"{filename}: require prompt, reward_model and data_source columns")
        if not table.num_rows:
            raise SystemExit(f"Empty dataset: {filename}")
        max_prompt = 0
        for row in table.select(["prompt", "reward_model"]).to_pylist():
            if not isinstance(row["reward_model"], dict) or row["reward_model"].get("ground_truth") is None:
                raise SystemExit(f"{filename}: reward_model.ground_truth is missing")
            ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            max_prompt = max(max_prompt, min(len(ids), prompt_cap))
        required = max_prompt + overhead + segments * response_cap
        if required > context_cap:
            safe_response = (context_cap - max_prompt - overhead) // segments
            raise SystemExit(
                f"{filename}: worst-case context {required} > MAX_MODEL_LEN={context_cap} "
                f"(prompt={max_prompt}, feedback={overhead}, generated={segments}*{response_cap}). "
                f"For this dataset/model set MAX_RESPONSE_LENGTH<={safe_response}, or use a "
                "model with a longer supported context and report the changed comparison budget."
            )
        print(f"[preflight] {filename}: {table.num_rows} rows, worst-case context {required}/{context_cap}")
    print(f"[preflight] {method}: at most {segments * response_cap} generated tokens; feedback adds {overhead}")


if __name__ == "__main__":
    main()
