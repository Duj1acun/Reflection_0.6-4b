"""Pure Python experiment contracts; deliberately no verl/torch imports."""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "experiments/four_methods/config.json"
METHODS = ("grpo", "reflect", "pag", "score")
BASE_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
REFLECT_PROMPT = (
    "You are a math reasoning assistant. When you make an error during reasoning, "
    "use <reflection>So far I have ... However ...</reflection> to summarize the "
    "reasoning so far and identify the mistake, then continue reasoning based on "
    "the reflection. Put your final answer within \\boxed{}."
)
VERIFY_PROMPT = (
    "Check the math solution step by step. If you find a mistake, state the wrong "
    "step, explain why it is wrong, and end with exactly 'The answer is wrong.' "
    "Otherwise, end with exactly 'The answer is correct.'"
)
REVISE_PROMPT = (
    "Review the previous solution and verification, then provide a corrected "
    "solution. Put your final answer within \\boxed{}."
)
SCORE_PROMPT = (
    "There might be an error in the solution above because of lack of understanding "
    "of the question. Please correct the error, if any, and rewrite the solution. "
    "Put your final answer within \\boxed{}."
)
# Manually adjudicated matches: only mathematical typesetting/spacing differs.
# Do NOT remove all punctuation to deduplicate: z^4+z^2+1 and z^4-z^2+1 differ.
FORMATTING_OVERLAP_IDS = {
    "418b8cc0-633f-4d8e-a944-9dd1755e185d": "test/intermediate_algebra/90.json",
    "61b6424a-23fb-45a8-9b03-d57d10ad1b36": "amc23-5",
    "06390219-477f-4848-8e3d-df373fb51e8d": "amc23-16",
    "791c794d-8e98-4b32-88ec-44b1cf856cd9": "amc23-16",
    "40e3125a-4f8d-4117-93d5-a388d9e842ff": "test/intermediate_algebra/582.json",
    "03397ac4-f821-471c-b44f-bc26cce8123f": "amc23-10",
}

def config(path=None):
    return json.loads(Path(path or os.environ.get("FOURWAY_CONFIG", DEFAULT_CONFIG)).read_text())

def gpu_count(c):
    ids = c["gpus"].split(",")
    if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or any(not x.isdigit() for x in ids):
        raise ValueError("gpus must contain one or two distinct numeric GPU indices")
    return len(ids)


def generated_budget(c, method):
    """Support legacy equal budgets and explicit per-method trajectory budgets."""
    limits = c["total_generated_tokens"]
    limit = limits[method] if isinstance(limits, dict) else limits
    stages = c["stage_tokens"][method]
    if not stages or any(type(v) is not int or v <= 0 for v in stages):
        raise ValueError(f"Invalid stage token budgets for {method}: {stages}")
    if type(limit) is not int or limit != sum(stages):
        raise ValueError(f"Trajectory budget disagrees with stages for {method}")
    return limit

def enabled():
    return bool(os.environ.get("FOURWAY_CONFIG"))

def normalize(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip()

def problem_id(s):
    return hashlib.sha256(normalize(s).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)

def messages(problem, method):
    return [{"role": "system", "content": REFLECT_PROMPT if method == "reflect" else BASE_PROMPT},
            {"role": "user", "content": problem}]

def judgment(text):
    match = re.search(r"The answer is (correct|wrong)\.\s*$", text)
    return match.group(1) if match else None

def token_runs(mask):
    """Half-open ranges of GENERATED tokens; excludes injected user feedback."""
    runs, start = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    return runs

def rank_key(summary):
    if summary.get("datasets"):
        from fractions import Fraction
        ds = list(summary["datasets"].values())
        p = sum(Fraction(round(x["pass_at_4"]*x["questions"]), x["questions"]) for x in ds) / len(ds)
        m = sum(Fraction(round(x["mean_at_4"]*x["samples"]), x["samples"]) for x in ds) / len(ds)
        return (p, m, -summary["step"])
    return (summary["macro_pass_at_4"], summary["macro_mean_at_4"], -summary["step"])

def summarize(records, counts=None):
    from collections import defaultdict
    counts = counts or {"math500": 500, "amc23": 40}
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["score_status"] in {"exception", "timeout"}:
            raise ValueError("Unresolved scoring error")
        grouped[r["dataset"]][r["uid"]].append(r)
    if set(grouped) != set(counts):
        raise ValueError("Missing validation dataset")
    out = {}
    for ds, expected in counts.items():
        groups = grouped[ds]
        if len(groups) != expected or any(len(g) != 4 or {x["sample_index"] for x in g} != set(range(4)) for g in groups.values()):
            raise ValueError(f"{ds}: require every question exactly four times")
        rows = [r for g in groups.values() for r in g]
        refs = [r for r in rows if r.get("has_reflection_open")]
        out[ds] = {
            "questions": expected, "samples": len(rows),
            "pass_at_4": sum(any(r["correct"] for r in g) for g in groups.values()) / expected,
            "mean_at_4": sum(r["correct"] for r in rows) / len(rows),
            "reflection_rate": len(refs) / len(rows),
            "reflection_accuracy": sum(r["correct"] for r in refs) / len(refs) if refs else None,
            "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / len(rows),
            "truncation_rate": sum(r["truncated"] for r in rows) / len(rows),
        }
    return {"datasets": out,
            "macro_pass_at_4": sum(v["pass_at_4"] for v in out.values()) / len(out),
            "macro_mean_at_4": sum(v["mean_at_4"] for v in out.values()) / len(out)}
