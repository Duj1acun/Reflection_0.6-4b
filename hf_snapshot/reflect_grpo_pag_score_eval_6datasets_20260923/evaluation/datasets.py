"""Unified loader for the 5 eval datasets -> list of dicts:

  {"problem": <str user text to put in the user message>,
   "gold":    <str gold answer (math) OR gold letter A-D (gpqa)>,
   "kind":    "math" | "gpqa",
   "uid":     <str>}

- math500 : data/rl/math500_rgrpo.jsonl
            `question` is a [system,user] chat list -> take the USER content as the
            problem (the embedded reflection system prompt is NOT reused; the eval
            harness applies the per-model system prompt). gold = `answer`.
- aime24/25/26 : data/eval/aime/aime{2024,2025,2026}.jsonl
            `problem` / `answer` (30 rows each).
- gpqa     : HF arrow cache Wanfq___gpqa/gpqa_diamond (198 rows). Standard
            simple-evals MCQ: 4 options shuffled DETERMINISTICALLY (seeded by
            Record ID) -> gold = the post-shuffle letter; problem text carries the
            MCQ + "Answer: $LETTER" instruction (gpqa judge needs `Answer: X`).
"""

from __future__ import annotations

import glob
import ast
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
HF_CACHE = os.environ.get("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
THIRD_PARTY = os.environ.get("THIRD_PARTY_ROOT", str(PROJECT_ROOT / "third_party"))
MATH500_PATH = os.environ.get("MATH500_PATH", str(DATA_ROOT / "rl" / "math500_rgrpo.jsonl"))
AIME_DIR = os.environ.get("AIME_DIR", str(DATA_ROOT / "eval" / "aime"))
GPQA_DIAMOND_GLOB = (
    os.environ.get("GPQA_CACHE_ROOT", HF_CACHE) + "/Wanfq___gpqa/gpqa_diamond/"
    "*/*/gpqa-train.arrow"
)
CLUTRR_DIR = os.environ.get(
    "CLUTRR_DIR",
    os.path.join(
        THIRD_PARTY,
        "clutrr_data",
        "pregen",
        "data_emnlp_final",
        "data_089907f8",
    ),
)
BBH_LOCAL_DIRS = [
    os.environ.get("BBH_DIR", ""),
    os.path.join(THIRD_PARTY, "BIG-Bench-Hard", "bbh"),
    os.path.join(THIRD_PARTY, "BIG-Bench-Hard"),
    os.path.join(HF_CACHE, "bbh"),
]

BBH_SELECTED_TASKS = [
    "boolean_expressions",
    "dyck_languages",
    "object_counting",
    "tracking_shuffled_objects_three_objects",
    "word_sorting",
    "web_of_lies",
    "navigate",
    "temporal_sequences",
]


def _bbh_all_tasks() -> list[str]:
    for base in BBH_LOCAL_DIRS:
        if not base:
            continue
        bbh_dir = os.path.join(base, "bbh") if os.path.isdir(os.path.join(base, "bbh")) else base
        if not os.path.isdir(bbh_dir):
            continue
        tasks = [
            os.path.splitext(fn)[0]
            for fn in os.listdir(bbh_dir)
            if fn.endswith(".json")
        ]
        if tasks:
            return sorted(tasks)
    return []

# simple-evals GPQA template (openai/simple-evals). The gpqa judge regex is
# r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"  -> the prompt must elicit "Answer: X".
GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{question}\n\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}"
)

# LogiQA 2.0: context + question + 4 options (A-D), same answer format as GPQA.
LOGIQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{context}\n\nQuestion: {question}\n\n"
    "A) {a}\nB) {b}\nC) {c}\nD) {d}"
)

BBH_TEMPLATE = (
    "Solve the following reasoning problem. Think step by step, then put your "
    "final answer within \\boxed{{}}.\n\n"
    "{question}"
)

STRATEGYQA_TEMPLATE = (
    "Answer the following yes/no question. Think step by step, then end your "
    "response with exactly one final line of the form 'Answer: yes' or "
    "'Answer: no'.\n\nQuestion: {question}"
)

DROP_TEMPLATE = (
    "Read the passage and answer the question. Think step by step, then put "
    "your final short answer within \\boxed{{}}.\n\n"
    "Passage: {passage}\n\nQuestion: {question}"
)

CLUTRR_TEMPLATE = (
    "Read the following family story and infer the kinship relation. Think "
    "step by step, then put only the final relation word or phrase within "
    "\\boxed{{}}.\n\n"
    "Story: {story}\n\n"
    "Question: What is {target_person}'s relationship to {source_person}?"
)


def _mmlu_pro_prompt(question: str, options: list[str]) -> str:
    """MMLU-Pro: 3-10 options, letters A-J. Build instruction dynamically."""
    n = len(options)
    letters = "ABCDEFGHIJ"[:n]
    opts = "\n".join(f"{letters[i]}) {options[i]}" for i in range(n))
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'Answer: $LETTER' (without "
        f"quotes) where LETTER is one of {letters}. Think step by step before "
        f"answering.\n\n{question}\n\n{opts}"
    )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_hf_dataset_candidates(
    candidates: list[tuple[str, str | None, str]],
    required_fields: tuple[str, ...] = (),
):
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    from datasets import load_dataset

    last_error: Exception | None = None
    for repo, config, split in candidates:
        try:
            if config is None:
                ds = load_dataset(repo, split=split)
            else:
                ds = load_dataset(repo, config, split=split)
            if required_fields and len(ds) > 0:
                row = ds[0]
                if not all(k in row for k in required_fields):
                    continue
            return ds
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("no dataset candidates configured")


def _find_bbh_json(task: str) -> str | None:
    for base in BBH_LOCAL_DIRS:
        if not base:
            continue
        for path in (
            os.path.join(base, f"{task}.json"),
            os.path.join(base, "bbh", f"{task}.json"),
        ):
            if os.path.isfile(path):
                return path
    return None


def _bbh_examples_from_json(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    examples = data.get("examples", data if isinstance(data, list) else [])
    out = []
    for i, ex in enumerate(examples):
        question = ex.get("input", ex.get("question", ex.get("problem", "")))
        target = ex.get("target", ex.get("answer", ex.get("label", "")))
        if question == "" or target == "":
            continue
        out.append({"input": str(question), "target": str(target), "idx": i})
    return out


def load_math500() -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(_read_jsonl(MATH500_PATH)):
        q = r["question"]
        if isinstance(q, list):
            user = next(
                (m["content"] for m in q if m.get("role") == "user"), None
            )
            if user is None:
                user = q[-1]["content"]
        else:
            user = str(q)
        gold = r.get("answer")
        if gold is None:
            gold = (r.get("reward_model") or {}).get("ground_truth", "")
        out.append(
            {
                "problem": user,
                "gold": str(gold),
                "kind": "math",
                "uid": str(r.get("unique_id", f"math500-{i}")),
            }
        )
    return out


def load_aime(year: int) -> list[dict[str, Any]]:
    path = os.path.join(AIME_DIR, f"aime{year}.jsonl")
    out = []
    for i, r in enumerate(_read_jsonl(path)):
        out.append(
            {
                "problem": r["problem"],
                "gold": str(r["answer"]).strip(),
                "kind": "math",
                "uid": str(r.get("unique_id", f"aime{year}-{i}")),
            }
        )
    return out


def load_gpqa_diamond() -> list[dict[str, Any]]:
    from datasets import Dataset

    matches = glob.glob(GPQA_DIAMOND_GLOB)
    if not matches:
        raise FileNotFoundError(f"GPQA arrow not found: {GPQA_DIAMOND_GLOB}")
    ds = Dataset.from_file(sorted(matches)[0])
    letters = ["A", "B", "C", "D"]
    out = []
    for i, r in enumerate(ds):
        correct = str(r["Correct Answer"]).strip()
        incorrect = [
            str(r["Incorrect Answer 1"]).strip(),
            str(r["Incorrect Answer 2"]).strip(),
            str(r["Incorrect Answer 3"]).strip(),
        ]
        opts = [correct] + incorrect
        rid = str(r.get("Record ID", i))
        # Deterministic per-question shuffle (seeded by Record ID).
        rng = random.Random(f"gpqa-{rid}")
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        gold_letter = letters[order.index(0)]  # where the correct (idx 0) landed
        problem = GPQA_TEMPLATE.format(
            question=str(r["Question"]).strip(),
            a=shuffled[0],
            b=shuffled[1],
            c=shuffled[2],
            d=shuffled[3],
        )
        out.append(
            {
                "problem": problem,
                "gold": gold_letter,
                "kind": "gpqa",
                "uid": f"gpqa_diamond-{rid}",
            }
        )
    return out


def load_amc23() -> list[dict[str, Any]]:
    """MathArena AMC23 = 40 problems from 2023 AMC10/12, integer answer."""
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    from datasets import load_dataset

    ds = load_dataset("math-ai/amc23", split="test")
    return [
        {
            "problem": r["question"],
            "gold": str(r["answer"]).strip(),
            "kind": "math",
            "uid": f"amc23-{r['id']}",
        }
        for r in ds
    ]


def load_gsm8k() -> list[dict[str, Any]]:
    """GSM8K test = 1319 grade-school math problems. Gold from '#### N'."""
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            {
                "problem": r["question"],
                "gold": gold,
                "kind": "math",
                "uid": f"gsm8k-{i}",
            }
        )
    return out


def load_logiqa() -> list[dict[str, Any]]:
    """LogiQA 2.0 test = 1572 logic-reasoning MCQ (A-D)."""
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    from datasets import load_dataset

    ds = load_dataset("jeggers/logiqa2_formatted", split="test")
    out = []
    for r in ds:
        opts = r["options"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        if len(opts) < 4:
            continue
        problem = LOGIQA_TEMPLATE.format(
            context=r["text"],
            question=r["question"],
            a=opts[0],
            b=opts[1],
            c=opts[2],
            d=opts[3],
        )
        out.append(
            {
                "problem": problem,
                "gold": str(r["answer_char"]).strip().upper(),
                "kind": "gpqa",  # MCQ A-D, reuse gpqa judge path
                "uid": f"logiqa-{r['id']}",
            }
        )
    return out


def load_mmlu_pro() -> list[dict[str, Any]]:
    """MMLU-Pro test = 12032 multi-domain MCQ (A-J)."""
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for r in ds:
        opts = list(r["options"])
        problem = _mmlu_pro_prompt(r["question"], opts)
        out.append(
            {
                "problem": problem,
                "gold": str(r["answer"]).strip().upper(),
                "kind": "mmlu_pro",  # MCQ A-J, dedicated judge path
                "uid": f"mmlupro-{r['question_id']}",
            }
        )
    return out


def _load_bbh_tasks(tasks: list[str]) -> list[dict[str, Any]]:
    out = []
    limit_per_task = os.environ.get("BBH_LIMIT_PER_TASK", "").strip()
    limit_per_task_int = int(limit_per_task) if limit_per_task else None
    for task in tasks:
        examples = []
        path = _find_bbh_json(task)
        if path is not None:
            examples = _bbh_examples_from_json(path)
        else:
            # Common HF mirrors expose one config per BBH task. Keep several
            # candidates so the loader works with whichever mirror is cached.
            ds = _load_hf_dataset_candidates(
                [
                    ("lukaemon/bbh", task, "test"),
                    ("lukaemon/bbh", task, "train"),
                    ("maveriq/bigbenchhard", task, "train"),
                    ("tasksource/bigbenchhard", task, "train"),
                ]
            )
            for i, r in enumerate(ds):
                question = (
                    r.get("input")
                    or r.get("question")
                    or r.get("problem")
                    or r.get("inputs")
                    or ""
                )
                target = (
                    r.get("target")
                    or r.get("answer")
                    or r.get("label")
                    or r.get("targets")
                    or ""
                )
                if isinstance(target, list):
                    target = target[0] if target else ""
                if question and target:
                    examples.append(
                        {"input": str(question), "target": str(target), "idx": i}
                    )

        if limit_per_task_int is not None:
            examples = examples[:limit_per_task_int]

        for ex in examples:
            out.append(
                {
                    "problem": BBH_TEMPLATE.format(question=ex["input"].strip()),
                    "gold": ex["target"].strip(),
                    "kind": "bbh",
                    "uid": f"bbh-{task}-{ex['idx']}",
                    "task": task,
                }
            )
    return out


def load_bbh_selected() -> list[dict[str, Any]]:
    """BIG-Bench Hard selected short-context reasoning tasks."""
    return _load_bbh_tasks(BBH_SELECTED_TASKS)


def load_bbh_all() -> list[dict[str, Any]]:
    """All official BIG-Bench Hard tasks found under third_party/BIG-Bench-Hard/bbh."""
    tasks = _bbh_all_tasks()
    if not tasks:
        raise FileNotFoundError(
            "No local BBH JSON files found. Expected third_party/BIG-Bench-Hard/bbh/*.json"
        )
    return _load_bbh_tasks(tasks)


def load_strategyqa_dev() -> list[dict[str, Any]]:
    """StrategyQA labeled dev-sized subset, question-only yes/no reasoning.

    Public HF mirrors with labels expose either 2290 train rows or a 1603/687
    split. We use the first 490 labeled rows from the 687-row split to match the
    dev-sized rebuttal budget.
    """
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    ds = _load_hf_dataset_candidates(
        [
            ("ChilleD/StrategyQA", None, "test"),
            ("ChilleD/StrategyQA", None, "validation"),
            ("wics/strategy-qa", None, "validation"),
            ("wics/strategy-qa", None, "test"),
            ("tasksource/strategy-qa", None, "validation"),
            ("tasksource/strategy-qa", None, "test"),
        ],
        required_fields=("question",),
    )
    out = []
    for i, r in enumerate(ds):
        if "answer" in r:
            ans = r["answer"]
        elif "label" in r:
            ans = r["label"]
        else:
            continue
        if isinstance(ans, str):
            gold = ans.strip().lower()
            if gold in {"true", "1"}:
                gold = "yes"
            elif gold in {"false", "0"}:
                gold = "no"
        else:
            gold = "yes" if bool(ans) else "no"
        if gold not in {"yes", "no"}:
            continue
        out.append(
            {
                "problem": STRATEGYQA_TEMPLATE.format(
                    question=str(r["question"]).strip()
                ),
                "gold": gold,
                "kind": "strategyqa",
                "uid": f"strategyqa-{r.get('qid', r.get('id', i))}",
            }
        )
    return out[:490]


def _drop_answer_aliases(row: dict[str, Any]) -> list[str]:
    aliases: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for v in value:
                add(v)
            return
        text = str(value).strip()
        if text:
            aliases.append(text)

    spans = row.get("answers_spans")
    if isinstance(spans, dict):
        add(spans.get("spans"))

    answer = row.get("answer")
    if isinstance(answer, dict):
        add(answer.get("spans"))
        add(answer.get("number"))
        date = answer.get("date")
        if isinstance(date, dict):
            add(" ".join(str(date.get(k, "")).strip() for k in ("day", "month", "year")).strip())
        else:
            add(date)
    else:
        add(answer)

    validated = row.get("validated_answers")
    if isinstance(validated, list):
        for ans in validated:
            if isinstance(ans, dict):
                add(ans.get("spans"))
                add(ans.get("number"))

    # Preserve order while de-duplicating.
    seen = set()
    deduped = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def load_drop_dev() -> list[dict[str, Any]]:
    """DROP validation split: passage-level discrete reasoning QA."""
    os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE)
    ds = _load_hf_dataset_candidates(
        [
            ("ucinlp/drop", None, "validation"),
            ("ucinlp/drop", None, "dev"),
        ],
        required_fields=("passage", "question"),
    )
    out = []
    for i, r in enumerate(ds):
        aliases = _drop_answer_aliases(dict(r))
        if not aliases:
            continue
        passage = re.sub(r"\s+", " ", str(r["passage"])).strip()
        question = str(r["question"]).strip()
        out.append(
            {
                "problem": DROP_TEMPLATE.format(passage=passage, question=question),
                "gold": aliases,
                "kind": "drop",
                "uid": f"drop-{r.get('query_id', r.get('id', i))}",
            }
        )
    return out


def load_clutrr() -> list[dict[str, Any]]:
    """CLUTRR systematic generalization split from the official EMNLP release.

    Default source: data_089907f8, train k=2,3 and test k=2..10.
    The target relation is the second query entity's relationship to the first.
    """
    if not os.path.isdir(CLUTRR_DIR):
        raise FileNotFoundError(f"CLUTRR_DIR not found: {CLUTRR_DIR}")
    files = sorted(
        f
        for f in glob.glob(os.path.join(CLUTRR_DIR, "*_test.csv"))
        if os.path.isfile(f)
    )
    if not files:
        raise FileNotFoundError(f"No CLUTRR *_test.csv files found under {CLUTRR_DIR}")

    out = []
    for path in files:
        split_name = os.path.basename(path).removesuffix("_test.csv")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                story = re.sub(r"\s+", " ", str(r.get("story", ""))).strip()
                target = str(r.get("target", "")).strip()
                if not story or not target:
                    continue
                try:
                    query = ast.literal_eval(str(r.get("query", "")))
                    source_person, target_person = str(query[0]), str(query[1])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"Bad CLUTRR query in {path}:{i}: {r.get('query')!r}") from exc
                out.append(
                    {
                        "problem": CLUTRR_TEMPLATE.format(
                            story=story,
                            source_person=source_person,
                            target_person=target_person,
                        ),
                        "gold": target,
                        "kind": "clutrr",
                        "uid": f"clutrr-{split_name}-{r.get('id', i)}",
                        "task": f"clutrr_{split_name}",
                    }
                )
    return out


_LOADERS = {
    "math500": load_math500,
    "aime24": lambda: load_aime(2024),
    "aime25": lambda: load_aime(2025),
    "aime26": lambda: load_aime(2026),
    "gpqa": load_gpqa_diamond,
    "amc23": load_amc23,
    "gsm8k": load_gsm8k,
    "logiqa": load_logiqa,
    "mmlu_pro": load_mmlu_pro,
    "bbh_all": load_bbh_all,
    "bbh_selected": load_bbh_selected,
    "clutrr": load_clutrr,
    "strategyqa_dev": load_strategyqa_dev,
    "drop_dev": load_drop_dev,
}

# Per-dataset per-round generation cap (aligned with training: 1024/round).
MAX_TOKENS = {
    "math500": 1024,
    "aime24": 1024,
    "aime25": 1024,
    "aime26": 1024,
    "gpqa": 1024,
    "amc23": 1024,
    "gsm8k": 1024,
    "logiqa": 1024,
    "mmlu_pro": 1024,
    "bbh_all": 1024,
    "bbh_selected": 1024,
    "clutrr": 1024,
    "strategyqa_dev": 1024,
    "drop_dev": 1024,
}

ALL_DATASETS = list(_LOADERS.keys())


def load_dataset(name: str) -> list[dict[str, Any]]:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choices={ALL_DATASETS}")
    return _LOADERS[name]()


if __name__ == "__main__":
    for n in ALL_DATASETS:
        try:
            d = load_dataset(n)
            print(f"{n:9s} n={len(d):4d}  ex.gold={d[0]['gold']!r}  "
                  f"kind={d[0]['kind']}  problem[:80]={d[0]['problem'][:80]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"{n:9s} LOAD ERROR: {type(e).__name__}: {e}")
