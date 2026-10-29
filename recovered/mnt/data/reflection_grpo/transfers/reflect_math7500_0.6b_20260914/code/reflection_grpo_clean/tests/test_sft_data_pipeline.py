import argparse
import json
import os

import pytest

from rgrpo.chat_template import chat_template_kwargs
from scripts.data.prepare_sft_data import (
    STUDENT_SYSTEM_PROMPT,
    command_merge_samples,
    dump_json,
    _expand_sampling_records,
    _completed_sampling_ids,
    _sampling_record_id,
    _select_shard,
    _stable_seed,
)
from rgrpo.sft_data import (
    HARD_THRESHOLD,
    assign_exact_splits,
    build_paper_phase2_mix,
    extract_json_object,
    extract_json_objects,
    problem_id,
    select_hard_records,
    select_pilot_records,
    validate_teacher_payload,
    write_jsonl,
    sha256_file,
)


def test_chat_template_thinking_env_is_strict():
    assert chat_template_kwargs({}) == {}
    assert chat_template_kwargs({"CHAT_TEMPLATE_ENABLE_THINKING": "false"}) == {
        "enable_thinking": False
    }
    assert chat_template_kwargs({"CHAT_TEMPLATE_ENABLE_THINKING": "YES"}) == {
        "enable_thinking": True
    }
    with pytest.raises(ValueError):
        chat_template_kwargs({"CHAT_TEMPLATE_ENABLE_THINKING": "maybe"})


def test_four_way_sampling_ids_and_seeds_are_stable():
    source = [{"problem_id": "abc123", "problem": "Compute 1+1.", "answer": "2"}]
    first = _expand_sampling_records(source, 4)
    second = _expand_sampling_records(source, 4)

    assert first == second
    assert [row["sample_index"] for row in first] == [0, 1, 2, 3]
    assert len({_sampling_record_id(row) for row in first}) == 4
    seeds = [_stable_seed(42, row["problem_id"], row["sample_index"]) for row in first]
    assert len(set(seeds)) == 4
    assert seeds == [
        _stable_seed(42, row["problem_id"], row["sample_index"]) for row in second
    ]


def test_completed_sampling_ids_keep_all_trajectories(tmp_path):
    output = tmp_path / "scored.jsonl"
    rows = _expand_sampling_records(
        [{"problem_id": "same", "problem": "p", "answer": "a"}], 4
    )
    write_jsonl(output, rows)
    assert _completed_sampling_ids(output) == {
        "same:sample:0",
        "same:sample:1",
        "same:sample:2",
        "same:sample:3",
    }


def test_sampling_prompt_is_plain_math_prompt():
    assert "step by step" in STUDENT_SYSTEM_PROMPT
    assert "\\boxed{}" in STUDENT_SYSTEM_PROMPT
    assert "reflection" not in STUDENT_SYSTEM_PROMPT.lower()


def test_problem_shards_are_disjoint_and_cover_source_order():
    source = [{"problem_id": str(index)} for index in range(9)]
    shard_zero = _select_shard(source, 2, 0)
    shard_one = _select_shard(source, 2, 1)

    assert [row["problem_id"] for row in shard_zero] == ["0", "2", "4", "6", "8"]
    assert [row["problem_id"] for row in shard_one] == ["1", "3", "5", "7"]
    assert {row["problem_id"] for row in shard_zero}.isdisjoint(
        {row["problem_id"] for row in shard_one}
    )
    assert len(shard_zero) + len(shard_one) == len(source)


def test_merge_sampling_shards_restores_canonical_order(tmp_path):
    source = [
        {"problem_id": "p0", "problem": "zero", "answer": "0"},
        {"problem_id": "p1", "problem": "one", "answer": "1"},
    ]
    source_path = tmp_path / "source.jsonl"
    write_jsonl(source_path, source)
    expanded = _expand_sampling_records(source, 2)
    shard_paths = [tmp_path / "shard0.jsonl", tmp_path / "shard1.jsonl"]
    write_jsonl(shard_paths[0], expanded[:2])
    write_jsonl(shard_paths[1], expanded[2:])
    common_manifest = {
        "source_sha256": sha256_file(source_path),
        "model": "model",
        "engine": "vllm",
        "system_prompt": STUDENT_SYSTEM_PROMPT,
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 1.0,
        "samples_per_problem": 2,
        "seed": 42,
        "max_new_tokens": 2048,
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "checkpoint_batch_size": 64,
    }
    for shard_index, shard_path in enumerate(shard_paths):
        dump_json(
            shard_path.with_suffix(shard_path.suffix + ".manifest.json"),
            {
                **common_manifest,
                "shard": {"num_shards": 2, "shard_index": shard_index},
            },
        )

    output = tmp_path / "merged.jsonl"
    command_merge_samples(
        argparse.Namespace(
            source=str(source_path),
            input=[str(path) for path in shard_paths],
            output=str(output),
            samples_per_problem=2,
            overwrite=False,
        )
    )
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["trajectory_id"] for row in merged] == [
        row["trajectory_id"] for row in expanded
    ]
    manifest = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text()
    )
    assert manifest["selected_problem_count"] == 2
    assert manifest["selected_trajectory_count"] == 4
    assert manifest["output_rows"] == 4


def test_hard_selection_boundary_dedup_and_conflict_quarantine():
    rows = [
        {
            "problem": "  Solve   x = 1 ",
            "answer": "1",
            "llama8b_solve_rate": HARD_THRESHOLD,
            "source": "a",
            "source_split": "train",
        },
        {
            "problem": "Solve x = 1",
            "answer": "1",
            "llama8b_solve_rate": 0,
            "source": "a",
            "source_split": "test",
        },
        {
            "problem": "Conflict",
            "answer": "2",
            "llama8b_solve_rate": 0,
            "source_split": "train",
        },
        {
            "problem": "Conflict",
            "answer": "3",
            "llama8b_solve_rate": 0,
            "source_split": "test",
        },
        {"problem": "Too easy", "answer": "4", "llama8b_solve_rate": 2 / 64},
    ]
    selected, conflicts, stats = select_hard_records(rows)
    assert [row["problem"] for row in selected] == ["Solve x = 1"]
    assert selected[0]["duplicate_count"] == 2
    assert conflicts[0]["problem"] == "Conflict"
    assert stats["eligible_rows"] == 4
    assert stats["selected_questions"] == 1
    assert stats["conflict_groups"] == 1


def test_pilot_and_splits_are_deterministic():
    records = [
        {
            "problem_id": problem_id(f"p{i}"),
            "problem": f"p{i}",
            "source": f"s{i % 3}",
            "domain": [f"d{i % 4}"],
            "llama8b_solve_rate": (i % 2) / 64,
        }
        for i in range(50)
    ]
    first = select_pilot_records(records, 20, seed=42)
    second = select_pilot_records(records, 20, seed=42)
    assert [x["problem_id"] for x in first] == [x["problem_id"] for x in second]
    assert len({x["problem_id"] for x in first}) == 20

    split = assign_exact_splits(records, seed=42)
    assert sum(x["split"] == "eval" for x in split) == 5
    assert {x["problem_id"] for x in split if x["split"] == "train"}.isdisjoint(
        {x["problem_id"] for x in split if x["split"] == "eval"}
    )


def test_phase2_mix_is_exact_for_64_reflections():
    reflection = [{"problem_id": f"r{i}"} for i in range(64)]
    perfect = [{"problem_id": f"p{i}"} for i in range(5)]
    mixed = build_paper_phase2_mix(reflection, perfect, seed=42)
    assert sum(x["type"] == "reflection" for x in mixed) == 64
    assert sum(x["type"] == "phase1_replay" for x in mixed) == 19
    assert sum(x["type"] == "perfect" for x in mixed) == 17


def test_all_json_candidates_can_be_recovered_from_reasoning_scratch():
    text = 'schema {"reasoning":"placeholder"}; final {"reasoning":"boxed"}'
    assert extract_json_objects(text) == [
        {"reasoning": "placeholder"},
        {"reasoning": "boxed"},
    ]
    assert extract_json_object(text) == {"reasoning": "placeholder"}


def test_teacher_semantic_error_and_full_completion_context():
    source = {
        "completion": "Step one is valid. The answer is 4.",
        "answer": "5",
        "problem": "Compute something.",
    }
    payload = extract_json_object(
        "```json\n"
        '{"verdict":"incorrect","cut_point":"ignored legacy locator",'
        '"verified_states":"step one is valid","error_reason":"the arithmetic is wrong",'
        '"new_direction":"recompute the last step","reasoning":"The answer is 5."}'
        "\n```"
    )
    normalized, errors = validate_teacher_payload(
        payload,
        source,
        scorer=lambda generated, answer: float(answer in generated),
    )
    assert not errors
    assert normalized is not None
    assert normalized["reflection_text"].startswith("<reflection> So far I have:")
    assert normalized["reflection_text"].endswith("</reflection>")
    assert normalized["cut_point"] == source["completion"]
    assert normalized["reflection_context_policy"] == "full_student_completion"

    payload.pop("cut_point")
    normalized, errors = validate_teacher_payload(payload, source)
    assert not errors
    assert normalized is not None
    assert normalized["cut_point"] == source["completion"]


def test_teacher_numeric_conclusion_format_repair_is_conservative():
    source = {
        "completion": "Start. Faulty step.",
        "answer": "053",
        "problem": "Compute a remainder.",
    }
    payload = {
        "verdict": "incorrect",
        "verified_states": "the start is valid",
        "error_reason": "the last step is wrong",
        "new_direction": "recompute it",
        "reasoning": "A correct calculation gives a remainder of 53.",
    }
    normalized, errors = validate_teacher_payload(
        payload,
        source,
        scorer=lambda generated, answer: float(f"\\boxed{{{answer}}}" in generated),
    )
    assert not errors
    assert normalized is not None
    assert normalized["reasoning_answer_format_repair"] == "boxed_numeric_conclusion"

    payload["reasoning"] = "I got 42, but the gold answer is 053."
    normalized, errors = validate_teacher_payload(
        payload,
        source,
        scorer=lambda generated, answer: float(f"\\boxed{{{answer}}}" in generated),
    )
    assert normalized is None
    assert "corrected_reasoning_wrong_answer" in errors


def test_teacher_does_not_require_verbatim_fault_quote():
    source = {
        "completion": "Valid. Now compute $f(z)=az^2+bz+cz^2$. Continue.",
        "answer": "1",
        "problem": "Find a value.",
    }
    payload = {
        "verdict": "incorrect",
        "verified_states": "the setup is valid",
        "error_reason": "the solution combines the two quadratic terms incorrectly",
        "new_direction": "reduce each power correctly",
        "reasoning": r"Therefore, the answer is \boxed{1}.",
    }
    normalized, errors = validate_teacher_payload(
        payload,
        source,
        scorer=lambda generated, answer: float(f"\\boxed{{{answer}}}" in generated),
    )
    assert not errors
    assert normalized is not None
    assert normalized["cut_point"] == source["completion"]
