# Reflection 0.6B / 4B

Training and evaluation sources, configuration, and result documents migrated from [Duj1acun/Reflect_Qwen3_4b](https://huggingface.co/Duj1acun/Reflect_Qwen3_4b/tree/8e0a9ea0253346fc454b19c1997c369d37bc19c2).

## Contents

- `project/`: the primary `/root/reflection_grpo_clean` source tree restored from the 2026-10-08 backup.
- `recovered/`: additional code, configuration, and reports referenced by that backup, preserving original paths.
- `hf_snapshot/`: separately published code, configuration, and reports, preserving HF paths.
- `migration/report.json`: per-file SHA256 values, source provenance, and exclusions.

The source HF repository remains private. Access is required to download model weights, datasets, and full recovery archives. These large artifacts remain on HF.

## Download model artifacts

Install `huggingface_hub`, run `hf auth login` with an authorized account, then use:

```bash
hf download Duj1acun/Reflect_Qwen3_4b --revision 8e0a9ea0253346fc454b19c1997c369d37bc19c2 --include 'models/*' --local-dir artifacts
```

Choose the required checkpoint directory in the HF file browser; other checkpoints include `global_step_450/` and `qwen3-0.6b-sft-checkpoint-268/`. Some recovery states are Qwen3-0.6B despite the HF repository name; consult the source configuration before loading a model.

## Migration scope

No downloaded code was executed. Backup files were checked against their recorded size and SHA256. Symlinks, source Git metadata, source automation, raw datasets, tokenizer files, weights, caches, archives, credential candidates, and files over 10 MiB were excluded. The report lists exclusions, including files omitted after credential-pattern checks. This verifies the selected transfer, not training/evaluation correctness or a complete security audit.

Original source files and their license notices are preserved. Consult the relevant source directories for dependency versions and execution instructions.
