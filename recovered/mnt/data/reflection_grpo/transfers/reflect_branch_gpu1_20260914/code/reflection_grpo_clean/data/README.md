# Data Layout

数据文件不随仓库复制。期望布局：

```text
data/
├── source/
│   └── limo_merged.jsonl
├── sft/
│   ├── reasoning_outputs.jsonl
│   └── reflection_data_prepared_perfect.jsonl
├── rl/
│   ├── limo_grpo.jsonl
│   ├── limo_rgrpo.jsonl
│   ├── math500_grpo.jsonl
│   └── math500_rgrpo.jsonl
└── eval/
    └── aime/aime{2024,2025,2026}.jsonl
```

论文的 teacher 数据未公开在原仓库。不要用占位数据运行正式训练；缺失时训练脚本应直接报错。

Qwen3 的 backbone-specific SFT 数据不放在本目录。其正式位置、`<=1/64`
Big-Math hard 选择规则、MiniMax 质量门和断点续跑命令见
[`SFT_DATA_PIPELINE.md`](../SFT_DATA_PIPELINE.md)。
