# REFLECT+: Strategic Clean-Slate Self-Correction

这是论文 **Beyond Spurious Reflection: Strategic Self-Correction in LLMs via Role-Aware Reinforcement Learning** 的整理版代码。论文 PDF 位于 [paper/EMNLP_26.pdf](paper/EMNLP_26.pdf)。

本目录从原仓库 commit `ce8e1b1` 整理而来，目标是让论文主线清晰、路径可配置、历史实验不干扰正式入口。原目录 `/root/reflection_grpo` 未被修改。

## 方法概览

REFLECT+ 分成两个阶段：

1. 两阶段 LoRA SFT：先学习根据完整的失败回答生成结构化 `<reflection>`，再学习只根据问题与 reflection 继续求解。首错定位描述错误的推理方向，不要求逐字引用失败回答。
2. Role-aware RL：rollout 遇到完整 reflection 后丢弃错误前缀，以 `(x, r)` 重建上下文；奖励使用任务正确性、格式门控与反思效率，advantage 使用同题 direct 子组的反事实均值和 difficulty-gated floor。

Qwen3-0.6B/4B 的模型专属 no-thinking SFT 数据构造、MiniMax 质量门和
断点续跑命令见 [SFT_DATA_PIPELINE.md](SFT_DATA_PIPELINE.md)。

论文最终实现由以下路径组成：

- Clean-Slate rollout：`agent_system/multi_turn_rollout/rollout_loop.py`
- 轨迹奖励：`rgrpo/core_trajectory_grpo.py`
- 反事实 advantage：`rgrpo/decoupled_advantage_dg.py`
- veRL 接入：`rgrpo/core_trajectory_grpo_decoupled_dg.py`

## 快速开始

```bash
cd /root/reflection_grpo_clean
python3 -m pip install -e .
cp .env.example .env
set -a
source .env
set +a
```

按需修改 `.env` 并导入环境变量后，依次执行：

```bash
python3 training/sft/train_phase1.py
python3 training/sft/train_phase2.py
python3 training/sft/merge_adapter.py
bash training/baselines/train_grpo.sh
bash training/rl/train_reflect_plus.sh
```

评测已转换的四个模型：

```bash
GPUS=0,1,2,3,4 \
EVAL_SEEDS="1234 2026 3407 4519 8675" \
bash evaluation/launchers/evaluate_paper.sh
```

完整数据约定、命令和 checkpoint 转换见 [REPRODUCE.md](REPRODUCE.md)。

## 重要限制

- 仓库不包含 API teacher 生成的冷启动数据、基座模型、LoRA adapter、RL checkpoint 或原始 generations。
- 论文冷启动语料统计为 7,632 个 reflection 样本和 2,057 个 perfect 样本；需按 [data/README.md](data/README.md) 放置。
- 原仓库中的 `refl_dg520 step 200` 来自一次历史运行，但日期脚本后来发生过命名和默认值漂移。本目录的 canonical 配置按论文附录恢复，不声称与历史运行逐字节相同。
- `paper/rebuttal` 是论文阶段的材料存档，其中脚本保留历史语境，不属于正式训练入口。

## 文档

- [REPRODUCE.md](REPRODUCE.md)：从数据到评测
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)：目录职责
- [PAPER_ARTIFACT.md](PAPER_ARTIFACT.md)：论文公式、实现与结果对应
- [paper/results/EVAL_RESULTS.md](paper/results/EVAL_RESULTS.md)：原项目评测汇总

本项目基于 veRL/verl-agent，许可证与版权声明见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
