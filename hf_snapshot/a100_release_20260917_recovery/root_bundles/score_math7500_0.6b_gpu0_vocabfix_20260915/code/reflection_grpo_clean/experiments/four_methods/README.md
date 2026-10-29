# Qwen3-0.6B 四方法对照实验

2026-09-14 的 MATH7500 / 总生成4096 新配置见 [MATH7500.md](MATH7500.md)，入口为
`run_math7500.sh`。下面保留原 DAPO 实验的配置与历史说明；该训练已按用户要求停止。

此目录准备 GRPO、REFLECT+、PAG、SCoRe 的统一对照。`prepare`、`preflight` 和
`dry-run` 不启动正式训练；正式训练必须显式使用 `train --method ...`。

## 数据和初始化

- 训练来源：`/root/Policy-As-GenVerifier/datasets/dapo17k.parquet`。
- 原始 17,917 条；NFKC/空白规范化后 17,167 个问题；隔离 20 组标签冲突，
  排除 8 个完全重合的验证题及 6 个已人工核对的排版变体，剩 **17,133 题**。
- 验证：指定 MATH500 parquet 的 500 题，及上次固定快照的 AMC23 40 题。
- 原文件不修改。四种方法的数据视图只改变 system prompt；问题、答案、顺序和 ID 一致。
- Base 为 `/mnt/data/models/Qwen3-0.6B`；REFLECT+ 使用此前 phase2 选中的
  `checkpoint-268`，先合并 LoRA，在 CPU 上校验合并前后的 logits，再保存 BF16 模型。
- 结果目录：`/mnt/data/reflection_grpo/artifacts/four_methods_0.6b_20260913`。
  `data/manifest.json` 记录哈希和长度，`data/excluded.jsonl` 记录排除原因，
  `data/batch_schedule.json` 记录 500 步的问题顺序。

## 固定比较条件

配置源为 `config.json`。每组 500 步，每步 64 题，每题 8 条轨迹；PPO mini-batch=64，
ppo_epochs=1。Actor AdamW：lr=1e-6，betas=(0.9,0.999)，weight_decay=0，
grad_clip=1，常数学习率、无 warmup；clip=(0.20,0.26)，entropy=0，
loss 聚合为 seq-mean-token-mean。PAG/SCoRe critic lr=2e-6、weight_decay=0.01。

训练采样 temperature=1、top_p=1、top_k=-1；数据/模型 seed=1234。
四组均关闭 thinking，所有反馈模板也关闭 thinking，禁止生成 `<think>`/`</think>`。
max_prompt_length=2048、max_model_len=12288，超长拒绝执行，不截掉题目或历史。

| 方法 | 起点 | 一条轨迹的固定分段 token 上限 | 保留的方法逻辑 |
|---|---|---|---|
| GRPO | Base | 8192 | 单轮、二元结果奖励、GRPO |
| REFLECT+ | phase2 checkpoint-268 | 4096+4096 | 完整反思标签触发续写、轨迹奖励、解耦优势 |
| PAG | Base | 解题3584+验证1024+再解题3584 | 验证判断、角色归一化、critic |
| SCoRe | Base | 4096+4096 | 前250步 turn-1 KL=0.1，后250步 joint reward shaping |

分段额度不转移；PAG 验证文本和 REFLECT+ 被丢弃尾部都计入真实生成 token。
SCoRe 第2阶段从第250步完整恢复 actor、critic、优化器和 dataloader，累计终点是500步。

所有 prompt 文本集中在 `fourway/contracts.py`，训练和验证共用：前三种 Base 方法首轮
使用普通 boxed prompt；REFLECT+ 要求 `So far I have ... However ...` 的反思格式。
PAG 对明确 correct 停止，否则重解；不使用 gold 决定是否重解。SCoRe 总是请求第二轮。

## 评测和选优

第0、50、100……500步，MATH500和AMC23每题采样4条；temperature=0.7、top_p=0.8、
top_k=20，固定 seed=1234。每个重复请求使用独立随机流，避免复用同一个 seed 导致4条相同。

验证统一读取最终有效解题段，由共享数学判分器评判；不使用复合 reward、PAG 验证文字、
或多轮中最好的答案代替最终答案。记录每段实际输入和输出 tokens、最终提取、判分状态、
pass@4、mean@4、反思率、反思后正确率、截断率和实际生成成本。
数学判分在可终止的 CPU 进程中运行；异常重试后仍失败则报错，不静默记作答错。

最优依据：两个数据集 pass@4 各占50%；并列比较 mean@4 的等权平均，再并列选更早步数。
第0步仅作基线。保存最佳与第500步结果；SCoRe 的最佳 checkpoint 标注所属阶段。
**两个数据集参与选优，结果是验证成绩，不是独立测试成绩。REFLECT+ 还包含额外 SFT 初始化。**

## 操作入口

从仓库根目录执行：

```bash
bash experiments/four_methods/run.sh prepare
bash experiments/four_methods/run.sh dry-run
bash experiments/four_methods/run.sh preflight
bash experiments/four_methods/run.sh preflight --gpu
```

GPU预检使用4题×2条轨迹、2道验证题×4次采样，保留正式上下文和采样长度，执行实际两卡FSDP前后向。
这是小批次集成检查，不能据此保证整个500步过程的显存峰值。SCoRe两个阶段分别探测，
第250步的完整FSDP checkpoint跨阶段恢复尚未实际执行；CPU测试覆盖驱动端保存/加载、
actor与critic加载调用、dataloader进度、选优和保留策略。
AdamW参数更新被禁用；分配优化器状态以检查其内存占用，不产生正式训练checkpoint。
日志中的 `Training Progress 0/1` 属于无参数更新的集成探针，不是正式500步任务。

正式训练入口（**准备过程不会执行以下命令**）：

```bash
bash experiments/four_methods/run.sh train --method grpo
bash experiments/four_methods/run.sh train --method reflect
bash experiments/four_methods/run.sh train --method pag
bash experiments/four_methods/run.sh train --method score
```

每组独占GPU0、1。上面的 `train --method` 只启动指定方法；显式顺序队列见下文，不会同时启动四组。
显式导出 `FOURWAY_CONFIG=/root/reflection_grpo_clean/experiments/four_methods/config.json`
后，原来的两个训练入口和 PAG/SCoRe quick_start 入口也会转到本配置；未设置时保留旧行为。

单独验证已保存 checkpoint（结果单独写入 `evaluations/`，不覆盖训练过程的验证记录）：

```bash
bash experiments/four_methods/run.sh evaluate --method pag --checkpoint /absolute/path/global_step_500
```

使用同一 wrx_env（torch2.6、transformers4.51.1、vLLM0.8.5）；两个仓库在不同进程中导入各自verl。
两卡FSDP、TP=1双采样副本、actor/critic不CPU offload；vLLM memory=0.35、并发64、
batched_tokens=16384、actor/critic动态tokens=16384。准备检查结果与解析后配置位于 `preparation/`。
每次正式训练会检查预检结果，拒绝覆盖已有实验目录。定期保存后仅清理本次实验自身已被替代的
checkpoint，保留best、latest以及SCoRe阶段切换所需的250步checkpoint。

## 已处理的实现问题

- 两个仓库默认配置、验证batch、随机种子及thinking模板不一致：统一解析配置和逐请求随机流。
- PAG验证段截断时可能把padding纳入loss：按实际生成段构造掩码，验证段上限固定1024。
- 中间回答截断后直接接用户消息、Qwen3 EOS配置为列表：补入正确的单个对话结束符，并从loss和生成成本中排除。
- PAG无效判断的训练/验证行为不同：统一仅在明确correct时停止。
- SCoRe训练判分包含反馈文字：按生成掩码提取各轮答案，PAG也统一剥离特殊token。
- 复合训练reward不等于pass@4：使用独立的统一最终答案判分与逐题4样本完整性检查。

验证代码：`tests/test_fourway.py`、`tests/test_fourway_native_rollouts.py`，以及已有
`tests/test_paired_eval.py`、`tests/test_trajectory_context.py`。CPU测试结果、GPU探针日志索引、
原文件备份和修改记录均保存在结果目录的 `preparation/` 下。

## 先运行三个baseline，再运行REFLECT+

用户于2026-09-13授权的正式顺序为 **GRPO → PAG → SCoRe → REFLECT+**。
队列代码为 `fourway/queue.py`，每组使用同一双卡配置；SCoRe内部先完成前250步，再恢复到累计500步。
每个方法正常退出且存在第500步与best checkpoint后才启动下一个方法；任何失败都会停止队列，后续方法保持pending。
配置改变、已有运行目录或重复队列都会拒绝继续，避免混入其他实验。

本次已使用独立后台会话启动，退出终端不影响运行。状态位于结果目录
`queue/status.json`，总日志为 `queue/queue.log`，PID保存在 `queue/pid`；
各组详细训练日志仍在 `runs/<method>/stage1.log`（SCoRe另有 `stage2.log`）。
CPU测试 `tests/test_fourway_queue.py` 覆盖启动顺序以及失败后停止后续方法。
