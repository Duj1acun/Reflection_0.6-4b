> 当前主实验已改为 GPU 0 单卡、最多三轮；请以 README_SINGLE_GPU.md、config.json 和 SERVER_STATUS.json 为准。以下双卡说明保留为迁移历史。

# REFLECT 服务器迁移包

本包只用于REFLECT，未加入源服务器正在运行的baseline队列。
训练参数：Qwen3-0.6B SFT phase2 checkpoint-268合并初始化，MATH7500清洗后7496题，
500步，64题×8轨迹，双GPU；每轮4096 tokens、最多两轮、总8192；上下文16384。
MATH500/AMC23每50步验证pass@4，温度0.7，禁止thinking。

## 包含内容

- `models/Qwen3-0.6B`：Base权重、配置和tokenizer，用于初始化来源核验。
- `models/reflect_merged`：真正用于REFLECT训练的已合并SFT权重，保留原合并校验记录。
- `models/adapter`：来源LoRA权重和配置，不包含不需要的SFT优化器状态。
- `assets/prepared_data`：同一批7496道训练题、540道验证题，以及固定题目顺序。
- `assets/sources`：原始MATH7500、MATH500和AMC23输入快照。
- `code/reflection_grpo_clean`：训练、采样、反思奖励、判分、验证及checkpoint代码。
- `code/Policy-As-GenVerifier`：统一配置CPU预检所需的另一个verl配置树；不启动PAG/SCoRe。
- `environment/packages.json`：源Python环境包版本清单，不等同于可跨机器直接安装的完整环境。
- `SHA256SUMS`：迁移负载校验清单。源机器的输出日志、缓存、.git、.env和SSH认证材料不在包内。

## 目标机器配置

当前训练器固定两张GPU；源机为2×A800 40GB、约64GiB主机内存。
目标GPU型号/显存不同时必须重新GPU预检，不能直接承诺同样吞吐或显存峰值。
建议为环境、训练checkpoint和日志预留至少50GB可用空间。
源环境Python3.10、Torch2.6.0+cu124、vLLM0.8.5、Transformers4.51.1、FlashAttention2.7.4.post1。
依赖安装需根据目标驱动、系统及是否联网决定；本包尚不含Conda环境或离线wheel包。

传输完成后，先在包目录校验文件，再根据目标环境重新生成绝对路径：

```bash
sha256sum -c SHA256SUMS
python3 configure.py --python /path/to/conda/env/bin/python --gpus 0,1
python3 run_reflect.py preflight
```

`configure.py`会重写包内adapter配置和运行数据manifest中的路径；请在配置前做传输校验。
它不会更改模型权重或启动训练。Ray临时路径、缓存和锁均可在新配置中指定，
无需复现源机器的`/root`、`/mnt/data`或`/home/vipuser`目录结构。
如包目录移动，重新运行configure.py；已经开始训练的输出目录不能被重新配置。

目标GPU空闲并且准备正式启动时，再运行：

```bash
python3 run_reflect.py preflight --gpu
python3 run_reflect.py train
```

启动命令默认前台运行；如需脱离终端，应在目标服务器使用独立会话启动并保存PID/日志。
不要直接调用代码目录中沿用源机器路径的历史shell入口；本迁移包使用根目录run_reflect.py。

## 状态

2026-09-14：已上传至目标服务器 `/root/reflect_math7500_0.6b_20260914`。
代码已重新同步，最新代码以 `CODE_SHA256SUMS` 为准；旧 `SHA256SUMS` 仅记录首次传输快照。
模型、数据首次传输全部通过 SHA256 校验；配置后 adapter 中的 Base 路径会发生预期变化。

远端独立环境：`/data/reflection_grpo/envs/reflect_math7500/bin/python`。
基于目标机器原环境克隆并调整训练依赖，原 `/root/.conda/envs/wrx_env` 保留。
新判分函数、异常重试和整批跳过已同步，连续 3 批判分失败停止；失败验证不记为零分。
CPU 预检和两张 A100 80GB 的 REFLECT 无更新预检通过；正式训练尚未启动。
预检使用 4 题 × 2 轨迹和小验证集，覆盖实际反思续写、判分、双卡反向传播及优化器状态分配；
不等同于完整 64 题 × 8 轨迹的训练性能或最终效果验证。

## 本服务器启动

配置已经完成，无需再次运行 configure.py。使用迁移包入口：

```bash
cd /root/reflect_math7500_0.6b_20260914
nohup python3 -u run_reflect.py train > results/launcher.log 2>&1 < /dev/null &
echo $! > results/launcher.pid
```

上面的命令需要手动执行；本次迁移没有启动正式训练。
训练使用 `config.json` 指定的独立环境及 GPU 0,1。
运行日志：`results/runs/reflect/stage1.log`，启动日志：`results/launcher.log`。
`results` 链接到 `/data/reflection_grpo/experiments/reflect_math7500_0.6b_20260914`。
CPU 预检：`results/preparation/cpu_preflight.json`；
双卡预检：`results/preparation/gpu_preflight.json`；
详细预检日志：`results/probes/reflect/stage1/stage1.log`。
更新代码前的备份：`/data/reflection_grpo/setup_logs/code_before_resync_20260914.tar.gz`。

