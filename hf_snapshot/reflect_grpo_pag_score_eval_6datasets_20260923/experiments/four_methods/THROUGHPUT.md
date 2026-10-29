# MATH7500 吞吐优化

**历史测速记录**：下述4.06倍吞吐和GPU预检对应原4096总预算、8192上下文。
后续长度已调整为GRPO4096、PAG4096+2048+4096、SCoRe4096+4096，上下文16384。
原测量不能作为新长度配置的显存/吞吐保证；新配置尚需GPU预检。当前设置见MATH7500.md。

配置：`config.math7500.json`。当前执行范围为 GRPO、PAG、SCoRe 三个 baseline，使用 GPU 0、1 两张 A800 40GB；REFLECT+ 暂不执行。
只调整运行效率参数，保留每个方法各500步（SCoRe两阶段合计）、每步64题×8条轨迹、
4096总生成预算、学习率、随机种子、训练数据，以及每50步的MATH500/AMC23 pass@4验证。
原配置备份在新实验结果目录的 `preparation/throughput/config.before.json`。

| 参数 | 原配置 | 新配置 |
|---|---:|---:|
| enforce_eager | true | false（CUDA Graph） |
| vLLM gpu_memory_utilization | 0.35 | 0.50 |
| max_num_seqs，每卡同时在途序列 | 16 | 32 |
| max_num_batched_tokens，预填充调度预算 | 16384 | 8192 |
| 验证每批题数 | 32 | 128 |

推理预填充预算与训练动态组批预算不同；actor/critic训练的tokens_per_gpu仍为16384。
验证从17批变为5批；每题4次采样和540题覆盖不变。验证批次划分改变了批次派生随机流，
新实验所有方法共用新配置；不把它与旧DAPO记录当作逐样本完全相同的随机试验。

## 实测

原配置在GPU0、新配置在GPU1，同时使用相同Base模型、同一组8道MATH训练题×8个独立seed，
temperature=1、top_p=1、生成上限4096。统计不含引擎加载和预热。

| 正常输出 | 原配置 | 新配置 |
|---|---:|---:|
| 回答数 | 64 | 64 |
| 实际生成tokens | 39402 | 38129 |
| 采样用时 | 125.13秒 | 29.81秒 |
| 吞吐 | 314.88 tokens/s | 1279.09 tokens/s |
| KV preemption/recompute | 0 | 0 |

按tokens/s约4.06倍。回答长度略有不同，因此不只比较墙钟时间。
新配置另做64条强制生成4096 tokens的压力测试：262144 tokens、180.24秒、1454.44 tokens/s，
0次KV重算，PyTorch峰值分配19.60 GiB（独立推理，不包含FSDP训练进程）。
原配置的正常输出测量完成后，停止其冗余长输出阶段；没有报告旧配置的完整长输出测试结果。
CUDA Graph捕获额外约0.09 GiB，独立引擎sleep/wake前后的短greedy输出一致。

这不是完整训练加速倍数，也不是整个数据集的准确率/耗时估计。测试只有8道题，没有交换GPU复测。
前向重算、反向传播、PAG/SCoRe的critic、权重同步、评分和周期性验证仍需时间。
不以此承诺整组训练时长；正式运行前几步后才有可靠的端到端速度估计。

测速脚本第一次在写完全部结果后，解释器退出阶段出现段错误。添加显式NCCL/显存清理后，
短采样复测仍在结束清理时以139退出；该问题尚未解决。采样、sleep/wake一致性检查和结果写入已完成，
进程退出后两张GPU均恢复空闲，但不能将独立测速脚本报告为完整通过。
`cleanup_smoke.json`记录4条短回答而非完整吞吐复测（首次写入的scope文本沿用了64条模板，以stats为准）。
正式Ray训练路径的全部双卡探针均正常退出，两条路径分别记录，相关日志不删除。

## 验证与启动状态

`preparation/throughput/`保存before/after.json、原始日志和CPU检查；
正式框架的双卡、无参数更新探针结果由 `preparation/gpu_preflight.json` 按配置哈希记录，
详细日志在 `probes/<method>/stageN/stageN.log`。探针覆盖GRPO、REFLECT+、PAG和SCoRe两个阶段。
它们是小批次集成检查，不保证任意长输出分布下的训练显存峰值；长输出压力检查另行记录如上。
运行探针不会启动正式500步训练。

当前配置的双卡预检全部通过，三个 baseline 的结果如下。包含原生训练框架的前向、反向、
优化器状态分配，但禁用 AdamW.step，不改变模型参数。

| 方法 | 检查结果 | 每卡峰值分配显存（取两卡最大值） |
|---|---|---:|
| GRPO | 通过 | 25.1 GiB |
| PAG | 通过，含 actor/critic | 30.0 GiB |
| SCoRe 第一阶段 | 通过，含参考策略和 actor/critic | 29.6 GiB |
| SCoRe 第二阶段 | 通过，含 actor/critic | 29.6 GiB |

PAG/SCoRe 的 PyTorch 保留显存可达到约37 GiB；不能把已分配显存与40 GiB的差额全部再给vLLM。
这些探针每步仅4题×2回答，尚未完成64题×8回答的整步吞吐测量，不能据此给出500步准确ETA。
正式运行时三个方法依次各占用两张卡；不是三个训练同时共享两张卡。

vLLM关于KV预算和chunked prefill的调参依据见[官方0.8.5说明](https://docs.vllm.ai/en/v0.8.5/performance/optimization.html)。
