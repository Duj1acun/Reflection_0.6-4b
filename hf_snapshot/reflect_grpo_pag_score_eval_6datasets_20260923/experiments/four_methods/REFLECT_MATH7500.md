# REFLECT 与 MATH7500 baseline 的参数对齐

独立配置为 `config.reflect_math7500.json`，入口为 `run_reflect_math7500.sh`。
正在运行的 `config.math7500.json` 和 GRPO → PAG → SCoRe 队列不修改。

REFLECT 训练、MATH500 验证、AMC23 验证均为每轮最多4096 tokens，
最多两轮（含首轮，最多一次反思触发续写），总生成上限8192 tokens。
没有触发反思时不强制生成第二轮；原始题目上限2048，上下文上限16384。

其余参数与baseline共用约定：7496道训练题，500步，64题×8轨迹/步，
actor学习率1e-6，训练温度1.0；验证温度0.7、pass@4、每50步一次，
MATH500共500题、AMC23共40题。禁止thinking，双卡0、1、FSDP、BF16，
vLLM每卡32并发、显存比例0.50、CUDA Graph、预填充预算8192 tokens。

初始化仍是SFT phase2 checkpoint-268的已验证合并模型；反思prompt、奖励及优势算法不变。
数据复制到独立输出目录并校验文件哈希，复用同一采样题目顺序。
合并初始化通过目录符号链接复用已有、经过哈希校验的模型，不重复合并。

结果目录：`/mnt/data/reflection_grpo/artifacts/reflect_math7500_0.6b_20260914`。
此配置尚未启动训练，也未加入当前baseline队列。

只做CPU预检：

```bash
bash experiments/four_methods/run_reflect_math7500.sh preflight
```

GPU空闲且准备运行REFLECT时，先执行其独立GPU预检，再启动：

```bash
bash experiments/four_methods/run_reflect_math7500.sh preflight --gpu
bash experiments/four_methods/run_reflect_math7500.sh train
```

上述两条命令未在本次参数调整中执行。新长度配置的GPU预检不能复用旧配置结果。
