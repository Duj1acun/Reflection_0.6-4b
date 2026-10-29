# SCORE 无效 token 故障与重启记录

## 2026-09-15 下午状态更新

15:29 实时检查时，SCORE 修复版已完成第 28 步并继续训练；期间两批数据因数学判分异常/超时跳过，随后恢复更新，未再发现无效 token 报错。

REFLECT 于 11:43 在第 62 步计算 log-prob 时退出（最后完成第 61 步）：实际序列 8710 token 超过 8192 的计算小批次预算。已将计算预算修复为 16384，通过长序列与完整批次 GPU 前后向预检，并于 15:44 从 step 50 checkpoint 开始续训。15:56 已验证第 51 步真实更新完成（梯度范数 0.324），继续运行；同期 SCORE 已完成第 31 步。新远端 bundle 为 `/root/reflect_tokenbudget_fix_20260915`；详细配置、验证及实际更新结果见 [REFLECT 修复记录](../reflect_tokenbudget_fix_20260915/DEBUG.md)。`brief_progress.py` 已跟随新 REFLECT bundle。

## 上午 SCORE 修复记录

旧实验在 2026-09-15 04:14:55（服务器时间）第 7 步退出，最后完整更新为第 6 步。错误是 `ValueError: Token id 151869 is out of vocabulary`，不是 CUDA OOM。旧日志与输出保留在 `/root/score_math7500_0.6b_gpu0_20260915`。

模型输出词表大小为 151936，tokenizer 的最大有效 token ID 为 151668，输出头末尾有 267 个 tokenizer 没有定义的位置。原 SCORE 两轮采样没有限制这些位置。第一轮输出中的无效 ID 进入第二轮 prompt 后，被 vLLM 的 `_validate_token_prompt` 拒绝。

修复：为每一轮 SCORE 请求追加无状态的 `TokenizerVocabMask`，将 tokenizer 不存在的位置的 logits 设为负无穷。保留 added tokens、EOS、合法 logits、现有 logits processors、每请求种子、温度及长度设置；支持词表内部空洞。通过 opt-in 的 `fourway` 入口生效，不修改正在运行的 REFLECT 代码和配置。修复已同步到本地源代码和新的远端 SCORE 代码快照。

验证：17 项 CPU 回归测试通过（旧 dataloader 测试的固定 DAPO 批量断言被排除，当前数据对齐另由入口预检检查）。真实 vLLM GPU 测试强制原错误 ID 151869 为最高 logit，稳定复现原错误；追加 mask 后，连续两轮、每轮 4 请求均输出有效 ID。GPU 测试自身第一次将 tuple 与 list 直接比较导致断言失败，修正为序列内容比较后通过，不涉及训练代码修改。

新实验：`/root/score_math7500_0.6b_gpu0_vocabfix_20260915`，GPU 0，独立 Ray 实例与后台 session。入口 `cd /root/reflection_grpo_clean && bash experiments/four_methods/run_math7500.sh train --method score` 的远端软链接已指向新快照；后台已有启动流程，勿重复执行。

旧实验未到第 50 步，没有 checkpoint，新实验从 base model 重新开始。仍为总计 500 步（stage1 250，stage2 续训至 500）、每步 64 题×8、actor LR 1e-6、每轮 4096、两轮共 8192、计算小批次 token 预算 12288、每 50 步保存和验证、val_before_train=False。验证 MATH500 + AMC23，pass@4，T=0.7。完整 GPU 前后向预检与正式启动状态见 `SERVER_STATUS.json`，结束本次检查时的实际步数另存 `LAUNCH_VERIFIED.json`。
