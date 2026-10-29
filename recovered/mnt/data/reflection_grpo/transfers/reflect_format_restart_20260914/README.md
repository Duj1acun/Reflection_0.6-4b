# REFLECT restart: SFT prompt, no format reward gate

User requested both experiments restart from the same SFT phase2 checkpoint268 after discovering fixed-phrase reward gating mismatched the SFT targets. Final instruction: use the exact SFT system prompt and disable format gating entirely.

```
You are a math reasoning assistant. When you make an error during reasoning, use <reflection>...</reflection> to reflect on the mistake, then continue reasoning based on the reflection.
```

The exact SFT prompt has no additional boxed-answer sentence. Existing math answer extraction and correctness scoring are preserved. Reflection tags still control rollout continuation. TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT=0; other reward terms and training parameters are unchanged. The legacy format/valid metrics become pass-through indicators and must not be interpreted as independent format compliance measurements.

- GPU0: /root/reflect_main_sftprompt_20260914/config.json
- GPU1: /root/reflect_branch_sftprompt_20260914/config.json
- New outputs have suffix `_sftprompt_nogate`.
- Previous run outputs and code are preserved. Both old process sessions were stopped and GPUs released before restarting.
- Both restart launchers run in detached sessions; preparation, regression tests, and mandatory GPU no-update preflight must pass before formal training starts. A failure stops the launcher and is recorded in SERVER_STATUS.json.
- Same 7,496 training questions, 540 validation questions, seed1234, 500updates, globalbatch64×8, one optimizer update per step, learning rate1e-6, 3rounds×4096, no thinking.

Read remote SERVER_STATUS.json, restart.log, regression_tests.log, gpu_preflight.log, and runs/reflect/stage1.log for actual status. Launcher creation is not proof of a completed optimizer update.

Startup verified: both jobs completed at least one real optimizer update and entered the next rollout. GPU0 first step160.683s; GPU1 first step239.768s with815valid leaves,512roots,one optimizer update. Positive gradient norms and reflection rewards; zero format-gate zero rate. Initial validation outputs match byte-for-byte across both jobs. See startup_verified.json for timestamp and evidence.
