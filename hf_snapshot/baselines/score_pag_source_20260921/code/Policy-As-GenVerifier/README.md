<h1 style="text-align: center;">PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.10406)
[![HomePage](https://img.shields.io/badge/home-000000?style=for-the-badge&logo=homeassistant&logoColor=white)](https://jackory.github.io/pag/)

</div>

## News
- **[2025/06/27]** 🎉 Code released
- **[2025/06/13]** 🎉 [HomePage](https://jackory.github.io/pag/) released

## Installation
This repository is based on verl commit 81a15ed7 (2025/04/03) and requires FSDP with vLLM>=0.8.2. Please refer to [verl installation](https://verl.readthedocs.io/en/latest/start/install.html) for setup instructions. Additionally, install [Math-Verify](https://github.com/huggingface/Math-Verify) as the verifier: `pip install math-verify`

## Quick Start
We provide training scripts for PAG and baseline methods including [SCoRe](https://arxiv.org/pdf/2409.12917) and Direct_MultiTurn:

- PAG: `bash quick_start/qwen1p5b_pag.sh`
- SCoRe: `bash quick_start/qwen1p5b_SCoRe.sh` 
- Direct_MultiTurn: `bash quick_start/qwen1p5b_multiturn.sh`

The evaluation pipeline follows the same procedure as training, please refer to `quick_start/evaluation.sh` for more details.

For debugging purposes, we provide two multi-turn test scripts:
- `tests/multi_turn/run_vllm_spmd_pag_rollout.py`
- `tests/multi_turn/run_vllm_spmd_direct_multiturn.py`

If you encounter CUDA errors during debugging, try commenting out `self.inference_engine.sleep(level=1)` in:
- `verl/workers/rollout/vllm_rollout/vllm_pag_rollout_spmd.py`
- `verl/workers/rollout/vllm_rollout/vllm_multiturn_rollout_spmd.py`

Note that this is only for debugging purposes.



## Citation
If you find this project helpful, please cite:

```bibtex
@article{jiang2025pag,
  title={PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier},
  author={Jiang, Yuhua and Xiong, Yuwen and Yuan, Yufeng and Xin, Chao and Xu, Wenyuan and Yue, Yu and Zhao, Qianchuan and Yan, Lin},
  journal={arXiv preprint arXiv:2506.10406},
  year={2025}
}