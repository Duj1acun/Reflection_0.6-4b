# Minimal shim: the original vendored verl third_party/vllm patch package is
# not present in this repo copy. We only need get_version() for version
# comparisons against the standard pip-installed vllm.
import importlib.metadata


def _patch_transformers_compat() -> None:
    """Bridge a tokenizer attribute removed after vLLM 0.8.5 was released."""
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda tokenizer: tokenizer.all_special_tokens
        )


_patch_transformers_compat()

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state


vllm_version = importlib.metadata.version("vllm")


def get_version(pkg: str) -> str:
    return importlib.metadata.version(pkg)
