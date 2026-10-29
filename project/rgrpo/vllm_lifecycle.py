"""Serialize CUDA work around vLLM virtual-memory lifecycle transitions."""
import torch


def synchronized_transition(operation, *args, **kwargs):
    # empty_cache() alone is not a device-wide quiescence guarantee.
    torch.cuda.synchronize()
    result = operation(*args, **kwargs)
    torch.cuda.synchronize()
    return result
