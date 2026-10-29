"""Independent seeded request streams for repeated prompts and resumed steps."""
from copy import deepcopy

def request_params(params, count, seed, rank=0, stage=0):
    # A single SamplingParams(seed=...) reused for repeated n=1 requests would
    # make all four answers identical. Give every request its own stream.
    values = []
    for index in range(count):
        item = deepcopy(params)
        item.seed = int((seed + rank * 1000003 + stage * 1000033 + index * 97) % (2**31 - 1))
        values.append(item)
    return values
