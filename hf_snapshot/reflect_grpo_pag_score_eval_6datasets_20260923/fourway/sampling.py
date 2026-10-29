"""Independent seeded request streams for repeated prompts and resumed steps."""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerVocabMask:
    """Exclude model output rows that have no corresponding tokenizer token.

    Qwen's output head can be padded beyond the tokenizer vocabulary. Those
    rows must never be sampled, especially when output IDs become a later
    turn's prompt. Keep added tokens (including EOS) and handle vocabulary
    holes too. This processor is stateless, pickleable, and has no GPU cache.
    """

    vocab_end: int
    holes: tuple

    @classmethod
    def from_tokenizer(cls, tokenizer):
        ids = sorted(set(tokenizer.get_vocab().values()))
        if not ids or ids[0] < 0:
            raise ValueError("Tokenizer must define nonnegative token IDs")
        holes = []
        start = 0
        for token_id in ids:
            if token_id > start:
                holes.append((start, token_id))
            start = token_id + 1
        return cls(start, tuple(holes))

    def __call__(self, output_token_ids, logits):
        logits[self.vocab_end:] = float("-inf")
        for start, end in self.holes:
            logits[start:end] = float("-inf")
        return logits

def request_params(params, count, seed, rank=0, stage=0, *, tokenizer=None):
    # A single SamplingParams(seed=...) reused for repeated n=1 requests would
    # make all four answers identical. Give every request its own stream.
    values = []
    vocab_mask = TokenizerVocabMask.from_tokenizer(tokenizer) if tokenizer is not None else None
    for index in range(count):
        item = deepcopy(params)
        if vocab_mask is not None:
            # Append after existing custom processors so they cannot restore
            # a padded token's probability. Do not mutate the shared params.
            item.logits_processors = list(item.logits_processors or []) + [vocab_mask]
        item.seed = int((seed + rank * 1000003 + stage * 1000033 + index * 97) % (2**31 - 1))
        values.append(item)
    return values
