"""Regression coverage for padded model output IDs entering a later turn."""
from copy import deepcopy
import pickle
from types import SimpleNamespace

import pytest
import torch

from fourway.sampling import TokenizerVocabMask, request_params


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_invalid_rows_have_zero_probability_without_changing_valid_logits(dtype):
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 2, '<eos>': 4})
    mask = TokenizerVocabMask.from_tokenizer(tokenizer)
    logits = torch.tensor([1., 999., 2., 999., 3., 999., 999.], dtype=dtype)
    original = logits.clone()
    masked = pickle.loads(pickle.dumps(mask))([], logits)
    assert torch.equal(masked[[0, 2, 4]], original[[0, 2, 4]])
    assert torch.isneginf(masked[[1, 3, 5, 6]]).all()
    assert torch.softmax(masked.float(), dim=0)[[1, 3, 5, 6]].sum() == 0
    assert masked.argmax().item() == 4  # Added EOS remains usable.


def test_actual_qwen_token_151869_is_excluded():
    # Include both base and added tokens, exactly as tokenizer.get_vocab does.
    tokenizer = SimpleNamespace(get_vocab=lambda: {str(i): i for i in range(151669)})
    logits = torch.zeros(151936)
    logits[151869] = 1e6
    logits[151645] = 1
    assert logits.argmax().item() == 151869
    masked = TokenizerVocabMask.from_tokenizer(tokenizer)([], logits)
    assert masked.argmax().item() == 151645
    assert torch.isneginf(masked[151669:]).all()


def test_request_cloning_preserves_streams_settings_and_existing_processors():
    from vllm import SamplingParams
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 0, '<eos>': 2})
    original = SamplingParams(n=1, temperature=.7, max_tokens=4096,
                              bad_words=['<think>', '</think>'],
                              logits_processors=[TokenizerVocabMask(8, ())])
    baseline = request_params(original, 4, 1234, stage=1)
    guarded = request_params(original, 4, 1234, stage=1, tokenizer=tokenizer)
    assert [p.seed for p in guarded] == [p.seed for p in baseline]
    assert len({p.seed for p in guarded}) == 4
    assert len(original.logits_processors) == 1
    for p in guarded:
        assert p.n == 1 and p.temperature == .7 and p.max_tokens == 4096
        assert p.bad_words == original.bad_words
        assert len(p.logits_processors) == 2
        assert p.logits_processors[-1] == TokenizerVocabMask(3, ((1, 2),))
        assert deepcopy(p).logits_processors == p.logits_processors
