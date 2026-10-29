"""Reproduce the real vLLM multi-turn failure, then prove the mask prevents it."""
from pathlib import Path
import json
import sys

b = Path(__file__).resolve().parent
c = json.loads((b / 'config.json').read_text())
sys.path.insert(0, c['reflection_repo'])

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from fourway.sampling import request_params


class ForcePaddedToken:
    def __init__(self, bad, fallback):
        self.bad, self.fallback = bad, fallback

    def __call__(self, output_ids, logits):
        logits.fill_(float('-inf'))
        logits[self.bad] = 1000.
        logits[self.fallback] = 0.
        return logits


tok = AutoTokenizer.from_pretrained(c['base_model'])
valid = set(tok.get_vocab().values())
bad = 151869
assert bad not in valid
fallback = tok.encode('2', add_special_tokens=False)[0]
prompt = tok.apply_chat_template([{'role': 'user', 'content': 'What is 1+1?'}],
                                tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
llm = LLM(model=c['base_model'], dtype='bfloat16', enforce_eager=True,
          tensor_parallel_size=1, gpu_memory_utilization=.25, max_model_len=512,
          max_num_batched_tokens=1024, max_num_seqs=4, enable_chunked_prefill=True,
          seed=c['seed'], disable_log_stats=True)
params = SamplingParams(n=1, temperature=0., max_tokens=1, detokenize=False,
                        bad_words=['<think>', '</think>'],
                        logits_processors=[ForcePaddedToken(bad, fallback)])
raw = llm.generate(prompt_token_ids=[prompt], sampling_params=params, use_tqdm=False)
print('UNGUARDED_OUTPUT', repr(raw[0].outputs[0].token_ids), flush=True)
assert list(raw[0].outputs[0].token_ids) == [bad]
try:
    llm.generate(prompt_token_ids=[prompt + [bad]], sampling_params=params, use_tqdm=False)
except ValueError as error:
    assert f'Token id {bad} is out of vocabulary' in str(error)
    print('REPRODUCED_ORIGINAL_VLLM_FAILURE', str(error), flush=True)
else:
    raise AssertionError('Expected the unguarded second-turn prompt to fail')

prompts = [prompt.copy() for _ in range(4)]
for stage in range(2):
    guarded = request_params(params, 4, c['seed'], stage=stage, tokenizer=tok)
    results = llm.generate(prompt_token_ids=prompts, sampling_params=guarded, use_tqdm=False)
    for i, result in enumerate(results):
        ids = list(result.outputs[0].token_ids)
        assert ids == [fallback] and set(ids) <= valid
        prompts[i] += ids
print('GPU_VOCAB_REGRESSION_PASSED: forced invalid ID excluded on both turns, 4 requests each', flush=True)
(b / 'GPU_VOCAB_REGRESSION.json').write_text(json.dumps({
    'status': 'passed', 'original_error_reproduced': True, 'bad_token': bad,
    'tokenizer_max_id': max(valid), 'guarded_turns': 2, 'requests_per_turn': 4,
    'optimizer_updates': 0}, indent=2) + '\n')
