import unittest
from types import SimpleNamespace
import numpy as np
from rgrpo.sft_training import causal_token_metrics, encode_supervised, validate_split_groups, EvaluateLastStep


class BoundaryTokenizer:
    eos_token = '<eos>'
    eos_token_id = 9
    pad_token_id = 9

    def __call__(self, text, **kwargs):
        # Token 2 straddles the boundary between prefix 'ab' and target 'cd'.
        assert text == 'abcd<eos>'
        return {'input_ids':[1,2,3,9], 'offset_mapping':[(0,1),(1,3),(3,4),(4,9)]}


class SFTTrainingTests(unittest.TestCase):
    def test_last_step_is_evaluated_and_saved_off_interval(self):
        callback=EvaluateLastStep()
        control=SimpleNamespace(should_evaluate=False,should_save=False)
        callback.on_step_end(None,SimpleNamespace(global_step=473,max_steps=474),control)
        self.assertFalse(control.should_evaluate)
        callback.on_step_end(None,SimpleNamespace(global_step=474,max_steps=474),control)
        self.assertTrue(control.should_evaluate)
        self.assertTrue(control.should_save)

    def test_accuracy_uses_next_token_and_ignores_prompt(self):
        labels=np.array([[-100,-100,4,5,9,-100]])
        predictions=np.array([[0,4,5,9,0,0]])
        self.assertEqual(causal_token_metrics((predictions,labels))['token_accuracy'],1.0)
        predictions[0,2]=0
        self.assertAlmostEqual(causal_token_metrics((predictions,labels))['token_accuracy'],2/3)

    def test_boundary_target_and_eos_are_supervised_padding_is_not(self):
        sample=encode_supervised(BoundaryTokenizer(),'ab','cd',6)
        self.assertEqual(sample['labels'].tolist(),[-100,2,3,9,-100,-100])
        self.assertEqual(sample['attention_mask'].tolist(),[1,1,1,1,0,0])

    def test_overlength_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError,'exceeds MAX_LEN'):
            encode_supervised(BoundaryTokenizer(),'ab','cd',3)

    def test_same_question_different_trajectories_cannot_cross_split(self):
        with self.assertRaisesRegex(ValueError,'Same question'):
            validate_split_groups([{'problem':'Find  x.'}],[{'problem':'Find x.'}])


if __name__=='__main__':
    unittest.main()
