"""Offline checks; no external requests or credentials required."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_two_stage_sft as pipeline


class PipelineTests(unittest.TestCase):
    def annotation(self):
        return {'cut_point':'I set x=2. Then x+1=4.', 'reflection':{
            'verified_state':['I set x=2.'], 'error_step':'Then x+1=4.',
            'error_reason':'I added incorrectly: 2+1 is 3, not 4.',
            'new_direction':'I should add 1 to 2 correctly.'}}

    def test_exact_prefix(self):
        p=self.annotation()
        self.assertEqual(pipeline.validate_diagnosis(json.dumps(p),p['cut_point']+' More wrong work.'),p)
        with self.assertRaises(ValueError):
            pipeline.validate_diagnosis(json.dumps(p),' '+p['cut_point'])

    def test_perfect_substring_is_not_verdict(self):
        p=self.annotation()
        p['reflection']['error_reason']='I incorrectly assumed it was a perfect square.'
        self.assertIsInstance(pipeline.validate_diagnosis(json.dumps(p),p['cut_point']),dict)
        for text in ('perfect','review','```json\n{}\n```'):
            with self.assertRaises(ValueError):
                pipeline.validate_diagnosis(text,p['cut_point'])

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.validate_diagnosis('{"cut_point":"a","cut_point":"b"}','b')

    def test_empty_states(self):
        p=self.annotation()
        p['reflection']['verified_state']=[]
        self.assertIn('not yet established',pipeline.reflection_text(p))
        self.assertNotIn('previous reasoning steps are correct',pipeline.reflection_text(p))

    def test_mock_two_stages_resume_and_export(self):
        import httpx
        requests=[]
        annotation=self.annotation()
        class FakeClient:
            def __init__(self,**kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            async def post(self,url,json,headers):
                requests.append(json)
                first=json['messages'][1]['content'].startswith('Question: ')
                content=__import__('json').dumps(annotation) if first else r'I compute 2+1=3. Thus \boxed{3}.'
                return httpx.Response(200,json={'model':'DeepSeek-V4-Pro','choices':[{'finish_reason':'stop','message':{'content':content,'reasoning_content':'mock thinking'}}],'usage':{'total_tokens':1}})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            row=dict(uid='test',index='test',student_model='qwen3-0.6b',trajectory_id='test',problem_id='p',problem='What is 2+1?',ground_truth='3',model_response=annotation['cut_point']+' More wrong work.',split='train')
            pipeline.jsonl(out/'candidates.jsonl',[row])
            pipeline.jsonl(out/'perfect.jsonl',[])
            args=argparse.Namespace(output=out,teacher='DeepSeek-V4-Pro',max_tokens=16384,limit=0,workers=2,attempts=3,phase1_prefix='full')
            manifest=dict(diagnosis=pipeline.DIAGNOSIS,continuation=pipeline.CONTINUATION,
                          diagnosis_user_template=pipeline.DIAGNOSIS_USER_TEMPLATE,
                          continuation_user_template=pipeline.CONTINUATION_USER_TEMPLATE)
            with patch.dict('os.environ',{'PARATERA_API_KEY':'offline-placeholder'}), patch('httpx.AsyncClient',FakeClient):
                asyncio.run(pipeline.run(args,manifest))
                asyncio.run(pipeline.run(args,manifest))
            self.assertEqual(len(requests),2)
            second=requests[1]['messages'][1]['content']
            self.assertEqual(second,pipeline.CONTINUATION_USER_TEMPLATE.format(problem=row['problem'],reflection_text=pipeline.reflection_text(annotation).strip()))
            self.assertEqual(requests[0]['messages'][1]['content'],pipeline.DIAGNOSIS_USER_TEMPLATE.format(**row))
            self.assertEqual(requests[0]['messages'][0]['content'],pipeline.DIAGNOSIS)
            self.assertEqual(requests[1]['messages'][0]['content'],pipeline.CONTINUATION)
            self.assertNotIn('More wrong work',str(second))
            with patch('rgrpo.math_score_light.compute_math_score',return_value=1.0):
                pipeline.export(args)
            record=pipeline.read(out/'qwen3-0.6b'/'reasoning_outputs.jsonl')[0]
            self.assertEqual(record['cut_point'],row['model_response'])
            self.assertEqual(record['diagnostic_cut_point'],annotation['cut_point'])
            with patch('rgrpo.math_score_light.compute_math_score',return_value=0.0):
                pipeline.export(args)
            self.assertEqual(pipeline.read(out/'qwen3-0.6b'/'reasoning_outputs.jsonl'),[])
            self.assertEqual(len(pipeline.read(out/'qwen3-0.6b'/'review.jsonl')),1)


if __name__=='__main__':
    unittest.main()
