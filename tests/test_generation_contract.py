"""Regression tests for output boundaries and context budgets (no ML imports)."""
import ast
from pathlib import Path
import unittest
from data.generation_contract import decode_new_tokens, validate_generation_budget

class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=False):
        return ''.join(tokens)

class GenerationContractTests(unittest.TestCase):
    def test_embedded_marker_does_not_leak_prompt(self):
        prompt='### Instruction:\nExplain ### Response:\n literally\n\n### Response:\n'
        result=decode_new_tokens(FakeTokenizer(), list(prompt+'answer'), len(prompt))
        self.assertEqual(result,'answer')
        self.assertNotEqual((prompt+'answer').split('### Response:\n',1)[-1].strip(),'answer')
    def test_completion_marker_is_preserved(self):
        prompt='input';answer='### Response:\nquoted output'
        self.assertEqual(decode_new_tokens(FakeTokenizer(),list(prompt+answer),len(prompt)),answer)
    def test_empty_completion(self):
        self.assertEqual(decode_new_tokens(FakeTokenizer(),list('input'),5),'')
    def test_invalid_boundary(self):
        for n in (-1,True,8):
            with self.subTest(n=n),self.assertRaises(ValueError):
                decode_new_tokens(FakeTokenizer(),list('short'),n)
    def test_exact_budget(self):
        validate_generation_budget(3936,160,4096)
    def test_excess_budget(self):
        with self.assertRaisesRegex(ValueError,'Context budget exceeded'):
            validate_generation_budget(3937,160,4096)
    def test_invalid_budget_values(self):
        for values in ((0,1,4096),(10,0,4096),(10,1,0),(True,1,4096),(10,-1,4096),(10,1,None),(10,1,4096.0)):
            with self.subTest(values=values),self.assertRaises(ValueError):
                validate_generation_budget(*values)
    def test_all_three_callers_use_contract(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('inference/inference.py','inference/infer_allam_helpers.py','evaluation/allam_eval/generation.py'):
            tree=ast.parse((root/name).read_text(encoding='utf-8'))
            called=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            with self.subTest(path=name):
                self.assertIn('decode_new_tokens',called)
                self.assertIn('validate_generation_budget',called)

if __name__=='__main__':unittest.main(verbosity=2)
