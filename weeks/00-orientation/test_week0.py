"""Offline checks: a tiny randomly initialized GPT-2, not downloaded weights."""

import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from week0 import generate_tokens, sample_next_token


class Week0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=16, n_layer=1, n_head=2,
        )).eval()
        cls.prompt = torch.tensor([[1, 2, 3]])

    def test_greedy_uses_largest_logit(self):
        token = sample_next_token(torch.tensor([[0.0, 3.0, 1.0]]), 0)
        self.assertEqual(token.item(), 1)

    def test_seed_repeats_sampling(self):
        logits = torch.zeros(1, 16)
        first = sample_next_token(logits, 0.8, torch.Generator().manual_seed(42))
        second = sample_next_token(logits, 0.8, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(first, second))

    def test_generation_preserves_prompt_and_adds_requested_tokens(self):
        output = generate_tokens(self.model, self.prompt, 4, 0)
        self.assertEqual(tuple(output.shape), (1, 7))
        self.assertTrue(torch.equal(output[:, :3], self.prompt))
        self.assertEqual(tuple(self.prompt.shape), (1, 3))

    def test_zero_tokens_returns_prompt(self):
        self.assertTrue(torch.equal(
            generate_tokens(self.model, self.prompt, 0, 0), self.prompt,
        ))

    def test_stops_at_eos(self):
        with torch.inference_mode():
            eos = self.model(self.prompt).logits[:, -1, :].argmax().item()
        output = generate_tokens(self.model, self.prompt, 4, 0, eos_token_id=eos)
        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertEqual(output[0, -1].item(), eos)

    def test_rejects_invalid_temperatures(self):
        for temperature in (-1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                sample_next_token(torch.zeros(1, 16), temperature)

    def test_rejects_invalid_lengths_and_prompts(self):
        cases = [
            (self.prompt, -1),
            (self.prompt, 14),
            (torch.empty((1, 0), dtype=torch.long), 1),
            (self.prompt.repeat(2, 1), 1),
        ]
        for prompt, count in cases:
            with self.subTest(shape=prompt.shape, count=count):
                with self.assertRaises(ValueError):
                    generate_tokens(self.model, prompt, count, 0)


if __name__ == "__main__":
    unittest.main()
