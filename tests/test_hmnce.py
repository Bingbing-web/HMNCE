from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from model.hmnce_model import HMNCE  # noqa: E402
from model.kge_score import TuckER  # noqa: E402


def params(**overrides):
    values = dict(
        init_dim=8,
        att_dim=8,
        head_num=2,
        drop=0.0,
        num_mixture=3,
        gamma=9.0,
        order=0.25,
        scale=0.25,
        top_k=2,
        hmnce_score_func="transe",
        disable_me=False,
        disable_hcl=False,
        disable_hsm=False,
        disable_hsa=False,
        disable_srm=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class HMNCESmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        literals = np.asarray(
            [[0.1, 0.0, 0.4], [0.3, 0.2, 0.0], [0.5, 0.7, 0.9], [0.0, 0.6, 0.8]],
            dtype=np.float32,
        )
        mask = np.asarray(
            [[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32
        )
        self.model = HMNCE(4, 2, literals, params=params(), observed_mask=mask)
        self.heads = torch.tensor([0, 1])
        self.relations = torch.tensor([0, 1])
        self.labels = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        self.negative_labels = 1.0 - self.labels
        self.neg = torch.ones(2)

    def test_training_loss_and_backward(self):
        self.model.train()
        loss = self.model(None, self.heads, self.relations, self.labels, self.neg, self.negative_labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_evaluation_is_deterministic(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(None, self.heads, self.relations, self.labels, self.neg, self.negative_labels)
            second = self.model(None, self.heads, self.relations, self.labels, self.neg, self.negative_labels)
        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(tuple(first.shape), (2, 4))

    def test_tucker_uses_relation_core(self):
        scorer = TuckER(4, 2, 8, 8, dropout=0.0).eval()
        head = torch.randn(2, 8)
        tails = torch.randn(2, 4, 8)
        relation_a = torch.zeros(2, 8)
        relation_b = torch.ones(2, 8)
        with torch.no_grad():
            score_a = scorer(head, relation_a, tails)
            score_b = scorer(head, relation_b, tails)
        self.assertFalse(torch.allclose(score_a, score_b))


if __name__ == "__main__":
    unittest.main()
