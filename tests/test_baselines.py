"""Small forward/backward checks for the original numerical baselines."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from model.drae_baseline import DRAE  # noqa: E402
from model.rakge_model import RAKGE  # noqa: E402


LITERALS = np.asarray(
    [[0.1, 0.4, 0.0], [0.3, 0.2, 0.5], [0.5, 0.7, 0.9], [0.0, 0.6, 0.8]],
    dtype=np.float32,
)
LABELS = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)


def parameters(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        init_dim=8,
        att_dim=8,
        head_num=2,
        num_mixture=3,
        drop=0.0,
        gamma=9.0,
        order=0.25,
        scale=0.25,
    )


class BaselineSmokeTest(unittest.TestCase):
    def exercise(self, model: torch.nn.Module, device: torch.device) -> None:
        model = model.to(device)
        heads = torch.tensor([0, 1], device=device)
        relations = torch.tensor([0, 1], device=device)
        labels = LABELS.to(device)
        negative_labels = 1.0 - labels
        active = torch.ones(2, device=device)

        model.train()
        loss = model(None, heads, relations, labels, active, negative_labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        model.eval()
        with torch.no_grad():
            predictions = model(None, heads, relations, labels, active, negative_labels)
        self.assertEqual(tuple(predictions.shape), (2, 4))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_rakge_cpu(self) -> None:
        device = torch.device("cpu")
        self.exercise(RAKGE(4, 2, LITERALS, params=parameters(device)), device)

    @unittest.skipUnless(torch.cuda.is_available(), "Original DRAE baseline requires CUDA")
    def test_drae_cuda(self) -> None:
        device = torch.device("cuda:0")
        self.exercise(DRAE(4, 2, LITERALS, params=parameters(device)), device)


if __name__ == "__main__":
    unittest.main()
