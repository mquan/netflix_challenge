"""Compaction preserves archived N1 token construction and displacement counts."""
import dataclasses
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from choice_context_audit import archived_module, compact
from test_selection_objective import fixture


class ContextAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'experiments/N1-input-comparison-audit.json'
        cls.module = archived_module(json.loads(path.read_text()))
        torch.set_num_threads(1)

    def test_sparse_compaction_preserves_every_token_tensor(self):
        module, d = self.module, fixture()
        users = np.array([1, 4])
        reduced = compact(d, users, module.gather)
        c = module.Config(device='cpu', seq_len=5, pad_width=4, choice_tokens=1)
        outputs = []
        for data in (d, reduced):
            rows, index, _ = module.gather(data['train_offsets'], users)
            outputs.append(module.sequence_inputs(data, users, rows, index, np.ones(len(index), bool),
                np.arange(2), np.array([7, 13]), np.full(2, 5), c, predicting=True))
        self.assertEqual(outputs[0][1], outputs[1][1])
        for name in outputs[0][0]:
            self.assertTrue(torch.equal(outputs[0][0][name], outputs[1][0][name]), name)

    def test_known_displacement_and_query_exclusion_at_two_budgets(self):
        module, d = self.module, fixture()
        users = np.array([1])
        rows, index, _ = module.gather(d['train_offsets'], users)
        for budget, expected in ((5, (3, 2)), (16, (5, 5))):
            for enabled in (0, 1):
                c = module.Config(device='cpu', seq_len=budget, pad_width=4, choice_tokens=enabled)
                tokens, _ = module.sequence_inputs(d, users, rows, index, np.ones(len(index), bool),
                    np.array([0]), np.array([7]), np.array([5]), c, predicting=True)
                stars, items = tokens['seq_stars'].numpy(), tokens['seq_items'].numpy()
                self.assertEqual(int((stars < module.MASK).sum()), expected[enabled])
                self.assertEqual(set(items[stars == module.CHOICE]), {8} if enabled else set())
                self.assertEqual(set(items[stars == module.MASK]), {7})


if __name__ == '__main__':
    unittest.main()
