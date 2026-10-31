#!/usr/bin/env python3
"""E3: fine-tune A2h with MSE + lambda * cross-entropy on its existing five-way head.

A2h already decodes five rating logits and is trained on MSE of their
softmax-weighted expectation (`masked_history.loss_value`, loss='mse'). E3
adds lambda * cross-entropy of the same logits against the observed star,
keeping the expected rating as the prediction.

This reuses SO1's matched fine-tuning harness (`selection_objective.run`)
unchanged: same warm checkpoint, data, seed, masks, batch order, optimizer
restart and the zero-weight auxiliary pass (which keeps dropout RNG use
matched). Only the rating loss is replaced, inside this process. With
--ce-weight 0 the run is SO1's control arm byte for byte
(tests/test_e3_distribution_loss.py), so SO1-control doubles as E3's control.
Neither selection_objective.py nor its dependencies are edited.

The report records `ce_weight` and this script's SHA-256 in `arguments`.
"""
import math
from pathlib import Path

from torch.nn import functional as F

import masked_history as mh
import selection_objective as so


def combined_loss(base, ce_weight):
    def loss(model, raw, ratings):
        if model.config.loss != 'mse' or model.config.head != 'distribution':
            raise ValueError('E3 requires the MSE-trained distribution head')
        value = base(model, raw, ratings)
        return value if ce_weight == 0 else value + ce_weight * F.cross_entropy(raw, ratings)
    return loss


def parser():
    p = so.parser()
    p.add_argument('--ce-weight', type=float, required=True, help='lambda on cross-entropy (0 = SO1 control)')
    return p


def run(a):
    if a.weight != 0:
        raise ValueError('E3 keeps the SO1 auxiliary weight at 0')
    if not math.isfinite(a.ce_weight) or a.ce_weight < 0:
        raise ValueError('invalid --ce-weight')
    a.e3_script_sha256 = mh.digest(Path(__file__).resolve())
    base = mh.loss_value
    so.mh.loss_value = combined_loss(base, a.ce_weight)
    try:
        return so.run(a)
    finally:
        so.mh.loss_value = base


if __name__ == '__main__':
    run(parser().parse_args())
