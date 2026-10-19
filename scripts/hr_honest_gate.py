#!/usr/bin/env python3
"""Cheap honest gate for a new P2 column (NEW_BASE_MODELS.md step 2).

Predicts P2 from a choice_sequence checkpoint trained on T-P2 (optional), then
fits ridge stacks on P2 user folds 2-4, chooses the penalty on fold 1 and
reports fold 0: the ten honest rehearsal columns other than A2h, plus
(a) A2h-select, (b) the candidate, (c) both. Only P2 ratings are read; probe
labels are never touched.
"""
import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

from gbdt_blend import blend_fold, load
from hr_path_residual_test import A2H, BANK

PENALTIES = [1, 10, 100, 1000]


def predict_p2(checkpoint, data, out):
    import torch
    import choice_sequence as cs
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    c = dataclasses.replace(cs.Config(**saved['config']), device='mps')
    model = cs.SequenceModel(saved['items'], c, saved['weeks'])
    model.load_state_dict(saved['state'])
    model.to(c.device)
    pred = cs.predict(model, data)
    pred.astype('<f4').tofile(out)
    return pred


def ridge_eval(x, y, folds):
    x = np.column_stack([np.ones(len(y)), x])
    fit, sel, held = folds >= 2, folds == 1, folds == 0
    best = None
    for lam in PENALTIES:
        w = np.linalg.solve(x[fit].T @ x[fit] + lam * np.eye(x.shape[1]), x[fit].T @ y[fit])
        s = float(np.sqrt(np.mean((np.clip(x[sel] @ w, 1, 5) - y[sel]) ** 2)))
        if best is None or s < best[0]:
            best = (s, lam)
    tr = folds >= 1
    w = np.linalg.solve(x[tr].T @ x[tr] + best[1] * np.eye(x.shape[1]), x[tr].T @ y[tr])
    e = np.clip(x[held] @ w, 1, 5) - y[held]
    return float(np.sqrt(np.mean(e ** 2))), best[1], e ** 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-data', required=True)
    p.add_argument('--column', required=True, help='candidate P2 column (written first if --checkpoint)')
    p.add_argument('--checkpoint')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    d = load(a.selection_data)
    if a.checkpoint:
        if pathlib.Path(a.column).exists():
            raise FileExistsError(a.column)
        predict_p2(a.checkpoint, d, a.column)
    y = d['probe']['rating'].astype(np.float64)
    folds = blend_fold(d['probe']['user'])
    cols = lambda paths: np.column_stack([np.fromfile(c, dtype='<f4').astype(np.float64) for c in paths])
    rest = cols([c for c in BANK if c != A2H])
    a2h, cand = cols([A2H]), cols([a.column])
    result = dict(candidate=a.column, alone=float(np.sqrt(np.mean((np.clip(cand[:, 0], 1, 5) - y) ** 2))),
                  a2h_alone=float(np.sqrt(np.mean((np.clip(a2h[:, 0], 1, 5) - y) ** 2))))
    sq = {}
    for name, x in [('bank10', rest), ('bank10+A2h', np.column_stack([rest, a2h])),
                    ('bank10+candidate', np.column_stack([rest, cand])),
                    ('bank10+A2h+candidate', np.column_stack([rest, a2h, cand]))]:
        r, lam, sq[name] = ridge_eval(x, y, folds)
        result[name] = dict(rmse=r, penalty=lam)
    users = d['probe']['user'][folds == 0]
    diff = sq['bank10+A2h+candidate'] - sq['bank10+A2h']
    _, inv = np.unique(users, return_inverse=True)
    per = np.bincount(inv, diff)
    se = float(np.sqrt(len(per)) * per.std() / len(diff))
    result['candidate_beyond_A2h'] = dict(delta_rmse=result['bank10+A2h+candidate']['rmse'] - result['bank10+A2h']['rmse'],
                                          delta_mse=float(diff.mean()), user_cluster_se_mse=se)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
