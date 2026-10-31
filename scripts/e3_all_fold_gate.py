"""E3's additional five-fold gate; the shared fold-0 gate stays unchanged.

For outer fold f, rotate the existing ridge split so f is held out, f+1
modulo 5 selects the penalty, and the other three folds fit selection models.
Refit on all four non-held folds. Thus outer fold 0 exactly reproduces SO1.
This module accepts predictions only after the shared completion/recipe audit.
"""
import math

import numpy as np

from gbdt_blend import blend_fold
from local_context_chain import paired
from selection_objective_gate import ridge_predict


def compare_all_folds(bank, control, treatment, y, users, minimum_gain=.0003):
    if not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError('minimum gain must be finite and positive')
    if (y.ndim != 1 or bank.ndim != 2 or len(bank) != len(y)
            or any(v.shape != y.shape for v in (control, treatment, users))):
        raise ValueError('prediction shapes differ')
    if not all(np.isfinite(v).all() for v in (bank, control, treatment, y)):
        raise ValueError('nonfinite input')
    folds = blend_fold(users)
    designs = dict(duplicate_control=np.column_stack([bank, control, control]),
                   control_plus_treatment=np.column_stack([bank, control, treatment]))
    predictions = {name: np.empty(len(y)) for name in designs}
    penalties = {name: [] for name in designs}
    for outer in range(5):
        rotated = (folds - outer) % 5
        for name, features in designs.items():
            values, penalty = ridge_predict(features, y, rotated)
            predictions[name][folds == outer] = values
            penalties[name].append(penalty)
    report = paired(y, predictions['duplicate_control'],
                    predictions['control_plus_treatment'], users)
    report.update(selected_penalties=penalties,
                  passes_all_fold_gate=(report['all_folds_improve']
                                        and report['delta_rmse'] <= -minimum_gain),
                  minimum_gain=minimum_gain,
                  scope='Additional E3 nested five-fold P2 check; require the shared fold-0 gate separately. No full-bank promotion claim.')
    return report, predictions
