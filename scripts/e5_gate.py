#!/usr/bin/env python3
"""E5 fixed P2 gate (NEXT_EXPLORATIONS.md): SO1's bank11 comparison, unchanged.

Primary: bank11 + Gaussian control + duplicate control vs bank11 + Gaussian
control + ordinal. Pass if the primary delta_rmse <= -0.0003. P2 only; the
original probe is not read.
"""
import json
import sys
from pathlib import Path

import numpy as np

from gbdt_blend import load
from selection_objective_gate import compare
from train_holdout import digest

PLAN = 'experiments/SO1-plan.json'
ARMS = {'control': 'artifacts/e5/E5-p2-r64-gaussian.f32', 'treatment': 'artifacts/e5/E5-p2-r64-ordinal.f32'}


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for name in ('gaussian', 'ordinal'):
        rec = json.loads(Path(f'experiments/E5-p2-r64-{name}.json').read_text())
        if rec['returncode'] != 0:
            raise ValueError(name + ' arm did not complete')
    plan = json.loads(Path(PLAN).read_text())
    d = load(plan['data'])
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    bank = np.column_stack([np.fromfile(c, dtype='<f4').astype(float) for c in plan['bank']])
    pred = {k: np.fromfile(v, dtype='<f4').astype(float) for k, v in ARMS.items()}
    result, _ = compare(bank, pred['control'], pred['treatment'], y, users)
    standalone = {k: float(np.sqrt(np.mean((np.clip(v, 1, 5) - y) ** 2))) for k, v in pred.items()}
    result.update(standalone_p2_rmse=standalone, arms={k: dict(path=v, sha256=digest(v)) for k, v in ARMS.items()},
                  plan_sha256=digest(PLAN), passes_gate=result['primary']['delta_rmse'] <= -0.0003,
                  metric='E5 fixed P2 gate (SO1 bank11 ridge). P2 development evidence only.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(primary=result['primary']['delta_rmse'], replacement=result['replacement_diagnostic']['delta_rmse'],
                          standalone=standalone, passes_gate=result['passes_gate'],
                          ci_mse=result['primary']['cluster_normal_95_mse_interval']), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
