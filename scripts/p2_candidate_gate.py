#!/usr/bin/env python3
"""Score any completed P2 control/candidate pair with the existing SO1 gate.

Training provenance is supplied and validated by each experiment's owner.
This command binds the scoring inputs and checks completed receipts, but does
not replace a trainer-specific recipe audit or the full-bank promotion gate.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from gbdt_blend import load
from selection_objective_gate import compare
from train_holdout import digest


SOURCES = ('p2_candidate_gate.py','selection_objective_gate.py','gbdt_blend.py',
           'local_context_chain.py','train_holdout.py')


def score(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    if (not math.isfinite(plan['minimum_gain']) or plan['minimum_gain'] <= 0
            or plan['protocol'] != 'SO1-P2-fold0-matched-pair-v1'):
        raise ValueError('invalid or unsupported fixed gate')
    if not plan['bank'] or set(plan['arms']) != {'control','treatment'}:
        raise ValueError('bank and one matched pair required')
    if not plan['protocol_reference']:
        raise ValueError('pre-registered experiment protocol reference required')
    required = [plan['data'],plan['protocol_reference'],*plan['bank']]
    required += [str(Path(__file__).resolve().with_name(name)) for name in SOURCES]
    if not set(required) <= set(plan['frozen']):
        raise ValueError('scoring plan lacks required input/source hashes')
    def verify():
        for path, expected in plan['frozen'].items():
            if digest(path) != expected:
                raise ValueError('frozen scoring input changed: '+path)
    verify()
    arms, evidence = {}, {}
    for name, arm in plan['arms'].items():
        # A receipt must bind its declared prediction and assert completion.
        # Owners translate their trainer's completion evidence into this small
        # contract after checking the fixed recipe, not after choosing a score.
        receipt_path = Path(arm['receipt'])
        receipt = json.loads(receipt_path.read_text())
        if (receipt['state'] != 'complete' or receipt['smoke_only']
                or receipt['experiment'] != plan['experiment'] or receipt['arm'] != name
                or receipt['data_sha256'] != plan['frozen'][plan['data']]
                or Path(receipt['prediction']).resolve() != Path(arm['prediction']).resolve()):
            raise ValueError(name+' receipt is incomplete or mismatched')
        if digest(arm['prediction']) != receipt['prediction_sha256']:
            raise ValueError(name+' prediction differs from completion receipt')
        arms[name] = np.fromfile(arm['prediction'],dtype='<f4').astype(float)
        evidence[name] = dict(receipt_sha256=digest(receipt_path),
                              prediction_sha256=receipt['prediction_sha256'],
                              receipt=str(receipt_path),prediction=arm['prediction'])
    d = load(plan['data'])
    y, users = d['probe']['rating'].astype(float), d['probe']['user']
    bank = np.column_stack([np.fromfile(path,dtype='<f4').astype(float) for path in plan['bank']])
    result, predictions = compare(bank,arms['control'],arms['treatment'],y,users)
    result.update(experiment=plan['experiment'],plan_sha256=digest(plan_path),arms=evidence,
        minimum_gain=plan['minimum_gain'],passes_p2_gate=result['primary']['delta_rmse']<=-plan['minimum_gain'],
        scope='P2 user fold 0 only; identical duplicate-control comparison, penalty selection on fold 1, fits on folds 2-4 then 1-4. Full-bank promotion remains separate.',
        training_provenance='Owner-validated completion receipts; this scorer does not audit trainer-specific settings.')
    verify()
    for arm in evidence.values():
        if (digest(arm['receipt']) != arm['receipt_sha256']
                or digest(arm['prediction']) != arm['prediction_sha256']):
            raise ValueError('arm changed during scoring')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return result,predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result,_ = score(args.plan,args.output)
    print(json.dumps(result,indent=2),flush=True)
