"""Blend the paired stack on rows with original Probe exposure represented in development."""
import importlib.util
import json
from pathlib import Path

import numpy as np

import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_stable_nonlinear as nonlinear
import own_stable_subspace as stable

OUT = ne.ROOT / 'own-supported-stack-20261127'
RECEIPT = Path('experiments/OWN-final-overnight-20261126.json')


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def stable_development(folds, report):
    freeze = stable.OUT / 'freeze.json'
    assert ne.sha(freeze) == report['freeze_sha256']
    info = read(freeze)
    assert ne.sha(stable.canonical.MARK) == info['raw_inputs_sha256']
    role = nonlinear.Role('probex', np.load(stable.PARENT / 'known-selected.npy'),
                          np.load(stable.OUT / 'fit.npz'), info['selected'])
    prior = np.empty(len(folds))
    for start in range(0, len(folds), 16384):
        rows = slice(start, start + 16384)
        prior[rows] = role.prior(rows, folds)
    return prior


def weights(residual, direction, exposure, train):
    result = np.zeros(3)
    for count in (0, 1):
        rows = train & (exposure == count)
        x = direction[rows]
        result[count] = np.clip((x @ residual[rows]) / (x @ x + 1), 0, .5)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    save('protocol.json', dict(primary='paired-10', baseline=str(stable.OUT),
         design='Fit a convex weight capped at 0.5 separately for 0 and 1 visible Probe ratings. Use zero for 2 or more.',
         gate='Improve pooled ProbeX RMSE and at least four held-user folds. Freeze one qualifying forecast.',
         exploratory=True, qualifying_labels_for_weight_selection=False))
    report = read(RECEIPT)
    primary = blend.OUT / report['parent_stage']
    paired_receipt = Path(report['parent_receipt'])
    assert ne.sha(paired_receipt) == report['parent_receipt_sha256']
    src = read(primary / 'freeze.json')
    assert not src['qualifying_labels_used_for_fitting']
    assert src['qualifying_hashes']['selected'] == report['sha256'] == ne.sha(primary / 'selected.f32')
    stable_report = report['stable_subspace_diagnostic']
    parent = stable.OUT / 'selected.f32'
    expected = read(stable.OUT / 'freeze.json')['qualifying_hashes']['selected']
    assert ne.sha(parent) == read(stable.OUT / 'independent-refit.json')['sha256'] == expected
    users = ne.history('probex', 'user_idxs')
    folds = ne.user_folds(users).astype(int)
    y = ne.history('probex', 'ratings').astype(float)
    prior = stable_development(folds, stable_report)
    other = np.load(primary / 'known-selected.npy')
    control = blend.metrics(prior, y, folds)
    np.testing.assert_allclose(control['folds'], stable_report['development'][stable_report['selected']]['folds'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(blend.metrics(other,y,folds)['folds'], report['known']['folds'], atol=1e-12,rtol=0)
    old_users = anchor.old_probe()[0]
    held = retained.map_rows()
    np.testing.assert_array_equal(old_users[held], users)
    observed = np.ones(len(old_users), bool)
    observed[held] = False
    nu = len(ne.history('trainx', 'user_cnts'))
    exposure = np.bincount(old_users[observed], minlength=nu)[users]
    assert exposure.max() == 1
    direction = other - prior
    err = y - prior
    candidate = prior.copy()
    for k in range(5):
        test = folds == k
        w = weights(err, direction, exposure, ~test)
        candidate[test] += w[exposure[test]] * direction[test]
    metrics = blend.metrics(candidate,y,folds)
    accept = metrics['pooled'] < control['pooled'] and sum(x < z for x,z in zip(metrics['folds'],control['folds'])) >= 4
    w = weights(err, direction, exposure, np.ones(len(y),bool))
    hashes = {}
    result = dict(control=control, candidate=metrics, weights=w.tolist(), selected='candidate' if accept else 'control',
                  qualifying_hashes=hashes, known_comparison_is_nested=False, provenance_receipt_sha256=ne.sha(RECEIPT))
    if accept:
        qu = ne.history('qual','user_idxs')
        groups = np.minimum(2, np.bincount(old_users, minlength=nu)[qu])
        base = np.fromfile(parent,'<f4').astype(float)
        pred = np.fromfile(primary / 'selected.f32','<f4').astype(float)
        selected = (base + w[groups]*(pred-base)).astype('<f4')
        assert np.array_equal(selected[groups==2],base[groups==2])
        path = OUT / 'selected.f32'
        selected.tofile(path)
        hashes['selected'] = ne.sha(path)
        result['qualifying_exposure_counts'] = np.bincount(groups).tolist()
    save('freeze.json',result)
    print(json.dumps(result),flush=True)
    if accept:
        ne.audit(OUT,ne.LABELS)
        spec = importlib.util.spec_from_file_location('verifier', ne.RANK/'evaluate.py')
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        verifier.verify_audit(OUT,hashes)


if __name__ == '__main__':
    main()
