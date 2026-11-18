#!/usr/bin/env python3
"""Verify the selected fresh native replay and evaluate without released features."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import shutil
import numpy as np
import upstream_repro as u
import upstream_exchange as x
import upstream_expansion as e
import upstream_snapshot_blend as b
import upstream_missing_screen as s


def main():
    u.validate_fresh()
    plan_path = s.REPORT/'selected-replay-plan.json'
    plan = u.read(plan_path)
    name = plan['job']
    if plan['source_commit'] != u.PIN:
        raise ValueError('Selected replay source changed')
    parent = u.BASE/'fresh-receipts/rbmx2-150.json'
    if u.sha(parent) != plan['reusable_parent_receipt_sha256']:
        raise ValueError('Frozen reusable parent changed')
    receipt_path = u.BASE/'fresh-receipts'/(name+'.json')
    receipt = u.read(receipt_path)
    if receipt['source_commit'] != u.PIN or receipt['job'] != name:
        raise ValueError('Wrong native replay receipt')
    for path, digest in receipt['outputs'].items():
        if u.sha(u.FRESH/path) != digest:
            raise ValueError('Native replay output changed')
    labels, users, fresh = e.collect_expansion()
    folds = x.user_folds(users)
    frozen = u.read(b.REPORT/'manifest.json')
    current = b.freeze(labels, users, fresh)
    if any(current[k] != frozen[k] for k in current):
        raise ValueError('Fresh baseline changed')
    for path, digest in frozen['context_inputs'].items():
        if u.sha(u.ROOT/path) != digest:
            raise ValueError('Context metadata changed')
    context = b.context_features(*[np.load(u.ROOT/p, mmap_mode='r', allow_pickle=False)
                                   for p in frozen['context_inputs']])
    if x.array_sha(context) != frozen['context_content_sha256']:
        raise ValueError('Context content changed')
    fourteen, _ = b.evaluate([fresh[n] for n in e.ARMS['all-fourteen']], [True]*14, labels, folds)
    if abs(x.metric(fourteen, labels)-0.8677119362283443) > 1e-9:
        raise ValueError('Fourteen control differs')
    rows = [r for r in frozen['entries'] if r['duplicate_of'] is None]
    arrays, clip = [fresh[r['name']] for r in rows], [r['clip'] for r in rows]
    control, _ = b.evaluate(arrays, clip, labels, folds, context, tuple(range(14)))
    if abs(x.metric(control, labels)-0.8663200739003126) > 1e-9:
        raise ValueError('Snapshot/context control differs')
    native = np.load(u.FRESH/'preds_new'/(name+'.probex.npy'), allow_pickle=False)
    if native.dtype != np.float32 or native.shape != labels.shape or not np.isfinite(native).all():
        raise ValueError('Malformed fresh replay vector')
    out, width = b.evaluate(arrays+[native], clip+[True], labels, folds, context, tuple(range(14)))
    dest = s.REPORT/'replay'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/(name+'.npz'), prediction=native)
    u.write(dest/(name+'.receipt.json'), receipt)
    shutil.copyfile(receipt_path.with_suffix('.log'), dest/(name+'.log'))
    artifact = u.ROOT/'artifacts/upstream-missing-screen'
    artifact.mkdir(parents=True, exist_ok=True)
    prediction_path = artifact/(name+'.f32')
    out.astype('<f4').tofile(prediction_path)
    # Released fixture is read only AFTER fresh predictions/evaluation are final.
    reference_manifest = u.read(u.BASE/'reference-manifest.json')
    reference_entry = next(r for r in reference_manifest['entries'] if r['name'] == name)
    reference_path = u.REFERENCE/reference_entry['path']
    if u.sha(reference_path) != reference_entry['sha256']:
        raise ValueError('Released diagnostic fixture changed')
    fixture = np.load(reference_path, allow_pickle=False)
    delta = [x.metric(out[folds==f], labels[folds==f])-x.metric(control[folds==f], labels[folds==f]) for f in range(5)]
    u.write(dest/'evaluation.json', dict(job=name, source_commit=u.PIN, rows=len(labels),
        selected_plan_sha256=u.sha(plan_path), baseline_manifest_sha256=u.sha(b.REPORT/'manifest.json'),
        prediction_columns=len(arrays)+1, design_columns=width,
        rmse=x.metric(out, labels), previous_fresh_control_rmse=x.metric(control, labels),
        original_fourteen_control_rmse=x.metric(fourteen, labels), delta_vs_previous=x.metric(out, labels)-x.metric(control, labels),
        fold_rmse=[x.metric(out[folds==f], labels[folds==f]) for f in range(5)], fold_delta_vs_previous=delta,
        fresh_base_rmse=x.metric(native, labels), released_base_rmse=x.metric(fixture, labels),
        fresh_released_prediction_rms_difference=float(np.sqrt(np.mean((native.astype(np.float64)-fixture)**2))),
        native_receipt_sha256=u.sha(receipt_path), exported_artifact_sha256=u.sha(dest/(name+'.npz')),
        output_prediction_sha256=u.sha(prediction_path),
        scope='Fresh output and fold-refitted fresh single-layer blend; recipe selected by the released-vector development screen, so this is exploratory development evidence, not untouched/selection-adjusted performance.',
        input_provenance='Only receipt-verified fresh predictions plus training/query metadata. Released vectors used solely for post-fit numerical comparison. New rating column has a global coefficient; original fourteen-only context interactions preserved.'))
    print(u.read(dest/'evaluation.json'))


if __name__ == '__main__':
    main()
