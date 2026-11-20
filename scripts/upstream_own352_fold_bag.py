#!/usr/bin/env python3
"""Qualifying inference from all original fifteen own352 saved fold models."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import datetime as dt
import hashlib
from pathlib import Path
import numpy as np
import torch
import upstream_own352_temporal_mps as study
import upstream_own330_production_inputs as base

u = study.u
REPORT = u.ROOT/'experiments/upstream-own352-fold-bag'
RUN = u.ROOT/'artifacts/upstream-own352-fold-bag'
REPLAY = u.ROOT/'experiments/upstream-own352-bounded-replay'
ROWS, CHUNK = 2817131, 65536


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sources():
    manifest = u.read(study.REPORT/'manifest.json')
    proof = u.read(REPLAY/'independent-verification.json')
    canonical = u.read(REPLAY/'independent-canonical-verification.json')
    if (proof['state'] != 'complete_actual_own352_bounded_original_statistics_scalers_and_saved_model_replays'
            or canonical['state'] != 'complete_separate_bounded_original352_tree_half_arithmetic_and_full_canonical_score'
            or proof['manifest_sha256'] != u.sha(study.REPORT/'manifest.json')
            or canonical['complete_model_preprocessing_verification_sha256'] != u.sha(REPLAY/'independent-verification.json')
            or canonical['manifest_sha256'] != u.sha(study.REPORT/'manifest.json')
            or not canonical['improves_previous'] or not canonical['all15_actual_saved_models_rehashed']
            or not all(r['actual_models_reopened'] == 3 and r['actual_saved_pipeline_bitwise_equal']
                       and r['both_original_eligible_scalers_recomputed'] and r['all_original_normalized_input_bits_equal']
                       for r in proof['records']) or len(proof['records']) != 5):
        raise ValueError('Complete original own352 models, statistics and canonical eligibility required')
    bindings = dict(canonical['source_sha256'])
    def bind(path, expected=None):
        key = str(path.relative_to(u.ROOT)); value = u.sha(path) if expected is None else expected
        if key in bindings and bindings[key] != value:
            raise ValueError('Original own source expectation conflicts: '+key)
        bindings[key] = value
    declaration = u.read(base.REPORT/'declaration.json')
    checked = u.read(base.REPORT/'independent-verification.json')
    if (checked['state'] != 'complete_both_full_paired_own330_source_column_reconstructions'
            or checked['declaration_sha256'] != u.sha(base.REPORT/'declaration.json')):
        raise ValueError('Complete original paired own330 production operators required')
    for key, value in declaration['source_sha256'].items():
        bind(u.ROOT/key, value)
    added = u.read(study.data.REPORT/'manifest.json')['recent_rows']
    if (len(added) != 22 or len(declaration['rows']) != 223 or manifest['config'] != study.CONFIG
            or manifest['nn_inputs'] != 579 or manifest['versions'] != study.n.versions()):
        raise ValueError('All original complete579 inputs and22 fixed native counterparts required')
    for row in added:
        bind(u.ROOT/row['paired_files']['qual'], row['qual_sha256'])
    for record in proof['records']:
        receipt = u.read(u.ROOT/record['receipt_path'])
        if (u.sha(u.ROOT/record['receipt_path']) != record['receipt_sha256']
                or receipt['manifest_sha256'] != u.sha(study.REPORT/'manifest.json')
                or receipt['outputs'] != record['outputs'] or len(receipt['chosen_iterations']) != 3
                or any(type(v) is not int or not 0 < v <= 40 for v in receipt['chosen_iterations'])):
            raise ValueError('Every actual original selected fold/seed pipeline required')
    for path in [Path(__file__), u.ROOT/'scripts/verify_upstream_own352_fold_bag.py',
                 u.ROOT/'OWN352_FOLD_BAG_PRODUCTION_PLAN.md', study.REPORT/'manifest.json',
                 REPLAY/'independent-verification.json', REPLAY/'independent-canonical-verification.json',
                 base.REPORT/'declaration.json', base.REPORT/'transport.json', base.REPORT/'independent-verification.json',
                 study.data.REPORT/'manifest.json']:
        bind(path)
    study.data.check(bindings)
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying fitting targets must remain zero')
    return bindings, declaration['rows'], added


class Inputs:
    def __init__(self, original, added):
        if len(added) != 22:
            raise ValueError('All22 original temporal qualifying vectors required')
        self.parent = base.Inputs('qual', original)
        self.added = added

    def block(self, lo, hi):
        z, f = self.parent.block(lo, hi)
        values = []
        # Keep only one additional descriptor open beyond the223-input parent.
        for row in self.added:
            reader = base.raw.Rows(u.ROOT/row['paired_files']['qual'], (ROWS,), np.float32)
            try:
                values.append(np.clip(reader.read(lo, hi), 0, 6))
            finally:
                reader.close()
        added = np.column_stack(values)
        result = (np.column_stack([z[:, :-1], added, z[:, -1:]]),
                  np.column_stack([f, added-np.float32(3.6)]))
        if (result[0].shape != (hi-lo, 428) or result[1].shape != (hi-lo, 578)
                or result[0].dtype != np.float64 or result[1].dtype != np.float32
                or not np.isfinite(result[0]).all() or not np.isfinite(result[1]).all()):
            raise ValueError('Complete finite original428/578 qualifying operators required')
        return result

    def close(self):
        self.parent.close()


def load_fold(fold):
    root = study.RUN/'nn'/str(fold)
    receipt = u.read(study.REPORT/'nn'/('%s.json'%fold))
    with np.load(root/'ridge.npz') as saved:
        weights = saved['full'].copy()
    with np.load(root/'scalers.npz') as saved:
        scaler = {key:saved['full_'+key].copy() for key in ['lo','hi','mean','std']}
    models = []
    for seed, count in zip(study.CONFIG['seeds'], receipt['chosen_iterations']):
        path = root/('nn-%s.pt'%seed)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['feature_width'] != 579 or saved['seed'] != seed+fold or saved['epochs'] != count
                or saved['config'] != study.CONFIG):
            raise ValueError('Actual complete original model seed/width/iterations/config differs')
        models.append(study.mps.load_model(path, study.CONFIG))
    return weights, scaler, models


def infer(fold, original, added):
    weights, scaler, models = load_fold(fold)
    inputs = Inputs(original, added); result = np.empty(ROWS, dtype='<f4')
    hashes = [hashlib.sha256(), hashlib.sha256()]
    retained = [hashlib.sha256(), hashlib.sha256()]
    try:
        for lo in range(0, ROWS, CHUNK):
            hi = min(lo+CHUNK, ROWS); z, f = inputs.block(lo, hi)
            if fold == 0:
                hashes[0].update(z.tobytes(order='C')); hashes[1].update(f.tobytes(order='C'))
                retained[0].update(np.column_stack([z[:, :405], z[:, -1:]]).tobytes(order='C'))
                retained[1].update(f[:, :556].tobytes(order='C'))
            baseline = np.clip(z@weights, 1.05, 4.95)
            x = study.scaling.append_baseline(f, baseline)
            study.scaling.apply_scaler(x, scaler, output=x)
            correction = np.zeros(hi-lo, dtype=np.float64)
            for model in models:
                correction += study.mps.predict(model, x)/3
            result[lo:hi] = np.clip(baseline+correction, 1.05, 4.95).astype('<f4')
            u.write(REPORT/'status.json', dict(state='full_original_fold_qualifying_inference',
                fold=fold, complete_rows=hi, total_rows=ROWS, observed=stamp()))
    finally:
        inputs.close(); del models; torch.mps.empty_cache()
    if fold == 0:
        if [h.hexdigest() for h in retained] != u.read(base.REPORT/'transport.json')['c_order_sha256']['qual']:
            raise ValueError('Every retained original own330 qualifying input bit must match')
        u.write(REPORT/'input-verification.json', dict(
            state='complete_original428_578_qualifying_stream_pending_independent_reconstruction',
            c_order_sha256=[h.hexdigest() for h in hashes], all_original330_bits_retained=True))
    return result


def main():
    if RUN.exists() or REPORT.exists():
        raise ValueError('Existing original fold bag or partial evidence retained')
    bindings, original, added = sources()
    RUN.mkdir(); REPORT.mkdir(); study.mps.MODEL_FACTORY = study.model_factory
    u.write(REPORT/'plan.json', dict(state='declared_all_original_fold_models_before_qualifying_inference',
        created=stamp(), source_sha256=bindings, original_rows=original, added_rows=added,
        config=study.CONFIG, folds=list(range(5)), seeds=study.CONFIG['seeds'], rows=ROWS,
        chunk_rows=CHUNK, qualifying_labels_read=False, models_fitted=0,
        original_declaration_sha256=u.sha(u.ROOT/'OWN352_FOLD_BAG_PRODUCTION_PLAN.md'),
        input_realization='Original A-trained fold models and paired B qualifying inputs remain distinct; no matched B-trained CV claim.',
        known_calibration='Only original held-user OOF vector; never in-sample fold averages.'))
    for fold in range(5):
        infer(fold, original, added).tofile(RUN/('fold-%s.f32'%fold))
    bag = np.zeros(ROWS, dtype=np.float64)
    for fold in range(5):
        bag += np.fromfile(RUN/('fold-%s.f32'%fold), dtype='<f4').astype(np.float64)
    (bag/5).astype('<f4').tofile(RUN/'nn-bag.f32')
    study.data.check(bindings)
    u.write(REPORT/'inference-completion.json', dict(
        state='complete_all_fifteen_original_models_full_qualifying_inference_pending_independent_replay',
        plan_sha256=u.sha(REPORT/'plan.json'), input_verification_sha256=u.sha(REPORT/'input-verification.json'),
        outputs={p.name:u.sha(p) for p in RUN.iterdir()}, qualifying_labels_read=False,
        models_fitted=0, complete_query_rows=ROWS, source_fold_models=15))


if __name__ == '__main__':
    main()
