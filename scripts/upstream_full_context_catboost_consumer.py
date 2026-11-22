"""Torch-isolated complete known-fold consumer component; no CLI or owner.

A future source-bound supervisor must authorize the actual producer/consumer
identities, source metadata, handoff and complete16GiB group budget first.
These routines provide no independent production launch or score promotion.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import numpy as np
import upstream_full_context_catboost_wire as wire
import upstream_full_context_catboost_fold as fold_math
import upstream_rbm400_latent_operators as operators
import upstream_rbm400_latent_cache_reference_v2 as cache
from upstream_factored_rbm_inputs import verify_files
from upstream_factored_rbm_admission import file_identity
import upstream_cf96_primary_lane as identity

ROWS = 1128164
QUALIFYING_ROWS = 2817131
DATA = 'experiments/upstream-rbm400-context-data-reference-v2'
REPLAY = 'experiments/upstream-rbm400-context-source-alignment-replay'
CONTROL = 'experiments/upstream-rbm400-context-source-alignment-queue'
ARRAYS = 'artifacts/upstream-rbm400-context-data-reference-v2'


def no_torch():
    if 'torch' in sys.modules:
        raise ValueError('Complete tree consumer must remain isolated from Torch')


def verified_json(path, expected_sha256, root, guard):
    path, root = Path(path), Path(root)
    if (not callable(guard) or not wire.digest_ok(expected_sha256) or root.resolve() != root
            or path.resolve() != path or not path.is_relative_to(root)):
        raise ValueError('Canonical owned metadata and independently frozen digest required')
    guard()
    with path.open('rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 16 * 2**20:
            raise ValueError('Bounded complete regular metadata required')
        try:
            h = hashlib.sha256(); parts = []
            while True:
                guard(); block = source.read(1048576)
                if not block: break
                parts.append(block); h.update(block)
            if h.hexdigest() != expected_sha256:
                raise ValueError('Actual owned metadata differs from frozen digest')
            return json.loads(b''.join(parts))
        finally:
            guard()
            if file_identity(before) != file_identity(os.fstat(source.fileno())) or file_identity(before) != file_identity(path.stat()):
                raise ValueError('Owned metadata identity changed during use')


def contract(metadata, expected_sha256, root):
    no_torch(); root = Path(root)
    if (root.resolve() != root or not wire.digest_ok(expected_sha256)
            or hashlib.sha256(wire.encoded(metadata)).hexdigest() != expected_sha256
            or metadata['format'] != 'own_full_context_tree_original_source_v1'
            or metadata['arm'] not in ('scalar', 'latent') or metadata['rows'] != ROWS
            or metadata['qualifying_rows'] != QUALIFYING_ROWS or metadata['ridge_penalty'] != 1000.
            or metadata['ridge_equations'] != 10 or metadata['qualifying_labels_read'] is not False
            or metadata['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or metadata['models_fitted'] != 0):
        raise ValueError('Exact original known-only full source metadata required')
    dimensions = operators.dimensions(metadata['retain_scalar'], metadata['arm'] == 'latent')
    if metadata['dimensions'] != dimensions:
        raise ValueError('Every original raw input and context column required')
    known = metadata['known_arrays']
    if set(known) != {'labels.npy', 'users.npy', 'groups.npy'}:
        raise ValueError('Only the three complete original known populations required')
    for name, dtype in [('labels.npy', '<f8'), ('users.npy', '<i4'), ('groups.npy', '|i1')]:
        row = known[name]; path = root / ARRAYS / name; relative = str(path.relative_to(root))
        if (Path(row['path']) != path or row['dtype'] != dtype or row['shape'] != [ROWS]
                or metadata['source_sha256'].get(relative) != row['sha256']):
            raise ValueError('Complete original prepared known file binding required')
    parents = {DATA + '/manifest.json': metadata['parent_CPU_manifest_sha256'],
               REPLAY + '/independent-verification.json': metadata['parent_CPU_verification_sha256']}
    if any(metadata['source_sha256'].get(name) != digest for name, digest in parents.items()) or CONTROL + '/completion.json' not in metadata['source_sha256']:
        raise ValueError('Complete actual source-aligned CPU proof and owned completion required')
    return wire.contract(ROWS, retain_scalar=metadata['retain_scalar'], latent=metadata['arm'] == 'latent',
        weights_sha256=metadata['weights_sha256'], admission_sha256=expected_sha256, operator_sha256=metadata['operator_sha256'])


def authenticate_sources(metadata, expected_sha256, root, guard):
    contract(metadata, expected_sha256, root)
    records = verify_files(root, metadata['source_sha256'], guard)
    for name, row in metadata['known_arrays'].items():
        actual = records[str(Path(row['path']).relative_to(root))]
        if any(actual[key] != row[key] for key in ('sha256', 'bytes', 'identity')):
            raise ValueError('Every original prepared known byte and identity required')
    source = metadata['source_sha256']
    manifest = verified_json(root / DATA / 'manifest.json', source[DATA + '/manifest.json'], root, guard)
    proof = verified_json(root / REPLAY / 'independent-verification.json', source[REPLAY + '/independent-verification.json'], root, guard)
    done = verified_json(root / CONTROL / 'completion.json', source[CONTROL + '/completion.json'], root, guard)
    if (manifest['state'] != 'complete_both_context_arms_data_and_known_ridge_pending_independent_full_replay'
            or manifest['rows'] != ROWS or manifest['qualifying_rows'] != QUALIFYING_ROWS
            or manifest['qualifying_labels_read'] is not False or manifest['neural_models_fitted'] != 0
            or manifest['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or manifest['lambda_fixed'] != 1000. or manifest['folds'] != 5 or manifest['groups'] != 10
            or manifest['alias']['retain_scalar'] != metadata['retain_scalar']
            or manifest['arms'][metadata['arm']] != metadata['dimensions']
            or manifest['virtual_c_order_sha256']['probex']['arm_operator_sha256'][metadata['arm']] != metadata['operator_sha256']
            or proof['state'] != 'complete_independent_both_context_arms_all_operators_aliases_statistics_and_known_ridge'
            or proof['manifest_sha256'] != metadata['parent_CPU_manifest_sha256']
            or proof['source_sha256'] != manifest['source_sha256']
            or any(source.get(name) != digest for name, digest in manifest['source_sha256'].items())
            or proof['qualifying_labels_read'] is not False or proof['neural_models_fitted'] != 0
            or proof['both_query_alias_independently_verified'] is not True
            or proof['every_both_arm_operator_bit_verified'] is not True
            or proof['every_group_statistic_and_ridge_prediction_bit_verified'] is not True
            or proof['rows'] != ROWS or proof['qualifying_rows'] != QUALIFYING_ROWS
            or proof['ridge_equations'] != 5 * metadata['dimensions']['design_columns']
            or done['state'] != 'complete_both_RBM400_context_arms_data_and_independent_known_ridge_replay'
            or done['verification_sha256'] != metadata['parent_CPU_verification_sha256']
            or done['qualifying_labels_read'] is not False):
        raise ValueError('Actual complete original CPU source/operator/statistic replay required')
    for name, row in metadata['known_arrays'].items():
        original = manifest['prepared'][name]
        if original['path'] != row['path'] or original['sha256'] != row['sha256'] or original['bytes'] != row['bytes']:
            raise ValueError('Known populations must match the actual complete CPU manifest')
    return records


@contextmanager
def known_arrays(metadata, records, root, guard):
    with ExitStack() as stack:
        result = {}
        for name, row in metadata['known_arrays'].items():
            path = Path(row['path'])
            result[name] = stack.enter_context(cache.mapped(path, records[str(path.relative_to(root))], row['dtype'], (ROWS,), guard))
        y, users, groups = [result[name] for name in ('labels.npy', 'users.npy', 'groups.npy')]
        if not np.isin(y, [1, 2, 3, 4, 5]).all():
            raise ValueError('Complete original known integer outcomes required')
        fold_math.masks(users, groups, 0)
        yield y, users, groups


def producer_exit(path, expected_sha256, captured, expected_contract, metadata_sha256, root, guard):
    no_torch()
    value = verified_json(path, expected_sha256, root, guard)
    digest = hashlib.sha256(wire.encoded(expected_contract)).hexdigest()
    if (value['state'] != 'complete_owned_original_tree_producer_zero_exit'
            or type(value['exit_code']) is not int or value['exit_code'] != 0
            or value['producer'] != captured or value['contract_sha256'] != digest
            or value['source_admission_sha256'] != metadata_sha256
            or value['qualifying_labels_read'] is not False):
        raise ValueError('Actual successful original producer exit bound to this exact stream required')
    actual = identity.process(captured['pid'])
    # Exec may replace a captured bootstrap command with the real producer.
    # The same PID and start time still denote the original live lifetime.
    if actual and actual['pid'] == captured['pid'] and actual['started'] == captured['started']:
        raise ValueError('Original producer is still live; EOF cannot authorize fitting')
    return value


def fit(features, baselines, metadata, metadata_sha256, wire_proof, *, producer_receipt,
        producer_receipt_sha256, producer_identity, root, guard, save_model, save_population, save_fold,
        release, train_fold=fold_math.run):
    """Run every original fold only after full handoff and actual zero-exit gate.

    The external supervisor must authenticate receipt provenance against its
    captured producer identity and authorized phase. Callback outputs must be
    durably retained in its fresh namespace; this function supplies no retry.
    """
    expected = contract(metadata, metadata_sha256, root)
    if (any(not callable(f) for f in (guard, save_model, save_population, save_fold, release, train_fold))
            or features.shape != (ROWS, expected['width']) or features.dtype != np.dtype('<f4')
            or not features.flags.c_contiguous or not features.flags.writeable
            or baselines.shape != (5, 2, ROWS) or baselines.dtype != np.dtype('<f8')
            or np.may_share_memory(features, baselines)
            or wire_proof['contract_sha256'] != hashlib.sha256(wire.encoded(expected)).hexdigest()
            or wire_proof['admission_sha256'] != metadata_sha256
            or wire_proof['operator_sha256'] != expected['operator_sha256']
            or wire_proof['exact_footer_and_EOF'] is not True or wire_proof['models_fitted'] != 0
            or wire_proof['qualifying_labels_read'] is not False):
        raise ValueError('Every original completed raw handoff row and population required')
    producer_exit(producer_receipt, producer_receipt_sha256, producer_identity, expected, metadata_sha256, root, guard)
    records = authenticate_sources(metadata, metadata_sha256, root, guard)
    # Recheck the in-memory transfer before any matrix column is replaced.
    raw_hash = hashlib.sha256(); hashes = [hashlib.sha256() for _ in range(10)]
    for lo in range(0, ROWS, 8192):
        guard(); hi = min(lo + 8192, ROWS); raw_hash.update(features[lo:hi].tobytes())
        for f in range(5):
            for p in range(2): hashes[2*f+p].update(baselines[f, p, lo:hi].tobytes())
    if raw_hash.hexdigest() != wire_proof['raw_sha256'] or [h.hexdigest() for h in hashes] != wire_proof['baseline_sha256']:
        raise ValueError('Transferred raw inputs or eligible baselines changed before fitting')
    prediction = np.empty(ROWS, dtype='<f4'); seen = np.zeros(ROWS, dtype=bool); receipts = []
    with known_arrays(metadata, records, root, guard) as (labels, users, groups):
        for fold in range(5):
            guard(); no_torch()
            result = train_fold(features, baselines[fold], labels, users, groups, fold,
                save_model=lambda model, seed, count: save_model(fold, model, seed, count),
                save_population=lambda stage, proof: save_population(fold, stage, proof), release=release, guard=guard)
            values = result.pop('prediction'); held = groups // 2 == fold
            if values.shape != (int(held.sum()),) or values.dtype != np.dtype('<f4') or not np.isfinite(values).all() or seen[held].any():
                raise ValueError('Every original held-row prediction exactly once required')
            prediction[held] = values; seen[held] = True
            save_fold(fold, result, values); receipts.append(result)
    if not seen.all(): raise ValueError('All five complete original held populations required')
    if authenticate_sources(metadata, metadata_sha256, root, guard) != records:
        raise ValueError('Complete original source identities changed during full tree fitting')
    no_torch(); guard()
    return dict(state='complete_five_original_tree_fold_slots_pending_independent_replay_and_owned_admission',
                folds=receipts, prediction=prediction, production_study_admitted=False, qualifying_labels_read=False)
