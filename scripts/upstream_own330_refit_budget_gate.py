#!/usr/bin/env python3
"""Bind full-refit medians to the original independently verified actual fits."""
from pathlib import Path
import numpy as np
import upstream_repro as u

DEVELOPMENT = u.ROOT / 'experiments/upstream-own330-mps'
PRODUCTION = u.ROOT / 'experiments/upstream-own330-production-inputs'


def validate(records, receipts, original_manifest_sha256):
    if len(records) != 5 or sorted(r['fold'] for r in records) != list(range(5)):
        raise ValueError('Exactly five original independently verified folds required')
    counts = []
    for record, receipt, digest in receipts:
        if (digest != record['receipt_sha256'] or receipt['fold'] != record['fold'] or
                receipt['manifest_sha256'] != original_manifest_sha256 or
                receipt['outputs'] != record['outputs']):
            raise ValueError('Original independently verified receipt identity changed')
        values = receipt['chosen_iterations']
        if len(values) != 3 or any(type(v) is not int or not 0 < v <= 40 for v in values):
            raise ValueError('All three original bounded nonzero selected budgets required')
        counts.append(values)
    if len(counts) != 5:
        raise ValueError('All five actual fold budgets required')
    return [int(np.median([row[j] for row in counts])) for j in range(3)]


def main():
    output = PRODUCTION / 'original-budget-verification.json'
    if output.exists():
        raise ValueError('Existing original-budget admission retained')
    proof_path = DEVELOPMENT / 'independent-verification.json'
    original = u.read(proof_path)
    declared = u.read(PRODUCTION / 'declaration.json')
    if (original['state'] != 'complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays'
            or original['manifest_sha256'] != u.sha(DEVELOPMENT / 'manifest.json')):
        raise ValueError('Original independent full model proof required')
    rows = []
    bindings = {str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__), proof_path,
        DEVELOPMENT / 'manifest.json', PRODUCTION / 'declaration.json']}
    for record in original['records']:
        path = u.ROOT / record['receipt_path']
        if path != DEVELOPMENT / 'nn' / (str(record['fold']) + '.json'):
            raise ValueError('Original fold receipt path differs')
        receipt = u.read(path); digest = u.sha(path)
        rows.append((record, receipt, digest)); bindings[str(path.relative_to(u.ROOT))] = digest
        for name, expected in record['outputs'].items():
            artifact = u.ROOT / 'artifacts/upstream-own330-mps/nn' / str(record['fold']) / name
            if artifact.name != name or u.sha(artifact) != expected:
                raise ValueError('Actual original fitted output changed')
            bindings[str(artifact.relative_to(u.ROOT))] = expected
    epochs = validate(original['records'], rows, original['manifest_sha256'])
    if declared['full_iterations'] != epochs or declared['seeds'] != [2026, 7, 13]:
        raise ValueError('Declared full-refit medians or seeds differ from original fits')
    u.write(output, dict(state='complete_full_refit_medians_bound_to_all_fifteen_actual_original_fits',
        full_iterations=epochs, seeds=declared['seeds'], source_sha256=bindings,
        qualifying_labels_read=False, models_fitted=False))


if __name__ == '__main__':
    main()
