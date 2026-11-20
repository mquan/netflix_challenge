#!/usr/bin/env python3
"""Admit every completed own trajectory, then fit the fixed matched ridge."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import datetime as dt
import hashlib
from pathlib import Path
import shutil
import signal
import numpy as np
import upstream_complete_trajectory_ops as ops
import upstream_own391_paired_data as parent
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

u = parent.u
REPORT = u.ROOT / 'experiments/upstream-complete-trajectory-data'
RUN = u.ROOT / 'artifacts/upstream-complete-trajectory-data'
DECL = u.ROOT / 'OWN_COMPLETE_TRAJECTORY_NEURAL_PLAN.md'
CF = u.ROOT / 'experiments/upstream-complete-cf-delta'
NATIVE = u.ROOT / 'experiments/upstream-original1200'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Guard:
    def __init__(self):
        import upstream_cf96_primary_lane as identity
        self.pid = os.getpid()
        self.queue = u.ROOT/'experiments/upstream-complete-trajectory-data-queue'
        owner = u.read(self.queue/'monitor.json'); child = u.read(self.queue/'child-ownership.json')
        actual = identity.process(self.pid)
        if (os.getpgid(self.pid) != self.pid or child['pid'] != self.pid or actual is None or
            child['started'] != actual['started'] or child.get('worker_command') != actual['command'] or
            not identity.same_owner(owner, identity.process(owner['pid']))):
            raise ValueError('Use the exact supervised dedicated scientific process before allocation')
        self.baseline = memory.memory()
        self.check(3 * 2**30)

    def check(self, reserve=0):
        while True:
            sample = dict(pid=self.pid, original_baseline=self.baseline, observed=stamp())
            try:
                workers, current = resources.footprints(self.pid), memory.memory()
                own = workers[str(self.pid)]
                safe = (max(own['bytes'], own['rss_bytes']) <= 3 * 2**30 and
                    sum(w['bytes'] for w in workers.values()) + reserve < 24 * 2**30 and
                    sum(w['rss_bytes'] for w in workers.values()) + reserve < 24 * 2**30 and
                    current['compressor_bytes'] < 5 * 2**30 and
                    current['swap_used_bytes'] - self.baseline['swap_used_bytes'] <= 128 * 2**20 and
                    shutil.disk_usage(u.ROOT).free >= 18.75 * 2**30)
                if safe: return
                sample.update(workers=workers, memory=current)
            except Exception as error:
                sample['measurement_error'] = repr(error)
            try:
                u.write(self.queue/('self-guard-%s.json'%self.pid), sample)
            except Exception:
                pass  # Preserve RAM even if the evidence disk itself is unavailable.
            os.killpg(self.pid, signal.SIGSTOP)
            # An explicitly reviewed same-process continuation rechecks the SAME
            # baseline here; no automatic restart, refit or loss of local state.


def verify_bindings(bindings):
    # These two producer inputs contain only synthetic zeros, never recovered truth.
    synthetic = u.read(u.BASE / 'fresh-training-plan.json')['synthetic_qualifying_payload_sha256']
    allowed = {'artifacts/' + name + '/source/data/qual_ratings/qual_ratings.csv.gz'
               for name in ['upstream-original1200', 'upstream-high-capacity-original']}
    for name, digest in bindings.items():
        ops.relative(name)
        if 'qual_ratings.csv' in name and (name not in allowed or digest != synthetic):
            raise ValueError('Recovered outcomes cannot enter fitting closure')
        if ops.sha(u.ROOT / name) != digest:
            raise ValueError('Original immutable source changed: ' + name)


def sources():
    """No scores or recovered labels select the complete candidate sequence."""
    mp, vp = parent.REPORT / 'manifest.json', parent.REPORT / 'independent-verification.json'
    m, v = u.read(mp), u.read(vp)
    if (v['state'] != 'complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions' or
        v['manifest_sha256'] != ops.sha(mp) or v['source_sha256'] != m['source_sha256'] or
        not v['matched_B_known_and_qualifying'] or v['qualifying_labels_read'] is not False or
        [m[k] for k in ['raw_predictions', 'design_columns', 'nonlinear_columns', 'nn_inputs']] != [391, 467, 617, 618]):
        raise ValueError('Full verified matched B391 parent required')
    im, iv, fm, fv = [u.read(CF / name) for name in
                      ['input-manifest.json', 'input-verification.json', 'fit-manifest.json', 'independent-verification.json']]
    cp = u.read(CF / 'plan.json')
    if (iv['state'] != 'complete_independent_all488_scalar_pairs_and_novelty' or
        iv['input_manifest_sha256'] != ops.sha(CF / 'input-manifest.json') or
        fv['state'] != 'complete_independent_expanded_CF95_TX50n_known_ridge_and_canonical_scores' or
        fv['fit_manifest_sha256'] != ops.sha(CF / 'fit-manifest.json') or
        fm['input_manifest_sha256'] != ops.sha(CF / 'input-manifest.json') or
        fm['input_verification_sha256'] != ops.sha(CF / 'input-verification.json') or
        any(row['qualifying_labels_read'] is not False for row in [cp, im, iv, fm, fv]) or
        any(row['source_sha256'] != cp['source_sha256'] for row in [im, iv, fm, fv]) or
        cp['old_scalar_columns'] != ops.RAW391 or cp['lambda_fixed'] != 1000 or len(cp['candidates']) != 97):
        raise ValueError('Complete original CF95/TX50n admission and full ridge replay required')
    bindings = dict(m['source_sha256'])
    ops.merge_bindings(bindings, cp['source_sha256'])
    ops.merge_bindings(bindings, {str((parent.RUN / name).relative_to(u.ROOT)): value
                                for name, value in m['prepared_sha256'].items()})
    files = [mp, vp, DECL, Path(__file__), Path(ops.__file__), Path(resources.__file__), Path(memory.__file__),
             u.ROOT / 'scripts/verify_upstream_complete_trajectory_data.py', u.BASE / 'fresh-training-plan.json']
    files += [CF / name for name in ['plan.json', 'input-manifest.json', 'input-verification.json',
                                    'fit-manifest.json', 'independent-verification.json']]
    added = list(cp['candidates'])
    native_specs = [
        ('upstream-high-capacity-original', 'tsvdx4-1600lm', 2, [], False,
         'tsvdx4-1600lm-verification.json', 'complete_original1600_both_full_phases_and_all_native_query_vectors'),
        ('upstream-original1200', 'tsvdx5-1200lm', 23, ['bias', 'mf', 'nsvd1', 'nbr'], True,
         'independent-verification.json', 'complete_independent_original1200_both23_phases_and_all58_numeric_outputs')]
    for namespace, name, epochs, components, training, proof_name, state in native_specs:
        report = u.ROOT / 'experiments' / namespace
        plan_path, proof_path = report / 'execution-plan.json', report / proof_name
        plan, proof = u.read(plan_path), u.read(proof_path)
        if (plan['source_pin'] != u.PIN or proof['state'] != state or proof['epochs_per_phase' if epochs == 2 else 'epochs_each'] != epochs or
            proof['factors'] != (1600 if epochs == 2 else 1200) or proof['qualifying_labels_read'] is not False or
            not plan['qualifying_targets_zero'] or plan['qualifying_labels_read'] is not False or
            proof['admission']['preparation_sha256'] != ops.sha(plan_path) or
            proof['admission']['qualifying_labels_read'] is not False or
            plan['full_identity_disjointness'] != [
                dict(training_rows=99352343, query_rows=1128164, all_training_identities_scanned=True, overlap=0),
                dict(training_rows=100480507, query_rows=2817131, all_training_identities_scanned=True, overlap=0)]):
            raise ValueError('Original full native source/data/run/output proof required')
        ops.merge_bindings(bindings, plan['bindings'])
        ops.merge_bindings(bindings, proof['outputs'])
        build_path = report/'original-build-receipt.json'
        build_digest = proof['admission']['original_build_receipt_sha256' if epochs == 2 else 'build_sha256']
        ops.merge_bindings(bindings, {str(build_path.relative_to(u.ROOT)): build_digest})
        if epochs == 2:
            completion_path = u.ROOT/'experiments/upstream-high-capacity-successor/completion.json'
            completion = u.read(completion_path)
            if (completion['state'] != 'complete_original1600_and_independent_full_output_verification' or
                completion['verification_sha256'] != ops.sha(proof_path)):
                raise ValueError('Original1600 successor completion must bind its full proof')
            files.append(completion_path)
        else:
            completion_path = u.ROOT/'experiments/upstream-original1200-exact-build-recovery/completion.json'
            completion = u.read(completion_path)
            if (completion['state'] != 'complete_original1200_full23_23_and_independent58_output_verification' or
                completion['proof_sha256'] != ops.sha(proof_path)):
                raise ValueError('Original1200 completion must bind its full independent proof')
            files.append(completion_path)
        prefix = 'artifacts/' + namespace + '/source/preds_new/' + name
        added += ops.native_rows(proof['outputs'], prefix, epochs, components, training)
        run_path = report / (name + '-run.json' if epochs == 2 else 'native-run.json')
        if ops.sha(run_path) != proof['original_run_sha256' if epochs == 2 else 'native_run_sha256']:
            raise ValueError('Original full native run receipt changed')
        binary = u.ROOT / 'artifacts' / namespace / 'target/release' / ('tsvdx4-new' if epochs == 2 else 'tsvdx5-new')
        ops.merge_bindings(bindings, {str(binary.relative_to(u.ROOT)): proof['admission']['binary_sha256']})
        files += [plan_path, proof_path, run_path]
    if len(added) != 128:
        raise ValueError('All97 CF/TX, all3 native1600 and all28 native1200 pairs required')
    ops.merge_bindings(bindings, {str(p.relative_to(u.ROOT)): ops.sha(p) for p in files})
    verify_bindings(bindings)
    return m, added, bindings, im['clipped_sha256']


class Inputs:
    def __init__(self, query, m, rows):
        self.query, self.rows = query, rows
        self.parent = parent.Inputs(query, m['original_rows'], m['temporal_rows'], m['new_rows'])

    def block(self, lo, hi):
        z, f = self.parent.block(lo, hi)
        return ops.extend(z, f, [ops.scalar(u.ROOT, row, self.query, lo, hi) for row in self.rows])

    def close(self):
        self.parent.close()


def admit():
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing complete trajectory study retained')
    guard = Guard()
    m, candidates, bindings, cf_hashes = sources()
    REPORT.mkdir(); RUN.mkdir()
    u.write(REPORT / 'plan.json', dict(source_sha256=bindings, candidates=candidates,
        worker_sha256=ops.sha(Path(__file__)), verifier_sha256=ops.sha(u.ROOT / 'scripts/verify_upstream_complete_trajectory_data.py'),
        declaration_sha256=ops.sha(DECL), lambda_fixed=1000, folds=5, groups=10,
        all128_candidates_before_alias_removal=True, qualifying_labels_read=False))
    hashes = {}
    for query, count in ops.COUNTS.items():
        old = [hashlib.sha256() for _ in range(391)]
        extra = [hashlib.sha256() for _ in candidates]
        operators = [hashlib.sha256(), hashlib.sha256()]
        source = parent.Inputs(query, m['original_rows'], m['temporal_rows'], m['new_rows'])
        try:
            for block, lo in enumerate(range(0, count, 8192)):
                hi = min(lo + 8192, count); z, f = source.block(lo, hi)
                operators[0].update(z.tobytes()); operators[1].update(f.tobytes())
                for i, column in enumerate(ops.RAW391):
                    old[i].update(np.clip(z[:, column], 0, 6).astype('<f4').tobytes())
                for i, row in enumerate(candidates):
                    extra[i].update(ops.scalar(u.ROOT, row, query, lo, hi).tobytes())
                if block % 20 == 0: guard.check()
        finally:
            source.close()
        if [h.hexdigest() for h in operators] != m['virtual_c_order_sha256'][query]:
            raise ValueError('Every original467/617 operator bit must replay')
        hashes[query] = dict(original=[h.hexdigest() for h in old], candidates=[h.hexdigest() for h in extra])
        if hashes[query]['original'] + hashes[query]['candidates'][:97] != cf_hashes[query]:
            raise ValueError('Every previously verified391+97 scalar bit must remain unchanged')
        print('Full519 scalar query streams:', query, flush=True)
    retained, aliases = ops.partition({q: hashes[q]['original'] for q in ops.COUNTS}, candidates,
                                      {q: hashes[q]['candidates'] for q in ops.COUNTS})
    verify_bindings(bindings); guard.check()
    u.write(REPORT / 'input-manifest.json', dict(state='complete_all519_scalar_pairs_pending_independent_replay',
        plan_sha256=ops.sha(REPORT / 'plan.json'), source_sha256=bindings, clipped_sha256=hashes,
        retained=retained, aliases=aliases, raw_predictions=391+len(retained), design_columns=467+len(retained),
        nonlinear_columns=617+len(retained), nn_inputs=618+len(retained), models_fitted=0,
        qualifying_labels_read=False, matched_B_known_and_qualifying=True))


def fit():
    if (REPORT / 'manifest.json').exists():
        raise ValueError('Existing original complete trajectory ridge retained')
    if any(RUN.iterdir()):
        raise ValueError('Partial original numerical artifacts retained; never restart fitting')
    guard = Guard(); m, candidates, bindings, _ = sources()
    p, im, iv = [u.read(REPORT / name) for name in ['plan.json', 'input-manifest.json', 'input-verification.json']]
    if (p['source_sha256'] != bindings or p['candidates'] != candidates or im['source_sha256'] != bindings or
        iv['state'] != 'complete_independent_all519_scalar_pairs_and_both_operators' or
        iv['input_manifest_sha256'] != ops.sha(REPORT / 'input-manifest.json') or
        iv['source_sha256'] != bindings or iv['qualifying_labels_read'] is not False):
        raise ValueError('Complete independently verified input list required before fitting')
    for name in ['labels.npy', 'groups.npy', 'users.npy']:
        u.link_or_copy(parent.RUN / name, RUN / name)
    y, groups = np.load(RUN/'labels.npy'), np.load(RUN/'groups.npy')
    d = im['design_columns']; gram = np.zeros((10, d, d)); rhs = np.zeros((10, d)); streams = {}
    cg, cr = np.zeros((10, 467, 467)), np.zeros((10, 467))
    for query, count in ops.COUNTS.items():
        reader = Inputs(query, m, im['retained']); hashes = [hashlib.sha256(), hashlib.sha256()]
        try:
            for block, lo in enumerate(range(0, count, 8192)):
                hi=min(lo+8192, count); z, f=reader.block(lo, hi)
                hashes[0].update(z.tobytes()); hashes[1].update(f.tobytes())
                if query == 'probex':
                    original = np.column_stack([z[:, :466], z[:, -1:]])
                    for group in range(10):
                        keep=groups[lo:hi] == group; a=z[keep]
                        gram[group] += a.T @ a; rhs[group] += a.T @ y[lo:hi][keep]
                        control_block = original[keep]
                        cg[group] += control_block.T @ control_block
                        cr[group] += control_block.T @ y[lo:hi][keep]
                if block % 20 == 0: guard.check()
        finally: reader.close()
        streams[query] = [h.hexdigest() for h in hashes]
    weights=np.stack([parent.n.ridge_weights(gram, rhs, np.arange(10)//2 != fold, 1000) for fold in range(5)])
    control=np.stack([parent.n.ridge_weights(cg, cr, np.arange(10)//2 != fold, 1000) for fold in range(5)])
    predictions=[np.empty(len(y), dtype='<f4') for _ in range(2)]
    reader=Inputs('probex', m, im['retained'])
    try:
        for block, lo in enumerate(range(0, len(y), 8192)):
            hi=min(lo+8192, len(y)); z, _=reader.block(lo, hi)
            for fold in range(5):
                held=groups[lo:hi]//2 == fold
                for values, a, w in zip(predictions, [z, np.column_stack([z[:, :466], z[:, -1:]])], [weights, control]):
                    values[lo:hi][held]=np.clip(a[held] @ w[fold], 1.05, 4.95).astype('<f4')
            if block % 20 == 0: guard.check()
    finally: reader.close()
    np.savez(RUN/'stats.npz', gram=gram, rhs=rhs, control_gram=cg, control_rhs=cr); np.save(RUN/'weights.npy', weights)
    np.save(RUN/'control391-weights.npy', control)
    predictions[0].tofile(RUN/'ridge.f32'); predictions[1].tofile(RUN/'control391.f32')
    if ops.sha(RUN/'control391.f32') != m['prepared_sha256']['ridge.f32']:
        raise ValueError('Complete matchedB391 control predictions must remain bitwise identical')
    verify_bindings(bindings); guard.check()
    u.write(REPORT/'manifest.json', dict(state='complete_all_trajectory_operators_and_known_ridge_pending_independent_replay',
        source_sha256=bindings, plan_sha256=ops.sha(REPORT/'plan.json'), input_manifest_sha256=ops.sha(REPORT/'input-manifest.json'),
        input_verification_sha256=ops.sha(REPORT/'input-verification.json'), retained=im['retained'], rows=1128164,
        qualifying_rows=2817131, raw_predictions=im['raw_predictions'], design_columns=d,
        nonlinear_columns=im['nonlinear_columns'], nn_inputs=im['nn_inputs'], matched_B_known_and_qualifying=True,
        complete_original_B391_operator_bits_preserved=True, virtual_c_order_sha256=streams,
        ridge_rmse=parent.score(predictions[0], y), control391_rmse=parent.score(predictions[1], y),
        ridge_fold_rmse=[parent.score(predictions[0][groups//2 == f], y[groups//2 == f]) for f in range(5)],
        prepared_sha256={p.name: ops.sha(p) for p in RUN.iterdir() if p.is_file()},
        qualifying_labels_read=False, neural_models_fitted=0, nonlinear_component_OOF_predictions_used_as_features=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase', choices=['admit', 'fit']); args=parser.parse_args()
    {'admit': admit, 'fit': fit}[args.phase]()
