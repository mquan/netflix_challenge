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
import upstream_weekend_trajectory_ops as ops
import upstream_own391_paired_data as parent
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

u = parent.u
REPORT = u.ROOT / 'experiments/upstream-weekend-trajectory-data'
RUN = u.ROOT / 'artifacts/upstream-weekend-trajectory-data'
DECL = u.ROOT / 'WEEKEND_COMPLETE_TRAJECTORY_PLAN.md'
CF = u.ROOT / 'experiments/upstream-complete-cf-delta'
NATIVE = u.ROOT / 'experiments/upstream-original1200'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Guard:
    def __init__(self):
        import upstream_cf96_primary_lane as identity
        self.pid = os.getpid()
        self.queue = u.ROOT/'experiments/upstream-weekend-trajectory-data-queue'
        owner = u.read(self.queue/'monitor.json'); child = u.read(self.queue/'child-ownership.json')
        actual = identity.process(self.pid)
        if (os.getpgid(self.pid) != self.pid or child['pid'] != self.pid or actual is None or
            child['started'] != actual['started'] or child.get('worker_command') != actual['command'] or
            not identity.same_owner(owner, identity.process(owner['pid']))):
            raise ValueError('Use the exact supervised dedicated scientific process before allocation')
        self.baseline = u.read(self.queue/'first-scientific-baseline.json')
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
    import upstream_complete_trajectory_data as local
    import upstream_remote_weekend_return as incoming
    m, added, bindings, cf_hashes = local.sources()
    if len(added)!=128: raise ValueError('Complete original128 local candidate sequence required')
    mp,vp=local.REPORT/'manifest.json',local.REPORT/'independent-verification.json'
    lm,lv=u.read(mp),u.read(vp)
    if (lv['state']!='complete_independent_all_trajectory_operators_statistics_ridge_and_stored_predictions' or
        lv['manifest_sha256']!=ops.sha(mp) or lv['source_sha256']!=lm['source_sha256'] or lm['source_sha256']!=bindings or
        lv['qualifying_labels_read'] is not False or not lv['every_full_operator_statistic_solve_and_prediction_bit_equal']):
        raise ValueError('Complete preceding local trajectory input/ridge replay required')
    mapping='artifacts/upstream-split3-pilot/probex-row-map.npy'
    weekend=u.read(incoming.PLAN)
    ops.merge_bindings(bindings,{mapping:weekend['row_map_sha256']})
    files=[mp,vp,local.REPORT/'input-manifest.json',local.REPORT/'input-verification.json',
           DECL,Path(__file__),Path(ops.__file__),u.ROOT/'scripts/verify_upstream_weekend_trajectory_data.py',incoming.PLAN]
    for job in weekend['jobs']:
        name=job['name'];report=u.ROOT/'experiments/upstream-remote-weekend-return'/name
        proof_path=report/'independent-verification.json';proof=u.read(proof_path)
        completion_path=u.ROOT/'experiments/upstream-remote-weekend-return-queue'/name/'completion.json';completion=u.read(completion_path)
        if (completion['state']!='complete_full_weekend_return_transport_and_independent_admission' or completion['job']!=name or
            completion['verification_sha256']!=ops.sha(proof_path) or completion['qualifying_labels_read'] is not False or
            proof['state']!='complete_independent_weekend_full_source_queries_projection_disjointness_and_all_compact_bits' or
            proof['job']!=name or proof['all_pairs']!=job['published_columns'] or proof['qualifying_labels_read'] is not False or
            proof['numerical_public_features_read'] is not False or proof['score_claimed'] is not False or
            proof['rows']!=dict(native_holdout=1760494,known=1128164,qualifying=2817131,training_scanned=98720013)):
            raise ValueError('Every original full weekend producer must independently complete before this study')
        ops.merge_bindings(bindings,proof['source_sha256']);files += [proof_path,completion_path]
        prefix='artifacts/upstream-remote-weekend-return/'+name+'/native/preds_3/'
        for column in job['published_columns']:
            pair={'probex':prefix+column+'.probe_3.npy','qual':prefix+column+'.qual.npy'}
            if any(path not in proof['source_sha256'] for path in pair.values()):
                raise ValueError('Both original native query expectations required')
            added.append(dict(id='weekend:'+column,paired_files=pair,known_row_map=mapping))
    if len(added)!=195: raise ValueError('All128 local and67 remote candidates required')
    ops.merge_bindings(bindings,{str(p.relative_to(u.ROOT)):ops.sha(p) for p in files})
    verify_bindings(bindings)
    return m,added,bindings,cf_hashes


def original_candidate_hashes(query):
    ledger=u.read(u.ROOT/'experiments/upstream-complete-trajectory-data/input-manifest.json')['clipped_sha256'][query]
    expected=list(ledger['original'])+list(ledger['candidates'])
    for job in u.read(u.ROOT/'experiments/upstream-remote-weekend/plan.json')['jobs']:
        proof=u.read(u.ROOT/'experiments/upstream-remote-weekend-return'/job['name']/'independent-verification.json')
        expected += [proof['clipped_pair_sha256'][column][query] for column in job['published_columns']]
    if len(expected)!=586:raise ValueError('Every original local and remote clipped query expectation required')
    return expected


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
        worker_sha256=ops.sha(Path(__file__)), verifier_sha256=ops.sha(u.ROOT / 'scripts/verify_upstream_weekend_trajectory_data.py'),
        declaration_sha256=ops.sha(DECL), lambda_fixed=1000, folds=5, groups=10,
        all195_candidates_before_alias_removal=True, qualifying_labels_read=False))
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
        if hashes[query]['original']+hashes[query]['candidates'] != original_candidate_hashes(query):
            raise ValueError('Every original391+128 and67 remote query hash must match its original admission')
        print('Full586 scalar query streams:', query, flush=True)
    retained, aliases = ops.partition({q: hashes[q]['original'] for q in ops.COUNTS}, candidates,
                                      {q: hashes[q]['candidates'] for q in ops.COUNTS})
    verify_bindings(bindings); guard.check()
    u.write(REPORT / 'input-manifest.json', dict(state='complete_all586_scalar_pairs_pending_independent_replay',
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
        iv['state'] != 'complete_independent_all586_scalar_pairs_and_both_operators' or
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
