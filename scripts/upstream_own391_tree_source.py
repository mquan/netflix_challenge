#!/usr/bin/env python3
"""Torch-capable391 source preparation/stream, isolated from LightGBM runtime."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import contextlib
import hashlib
from pathlib import Path
import sys
import numpy as np
import upstream_repro as u
import upstream_tree_pipe as wire

REPORT=u.ROOT/'experiments/upstream-own391-cpu-tree'
RUN=u.ROOT/'artifacts/upstream-own391-cpu-tree'
DECL=u.ROOT/'OWN391_CPU_TREE_PLAN.md'


def declare():
    import upstream_own391_paired_data as data
    if REPORT.exists() or RUN.exists():raise ValueError('Existing tree study retained')
    mp=data.REPORT/'manifest.json';vp=data.REPORT/'independent-verification.json';m,v=u.read(mp),u.read(vp)
    partner=u.ROOT/'experiments/upstream-own352-bounded-replay/independent-canonical-verification.json'
    proof=u.read(partner)
    if (v['state']!='complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions' or
        v['manifest_sha256']!=u.sha(mp) or v['source_sha256']!=m['source_sha256'] or
        not v['every_full_operator_statistic_solve_and_prediction_bit_equal'] or not v['matched_B_known_and_qualifying'] or
        v['qualifying_labels_read'] is not False or
        proof['state']!='complete_separate_bounded_original352_tree_half_arithmetic_and_full_canonical_score'):
        raise ValueError('Full matched391 data and actual verified fixed NN partner required')
    bindings=dict(m['source_sha256'])
    def bind(path,digest=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if digest is None else digest
        if key in bindings and bindings[key]!=value:raise ValueError('Original source expectation conflicts')
        bindings[key]=value
    for name,digest in m['prepared_sha256'].items():bind(data.RUN/name,digest)
    nn=u.ROOT/'artifacts/upstream-own352-temporal-mps/nn.f32'
    replay_path=u.ROOT/'experiments/upstream-own352-bounded-replay/independent-verification.json';replay=u.read(replay_path)
    if (proof['complete_model_preprocessing_verification_sha256']!=u.sha(replay_path) or
        proof['manifest_sha256']!=replay['manifest_sha256'] or
        u.sha(nn)!=replay['nn_sha256']):raise ValueError('Actual original fully replayed NN partner differs')
    for key,digest in proof['source_sha256'].items():bind(u.ROOT/key,digest)
    bind(nn,replay['nn_sha256'])
    for path in [mp,vp,partner,replay_path,DECL,Path(__file__),Path(wire.__file__),
        u.ROOT/'scripts/upstream_own391_tree_worker.py',u.ROOT/'scripts/verify_upstream_own391_cpu_tree.py',
        u.ROOT/'scripts/upstream_own391_cpu_tree_queue.py',u.ROOT/'scripts/upstream_nonlinear.py']:
        bind(path)
    data.check(bindings);REPORT.mkdir();RUN.mkdir()
    for name in ['labels.npy','users.npy','groups.npy','stats.npz']:u.link_or_copy(data.RUN/name,RUN/name)
    u.link_or_copy(nn,RUN/'partner-nn.f32')
    from upstream_nonlinear import CONFIG,versions
    u.write(REPORT/'manifest.json',dict(state='declared_full_budget_matched391_cpu_tree',source_sha256=bindings,
        config=CONFIG,versions=versions(),rows=1128164,features=617,nn_inputs=618,design_columns=467,
        data_manifest_sha256=u.sha(mp),partner_nn_sha256=u.sha(nn),reference_rmse=proof['rmse'],
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},qualifying_labels_read=False,
        nonlinear_OOF_predictions_used_as_features=False))
    print('Complete matched391 CPU tree declaration; no fit',flush=True)


def stream(independent):
    output=sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        import upstream_own391_paired_data as data
        import verify_upstream_own391_paired_data as alternate
        m=u.read(REPORT/'manifest.json');dm=u.read(data.REPORT/'manifest.json');data.check(m['source_sha256'])
        if m['data_manifest_sha256']!=u.sha(data.REPORT/'manifest.json'):raise ValueError('Frozen391 data differs')
        with np.load(RUN/'stats.npz') as saved:gram,rhs=saved['gram'],saved['rhs']
        weights=np.stack([data.n.ridge_weights(gram,rhs,(np.arange(10)//2!=fold)&selector,1000)
            for fold in range(5) for selector in [np.arange(10)%2==0,np.ones(10,dtype=bool)]])
        wire.write_json(output,dict(state='matched391_tree_pipe_v1',rows=1128164,features=617,baselines=10,chunk=8192,
            manifest_sha256=u.sha(REPORT/'manifest.json'),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))
        reader=data.old.base.Inputs('probex',dm['original_rows']) if independent else data.Inputs('probex',dm['original_rows'],dm['temporal_rows'],dm['new_rows'])
        digests=[hashlib.sha256(),hashlib.sha256(),hashlib.sha256()]
        try:
            for lo in range(0,1128164,8192):
                hi=min(lo+8192,1128164)
                z,f=alternate.block(reader,'probex',dm['temporal_rows'],dm['new_rows'],lo,hi) if independent else reader.block(lo,hi)
                baselines=np.column_stack([np.clip(z@w,1.05,4.95) for w in weights])
                digests[0].update(z.tobytes());digests[1].update(f.tobytes());digests[2].update(baselines.tobytes())
                wire.write_array(output,f);wire.write_array(output,baselines)
        finally:reader.close()
        if [d.hexdigest() for d in digests[:2]]!=dm['virtual_c_order_sha256']['probex']:
            raise ValueError('Every original391 known operator bit required')
        data.check(m['source_sha256'])
        wire.write_json(output,dict(operators_sha256=[d.hexdigest() for d in digests[:2]],baseline_sha256=digests[2].hexdigest()))
        output.flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','stream','stream-independent']);args=parser.parse_args()
    declare() if args.action=='declare' else stream(args.action=='stream-independent')
