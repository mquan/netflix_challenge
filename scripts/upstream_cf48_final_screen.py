#!/usr/bin/env python3
"""Fixed published CF48 epoch95 addition to our verified own102 development bank."""
from pathlib import Path
import math
import numpy as np
import upstream_cf48_102 as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-cf48-final-screen'
REPORT=u.ROOT/'experiments/upstream-cf48-final-screen'
NAME='cfnade-48_ep95'


def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing final-snapshot screen preserved')
    intake=u.ROOT/'experiments/upstream-remote-cf48-final/paired-receipt.json';received=u.read(intake)
    if (received['state']!='verified_complete_resumed_remote_prediction_intake' or not received['full95_model_complete'] or
            not received['reconstructed_native_terminal_and_epoch95_sha256_match'] or received['qualifying_labels_read'] is not False):
        raise ValueError('Verified complete own resumed CF48 final pair required')
    temporal.configure();parent=temporal.n.verify_prepared()
    proof_path=temporal.REPORT/'cache-column-verification.json';checked=u.read(proof_path)
    if checked['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json') or not checked['all_retained_and_new_columns_checked']:
        raise ValueError('Verified own102 cache required')
    sources={**received['source_sha256'],**received['outputs']}
    for p in [intake,proof_path,temporal.REPORT/'manifest.json',Path(__file__),
            u.ROOT/'scripts/verify_upstream_ready_public_schema.py']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Source/output binding changed: '+name)
    pairs=received['pairs'][NAME];row=dict(name=NAME,paired_files=pairs,duplicate_of=None)
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.clip(np.load(u.ROOT/pairs['probex'],allow_pickle=False),0,6)[:,None]
    if extra.shape!=(len(y),1):raise ValueError('Final CF48 known-row shape differs')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(ready_unused=[row],fixed_additions=[NAME],full95_model_complete=True,
        rule='One published final-snapshot addition fixed before scoring; parent already includes the published epoch25 snapshot. Retain control and all-one arms. No qualifying outcomes.',
        qualifying_labels_read=False,public_prediction_values_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0]));records=[]
    for arm,result in scored.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,result['prediction'])
        value=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(value-parent['baseline_rmse'])>1e-12:raise ValueError('Own102 control differs')
        records.append(dict(arm=arm,rmse=value,prediction_sha256=u.sha(p),
            fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=[NAME],source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),full95_model_complete=True,qualifying_labels_read=False,
        public_prediction_values_read=False,scope='Known-ProbeX fixed final95-snapshot ridge diagnostic. No nonlinear or qualifying promotion.'))
    print(records,flush=True)


if __name__=='__main__':main()
