#!/usr/bin/env python3
"""Prepare a fixed two-temporal-model extension of our own91 development bank."""
import datetime as dt
from pathlib import Path
import shutil
import numpy as np
import upstream_source_extension as parent
import upstream_remote_batch as algebra
from upstream_full_reproduce import receipt_valid

u,n,x=parent.u,parent.n,parent.x
RUN=u.ROOT/'artifacts/upstream-temporal93'
REPORT=u.ROOT/'experiments/upstream-temporal93'
NEW=['tsvdx4-60o','tsvdx4-q16tb']


def code_hashes():
    return {**parent.code_hashes(),str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__))}


def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing temporal93 preparation preserved')
    parent.configure();pm=n.verify_prepared();jobs=u.read(u.BASE/'registry.json')['jobs']
    proof=parent.REPORT/'independent-verification.json';evaluation=parent.REPORT/'evaluation.json'
    if u.read(proof)['evaluation_sha256']!=u.sha(evaluation):raise ValueError('Parent development score binding differs')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
        script_sha256=u.sha(Path(__file__)),fixed_additions=NEW,config=n.CONFIG,
        parent_manifest_sha256=u.sha(parent.REPORT/'manifest.json'),
        rule='Append both completed original temporal models together. No feature subset/weight selection on qualifying outcomes. Known-ProbeX five-user-fold ridge control and all-two arm only. Ridge diagnostics are not a nonlinear or qualifying promotion.',
        nonlinear_fitting_started=False,qualifying_labels_read=False))
    sources={str(p.relative_to(u.ROOT)):u.sha(p) for p in [parent.REPORT/'manifest.json',proof,evaluation,REPORT/'plan.json']}
    added=[]
    for name in NEW:
        if not receipt_valid(name,jobs):raise ValueError('Complete original two-phase native receipt required: '+name)
        receipt=u.BASE/'fresh-receipts'/(name+'.json');sources[str(receipt.relative_to(u.ROOT))]=u.sha(receipt)
        for split,count in [('probex',1128164),('qual',2817131)]:
            path=u.FRESH/'preds_new'/(name+'.'+split+'.npy');values=np.load(path,mmap_mode='r',allow_pickle=False)
            if values.shape!=(count,) or values.dtype!=np.dtype('float32') or not np.isfinite(values).all():
                raise ValueError('Malformed paired native vector')
            sources[str(path.relative_to(u.ROOT))]=u.sha(path)
            if split=='probex':added.append(np.clip(values,0,6))
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying targets must remain zero')
    y,users,groups=[np.load(parent.RUN/(key+'.npy')) for key in ['labels','users','groups']]
    for key,name in [('labels','ratings.npy'),('users','user_idxs.npy')]:
        path=u.FRESH/'data/probex'/name
        if not np.array_equal(y if key=='labels' else users,np.load(path,allow_pickle=False)):
            raise ValueError('Known-ProbeX identity differs')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    old=np.load(parent.RUN/'design.npy',mmap_mode='r');features=np.load(parent.RUN/'features.npy',mmap_mode='r')
    if old.shape!=(len(y),167) or features.shape!=(len(y),227):raise ValueError('Frozen91 parent layout differs')
    extra=np.column_stack(added);scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]))
    if abs(x.metric(scored['control']['prediction'],y)-pm['baseline_rmse'])>1e-9:
        raise ValueError('Actual parent ridge control does not reproduce')
    RUN.mkdir()
    z=np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,shape=(len(y),169))
    f=np.lib.format.open_memmap(RUN/'features.npy',mode='w+',dtype=np.float32,shape=(len(y),229))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y));z[lo:hi]=algebra.augmented_block(old,extra,lo,hi)
        f[lo:hi]=np.column_stack([features[lo:hi],extra[lo:hi]-np.float32(3.6)])
    z.flush();f.flush()
    for key in ['labels','users','groups']:shutil.copyfile(parent.RUN/(key+'.npy'),RUN/(key+'.npy'))
    np.save(RUN/'baseline.npy',scored['main']['prediction'])
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    records=[]
    for arm,result in scored.items():
        path=RUN/(arm+'-ridge.npy');np.save(path,result['prediction'])
        records.append(dict(arm=arm,rmse=x.metric(result['prediction'],y),
            fold_rmse=[x.metric(result['prediction'][groups//2==fold],y[groups//2==fold]) for fold in range(5)],prediction_sha256=u.sha(path)))
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,qualifying_labels_read=False,
        scope='Fixed all-two temporal addition on known-ProbeX only. Same five user folds; diagnostic ridge comparison, not a global nonlinear best or qualifying result.'))
    sources[str((REPORT/'ridge-screen.json').relative_to(u.ROOT))]=u.sha(REPORT/'ridge-screen.json')
    manifest=dict(config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},
        rows=len(y),prediction_columns=93,design_columns=169,nonlinear_features=229,
        baseline_rmse=x.metric(scored['main']['prediction'],y),parent_baseline_rmse=pm['baseline_rmse'],
        new_native_predictions=NEW,qualifying_labels_read=False,isolation=pm['isolation'],
        scope='Adaptive own-training known-ProbeX development. Both original temporal models are complete for both native phases; original capacity/source/epochs preserved. Cached parent source realization is retained for development; no production bank or recovered-label audit is created.',
        arms=['control','tree','nn','fixed-half-tree-nn'])
    u.write(REPORT/'manifest.json',manifest)
    print(dict(state='prepared',records=records,nonlinear_fitting_started=False),flush=True)


if __name__=='__main__':prepare()
