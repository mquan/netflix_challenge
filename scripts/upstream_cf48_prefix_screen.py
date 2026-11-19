#!/usr/bin/env python3
"""Fixed published CF48 epoch25 diagnostic from our ongoing original95 trajectory."""
import math
from pathlib import Path
import numpy as np
import upstream_ready_public_schema as ready
import upstream_cfnade48 as cf
import upstream_temporal93 as temporal
import upstream_remote_batch as algebra

u=ready.u
REPORT=u.ROOT/'experiments/upstream-cf48-prefix-screen'
RUN=u.ROOT/'artifacts/upstream-cf48-prefix-screen'
NAME='cfnade-48_ep25'


def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing prefix experiment preserved')
    plan=cf.verify_plan()
    if (plan['source_pin']!=u.PIN or plan['hidden']!=160 or plan['rank']!=48 or
            plan['epochs_per_phase']!=95 or plan['qualifying_labels_read'] is not False):
        raise ValueError('Original160/48/95 own-training configuration required')
    package_path=u.ROOT/'experiments/upstream-remote9h/cfnade48-package.json'
    package=u.read(package_path)
    manifest_path=u.ROOT/'artifacts/remote9h-transfer/cfnade48-package/manifest.json'
    manifest=u.read(manifest_path)
    if (u.sha(manifest_path)!=package['manifest_sha256'] or manifest['source_id']!=plan['source_id'] or
            manifest['data_id']!=plan['data_id'] or manifest['original_plan_sha256']!=u.sha(cf.REPORT/'plan.json')):
        raise ValueError('Own captured warm package provenance differs')
    old_plan=u.read(ready.REPORT/'plan.json');old_screen=u.read(ready.REPORT/'ridge-screen.json')
    old_proof=u.read(ready.REPORT/'ridge-verification.json')
    if old_proof['screen_sha256']!=u.sha(ready.REPORT/'ridge-screen.json'):
        raise ValueError('Independent ready-eight diagnostic binding differs')
    if NAME not in {r['name'] for r in old_plan['coverage']}:
        raise ValueError('Snapshot must be fixed by the published schema before scoring')
    sources=dict(old_screen['source_sha256'])
    for p in [Path(__file__),Path(cf.__file__),Path(ready.__file__),Path(algebra.__file__),
            cf.REPORT/'plan.json',package_path,manifest_path,ready.REPORT/'plan.json',
            ready.REPORT/'ridge-screen.json',ready.REPORT/'ridge-verification.json',
            u.ROOT/'scripts/verify_upstream_ready_public_schema.py']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    row=dict(name=NAME,producer='cfnade-48',duplicate_of=None,paired_files={},sha256={},
        prediction_epoch=25,original_configured_epochs_per_phase=95,
        full95_model_complete=False,scope='Complete epoch25 prediction pair from our own ongoing95-epoch trajectory, not a completed95-epoch native job.')
    for train,query,rows in [('trainx','probex',1128164),('fulltrain','qual',2817131)]:
        path=cf.OUT/(NAME+'.'+query+'.npy')
        expected=manifest['phases'][train]['prediction_sha256'][path.name]
        if package['phases'][train]['prediction_sha256'][path.name]!=expected or u.sha(path)!=expected:
            raise ValueError('Actual own snapshot differs from captured warm trajectory')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=(rows,) or values.dtype!=np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError('Invalid own prefix vector')
        row['paired_files'][query]=str(path.relative_to(u.ROOT));row['sha256'][query]=expected
        sources[str(path.relative_to(u.ROOT))]=expected
    if np.any(np.load(u.FRESH/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Qualifying outcomes are not zero')
    y=np.load(temporal.RUN/'labels.npy');users=np.load(temporal.RUN/'users.npy');groups=np.load(temporal.RUN/'groups.npy')
    for key,expected in [('ratings',y),('user_idxs',users)]:
        path=u.FRESH/'data/probex'/(key+'.npy')
        if not np.array_equal(np.load(path,allow_pickle=False),expected):raise ValueError('Known row identity differs')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    selected=[r for r in old_plan['ready_unused'] if r['duplicate_of'] is None]+[row]
    if len(selected)!=9:raise ValueError('Fixed all-nine arm changed')
    REPORT.mkdir();RUN.mkdir()
    fixed=dict(fixed_additions=[r['name'] for r in selected],ready_unused=old_plan['ready_unused']+[row],
        source_pin=u.PIN,original_cf48_epochs=95,snapshot_epoch=25,full95_model_complete=False,
        decision='Fixed published epoch25 plus every independently verified ready-eight addition. Retain control, ready-eight, prefix-only and all-nine arms; no score-based feature subset selection.',
        qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Known-ProbeX diagnostic. Snapshot generation completed for both phases; whole original95-epoch job is still running remotely and not claimed complete.')
    u.write(REPORT/'plan.json',fixed);sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex'],allow_pickle=False),0,6) for r in selected])
    arms=dict(control=[],ready_eight=list(range(8)),prefix_only=[8],main=list(range(9)))
    scored=algebra.screen(old,extra,y,groups,arms);records=[]
    for arm,result in scored.items():
        prediction=result['prediction'];score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(prediction,y))/len(y))
        if arm in ['control','ready_eight']:
            previous=next(r for r in old_screen['records'] if r['arm']==('control' if arm=='control' else 'main'))
            if abs(score-previous['rmse'])>1e-12:raise ValueError('Preceding ridge control changed')
        path=RUN/(arm+'-ridge.npy');np.save(path,prediction)
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(path),
            fold_rmse=[ready.extension.x.metric(prediction[groups//2==f],y[groups//2==f]) for f in range(5)]))
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=fixed['fixed_additions'],
        source_sha256=sources,script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,
        public_prediction_values_read=False,full95_model_complete=False,
        scope='Fixed published-prefix diagnostic from own95-epoch trajectory; no whole-job completion, nonlinear promotion or qualifying result.'))
    print(dict(records=records,full95_model_complete=False),flush=True)


if __name__=='__main__':main()
