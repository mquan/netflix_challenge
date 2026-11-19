#!/usr/bin/env python3
"""Known-only fixed batch of every unused captured CF48/96 snapshot pair."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import argparse
import math
import time
import numpy as np
import upstream_repro as u
import upstream_latent90_features as temporal
import upstream_remote_batch as algebra
import upstream_latent90_local as resources

RUN=u.ROOT/'artifacts/upstream-warm59-screen'
REPORT=u.ROOT/'experiments/upstream-warm59-screen'
PACKAGES=[('cfnade-48','artifacts/remote9h-transfer/cfnade48-package','preds_preset48',25,30),
          ('cfnade-96','artifacts/remote9h-cf96-transfer/cfnade96-package','preds_new',10,31)]

def inputs():
    rows=[];sources={}
    cf48=u.read(u.ROOT/'experiments/upstream-remote-cf48-final/paired-receipt.json')
    if (cf48['state']!='verified_complete_resumed_remote_prediction_intake' or not cf48['full95_model_complete'] or
        not cf48['reconstructed_native_terminal_and_epoch95_sha256_match'] or cf48['qualifying_labels_read'] is not False):
        raise ValueError('Verified completed own CF48 trajectory required')
    cf96=u.read(u.ROOT/'experiments/upstream-remote9h-cf96/local-preflight-verification.json')
    for name,bundle_rel,preds,excluded,last in PACKAGES:
        bundle=u.ROOT/bundle_rel;mp=bundle/'manifest.json';m=u.read(mp)
        if m['source_pin']!=u.PIN or m['qualifying_labels_read'] is not False:
            raise ValueError('Original own source lineage required')
        if name=='cfnade-48':
            package=u.read(u.ROOT/'experiments/upstream-remote9h/cfnade48-package.json')
            if (package['manifest_sha256']!=u.sha(mp) or package['source_id']!=cf48['source_id'] or
                m['source_id']!=cf48['source_id'] or m['data_id']!=cf48['data_id']):raise ValueError('Own CF48 package differs')
            warm_scope='Captured complete snapshot of own completed95-epoch trajectory; coordinator received final vector, not all final tensors.'
        else:
            if (cf96['state']!='verified_local_transfer_preflight' or cf96['manifest_sha256']!=u.sha(mp) or
                not cf96['all84_tensors_per_phase_verified'] or not cf96['native_original_source_parity_passed']):
                raise ValueError('Previously independently checked own CF96 package required')
            warm_scope='Captured complete earlier snapshot pair; original full95-epoch job remains incomplete and ineligible as a completed95-epoch model.'
        if min(v['epoch'] for v in m['phases'].values())!=last:raise ValueError('Fixed captured horizon changed')
        for epoch in range(1,last+1):
            if epoch==excluded:continue
            pairs={}
            for phase,split,count in [('trainx','probex',1128164),('fulltrain','qual',2817131)]:
                filename=f'{name}_ep{epoch:02}.{split}.npy';rel=f'native-source/{preds}/{filename}';path=bundle/rel
                digest=m['phases'][phase]['prediction_sha256'][filename]
                if m['files'][rel]!=digest or u.sha(path)!=digest:raise ValueError('Captured snapshot changed')
                a=np.load(path,mmap_mode='r',allow_pickle=False)
                if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid captured pair')
                pairs[split]=str(path.relative_to(u.ROOT));sources[pairs[split]]=digest
            rows.append(dict(name=f'{name}_ep{epoch:02}',paired_files=pairs,duplicate_of=None,scope=warm_scope))
        sources[str(mp.relative_to(u.ROOT))]=u.sha(mp)
    if len(rows)!=59:raise ValueError('Fixed all59 batch required')
    digests=[tuple(sources[r['paired_files'][s]] for s in ['probex','qual']) for r in rows]
    if len(set(digests))!=59:raise ValueError('Exact duplicate captured pairs require explicit inventory review')
    for rel in ['experiments/upstream-remote-cf48-final/paired-receipt.json',
                'experiments/upstream-remote9h/cfnade48-package.json',
                'experiments/upstream-remote9h-cf96/local-preflight-verification.json',
                'experiments/upstream-cf96-prefix10-screen/paired-receipt.json']:
        sources[rel]=u.sha(u.ROOT/rel)
    return rows,sources

def declare():
    if (REPORT/'plan.json').exists():raise ValueError('Existing declaration preserved')
    rows,sources=inputs();REPORT.mkdir(parents=True,exist_ok=True)
    u.write(REPORT/'plan.json',dict(created=resources.stamp(),ready_unused=rows,fixed_additions=[r['name'] for r in rows],
        source_sha256=sources,excluded_existing=['cfnade-48_ep25','cfnade-96_ep10'],
        rule='All unused paired snapshots in the immutable pre-existing own packages, in epoch order. No epoch search or qualifying outcomes. Fixed control/all59 arms only.',
        whole_cf96_95_completion_claimed=False,exact_published_raw_schema_claimed=False,
        qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Own earlier trajectory snapshots extend the released raw-column list. Complete snapshot pairs are distinct from full95-epoch preset completion. No new inference replay or qualifying score claimed.'))

def run():
    if RUN.exists() or (REPORT/'ridge-screen.json').exists():raise ValueError('Existing partial outputs preserved')
    plan=u.read(REPORT/'plan.json');rows,sources=inputs()
    if plan['ready_unused']!=rows or plan['source_sha256']!=sources:raise ValueError('Declared available batch changed')
    previous=resources.memory.memory();good=0
    while good<2:
        current=resources.memory.memory();workers=resources.footprints()
        good=good+1 if (sum(v['bytes'] for v in workers.values())<19*2**30 and current['compressor_bytes']<8*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20) else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_bounded_ridge_diagnostic_memory',observed=resources.stamp(),
            workers=workers,memory=current,good_samples=good,
            scope='Small streamed ridge diagnostic, no NN/tree fit. Added matrix266MiB; bounded8192-row algebra blocks.'))
        if good<2:previous=current;time.sleep(30)
    mp=temporal.REPORT/'manifest.json';m=u.read(mp);cp=temporal.REPORT/'cache-column-verification.json'
    if u.read(cp)['manifest_sha256']!=u.sha(mp) or m['prediction_columns']!=107:raise ValueError('Verified own107/all90 control required')
    for name in ['design.npy','labels.npy','users.npy','groups.npy']:
        path=temporal.RUN/name
        if u.sha(path)!=m['prepared_sha256'][name]:raise ValueError('Control identity/design changed')
        sources[str(path.relative_to(u.ROOT))]=m['prepared_sha256'][name]
    for p in [mp,cp,REPORT/'plan.json',Path(__file__),Path(algebra.__file__),u.ROOT/'scripts/verify_upstream_warm59_screen.py']:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.empty((len(y),59),dtype=np.float32)
    for j,row in enumerate(rows):np.clip(np.load(u.ROOT/row['paired_files']['probex'],mmap_mode='r'),0,6,out=extra[:,j])
    RUN.mkdir();u.write(REPORT/'status.json',dict(state='streamed_ridge_diagnostic_running',observed=resources.stamp()))
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=list(range(59))));records=[]
    for arm,result in scored.items():
        path=RUN/(arm+'-ridge.npy');np.save(path,result['prediction'])
        value=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(value-m['baseline_rmse'])>1e-12:raise ValueError('Control differs')
        records.append(dict(arm=arm,rmse=value,prediction_sha256=u.sha(path),
            fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=plan['fixed_additions'],source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Fixed all59 captured own snapshot addition to own107, original five user folds and ridge penalty. Diagnostic only, not nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declare',action='store_true');a=p.parse_args()
    declare() if a.declare else run()
