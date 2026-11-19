#!/usr/bin/env python3
"""Fixed all-three whole-native raw-predictor addition to own107/all90."""
from pathlib import Path
import math
import numpy as np
import upstream_latent90_features as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-whole110-screen'
REPORT=u.ROOT/'experiments/upstream-whole110-screen'
NEW=['rbm50lat','tx20lat','tsvdx4p-1000lm']
PROOFS={name:u.ROOT/f'experiments/upstream-latent90-local/{name}-completion-verification.json' for name in NEW[:2]}
PROOFS[NEW[2]]=u.ROOT/'experiments/upstream-rank1000-completion/verification.json'

def verified_inputs():
    rows=[];sources={}
    for name in NEW:
        path=PROOFS[name];proof=u.read(path)
        expected='complete_original_two_phase_latent_model' if name in NEW[:2] else 'complete_original_two_phase_rank1000'
        if proof['state']!=expected or proof['epochs_per_phase']!=20 or not proof['both_full_epoch_traces'] or proof['qualifying_labels_read'] is not False:
            raise ValueError('Complete independently checked20/20 native model required')
        if name==NEW[2] and (proof['factors']!=1000 or proof['logical_chunks']!=16):raise ValueError('Original full rank required')
        for rel,digest in proof['bindings'].items():
            if rel in sources and sources[rel]!=digest:raise ValueError('Conflicting native lineage')
            sources[rel]=digest
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        tree=u.ROOT/('artifacts/upstream-latent90-local/source' if name in NEW[:2] else 'artifacts/upstream-parallel-mapped/source')
        pairs={s:str((tree/'preds_new'/f'{name}.{s}.npy').relative_to(u.ROOT)) for s in ['probex','qual']}
        if any(p not in proof['bindings'] for p in pairs.values()):raise ValueError('Native complete pair not bound')
        rows.append(dict(name=name,paired_files=pairs,duplicate_of=None))
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Completed native binding changed: '+name)
    return rows,sources

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing whole110 screen preserved')
    temporal.configure();parent=temporal.n.verify_prepared();rows,sources=verified_inputs()
    if parent['support_columns']!=90 or parent['prediction_columns']!=107:raise ValueError('Complete own107/all90 parent required')
    proofpath=temporal.REPORT/'cache-column-verification.json'
    if u.read(proofpath)['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json'):raise ValueError('Parent proof differs')
    for p in [proofpath,temporal.REPORT/'manifest.json',Path(__file__)]:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in parent['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
    for row in rows:
        for split,count in [('probex',1128164),('qual',2817131)]:
            p=u.ROOT/row['paired_files'][split];a=np.load(p,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Native pair shape/precision/finiteness differs')
            if u.sha(p)!=sources[str(p.relative_to(u.ROOT))]:raise ValueError('Paired source changed')
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex']),0,6) for r in rows])
    if old.shape!=(len(y),183):raise ValueError('Own107 ridge schema differs')
    RUN.mkdir();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(ready_unused=rows,fixed_additions=NEW,
        rule='Fixed control/all-three addition from complete original20/20 native runs. Original user folds and ridge penalty. Native side outputs extend the released raw-column schema; no subset search or qualifying outcomes.',qualifying_labels_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1,2]));records=[]
    for arm,result in scored.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Parent control differs')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(p),fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Known-ProbeX diagnostic on three complete own native pairs, including side outputs beyond the released raw-column list. No nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':main()
