#!/usr/bin/env python3
"""Fixed equal pool of all six completed native-append own107/item154 MPS NNs."""
from pathlib import Path
import datetime as dt
import math
import numpy as np
import upstream_repro as u
RUN=u.ROOT/'artifacts/upstream-native-nn-pool6'
REPORT=u.ROOT/'experiments/upstream-native-nn-pool6'
ORIGINS=['upstream-item154-mps-context','upstream-split3-virtual-mps-context',
    'upstream-attention2-virtual-mps-context','upstream-knn3pair2-compact-mps-context',
    'upstream-split3eight-compact-mps-context','upstream-cf96-prefix2030-compact-mps-context']
TREE='upstream-item154-tree'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing fixed pool evidence preserved')
    REPORT.mkdir();RUN.mkdir()
    declaration=dict(created=stamp(),origins=ORIGINS,tree=TREE,neural_weights=[1/6]*6,
        tree_pool_weights=[.5,.5],script_sha256=u.sha(Path(__file__)),
        reference_development_rmse=.8592012938198756,
        rule='All six completed MPS H128/rank32 neural variants retaining own107 and exactly154 context columns, with only original-native raw prediction additions. Base,four split3,attention2,KNN2,all-eight split3,CF96prefix20/30. Equal six-NN pool, castfloat32, then fixed half with existingtree154. No subset/weight search, new source/model fitting or qualifying outcomes. Individual negative results remain preserved. CF96 whole95/95 source remains incomplete; only independently verified own prefixes are used.',
        qualifying_labels_read=False,automatic_production=False)
    u.write(REPORT/'plan.json',declaration)
    bindings={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),str((REPORT/'plan.json').relative_to(u.ROOT)):u.sha(REPORT/'plan.json')}
    identity=None;vectors=[]
    for family,name in [('nn',name) for name in ORIGINS]+[('tree',TREE)]:
        report=u.ROOT/'experiments'/name;cache=u.ROOT/'artifacts'/name
        mp=report/'manifest.json';m=u.read(mp);ep=report/'evaluation.json';vp=report/'independent-verification.json';v=u.read(vp)
        if (m['source_commit']!=u.PIN or v['state']!='complete' or v['evaluation_sha256']!=u.sha(ep) or
            not v['fixed_average_bitwise_equal'] or m['qualifying_labels_read'] is not False):
            raise ValueError('Complete original own known-only source proof required')
        if family=='nn' and (m['total_learned_support_columns']!=154 or m['config']['nn_hidden']!=128 or m['config']['context_rank']!=32 or m['config']['neural_backend']!='torch-mps-float32'):
            raise ValueError('Fixed full154-context MPS family eligibility differs')
        for name,digest in m['code_sha256'].items():bindings[name]=digest
        replay=report/(family+'-replay.json');r=u.read(replay)
        if [q['fold'] for q in r['records']]!=list(range(5)) or not all(q['bitwise_equal'] for q in r['records']):
            raise ValueError('All five actual saved-model replays required')
        for p in [mp,ep,vp,replay]:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
        arrays={k:np.load(cache/(k+'.npy'),allow_pickle=False) for k in ['labels','users','groups']}
        if identity is None:identity=arrays
        elif any(not np.array_equal(identity[k],arrays[k]) for k in arrays):raise ValueError('Source user/row partitions differ')
        for key in arrays:
            p=cache/(key+'.npy');bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
            if m['prepared_sha256'][p.name]!=u.sha(p):raise ValueError('Original row identity binding differs')
        for fold in range(5):
            rp=report/family/f'{fold}.json';receipt=u.read(rp)
            if receipt['manifest_sha256']!=u.sha(mp) or receipt['fold']!=fold or receipt['family']!=family:
                raise ValueError('Actual originating full-fold receipt required')
            bindings[str(rp.relative_to(u.ROOT))]=u.sha(rp)
            for filename,digest in receipt['outputs'].items():
                p=cache/family/str(fold)/filename
                if u.sha(p)!=digest:raise ValueError('Actual saved model/fold prediction changed')
                bindings[str(p.relative_to(u.ROOT))]=digest
        p=cache/(family+'.f32');values=np.fromfile(p,dtype='<f4')
        checked=next(q for q in v['records'] if q['arm']==family)
        if values.shape!=(1128164,) or not np.isfinite(values).all() or checked['prediction_sha256']!=u.sha(p):raise ValueError('Bound full vector differs')
        for fold in range(5):
            actual=np.fromfile(cache/family/str(fold)/'prediction.f32',dtype='<f4')
            if not np.array_equal(actual.view(np.uint32),values[arrays['groups']//2==fold].view(np.uint32)):
                raise ValueError('Assembled vector differs from actual original fold outputs')
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p);vectors.append(values)
    labels=np.load(u.FRESH/'data/probex/ratings.npy');users=np.load(u.FRESH/'data/probex/user_idxs.npy')
    if not np.array_equal(labels,identity['labels']) or not np.array_equal(users,identity['users']):raise ValueError('Canonical identities differ')
    _,first,inverse=np.unique(users,return_index=True,return_inverse=True)
    folds=identity['groups']//2
    if not np.array_equal(folds,folds[first][inverse]):raise ValueError('User crosses outer folds')
    pooled=(sum(v.astype(np.float64) for v in vectors[:6])/6).astype('<f4')
    hybrid=((vectors[-1].astype(np.float64)+pooled.astype(np.float64))*.5).astype('<f4')
    records=[]
    for arm,values in [('nn-pool',pooled),('fixed-half-tree-pool',hybrid)]:
        path=RUN/(arm+'.f32');values.tofile(path)
        rmse=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(values,labels))/len(labels))
        records.append(dict(arm=arm,rmse=rmse,prediction_sha256=u.sha(path),
            fold_rmse=[math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(values[folds==f],labels[folds==f]))/int((folds==f).sum())) for f in range(5)]))
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Bound original source evidence changed')
    u.write(REPORT/'verification.json',dict(state='complete_fixed_six_native_nn_pool_development',observed=stamp(),
        records=records,source_sha256=bindings,origins=ORIGINS,tree=TREE,neural_weights=[1/6]*6,
        tree_pool_weights=[.5,.5],rows=len(labels),users=len(first),users_disjoint=True,
        all35_origin_fold_receipts_and_actual_models_bound=True,all35_original_family_replay_rows_verified=True,
        reference_development_rmse=declaration['reference_development_rmse'],
        improves_reference=records[1]['rmse']<declaration['reference_development_rmse'],
        qualifying_labels_read=False,production_started=False,goal_complete=False))
    print(records,flush=True)

if __name__=='__main__':main()
