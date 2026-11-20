#!/usr/bin/env python3
"""One separately fixed3:1 pool/tree follow-up after matched NN-family gain."""
from pathlib import Path
import math
import numpy as np
import upstream_native_nn_pool6 as pool
u=pool.u
RUN=u.ROOT/'artifacts/upstream-native-nn-pool6-tree31'
REPORT=u.ROOT/'experiments/upstream-native-nn-pool6-tree31'

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing fixed3:1 candidate preserved')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(created=pool.stamp(),weights=dict(tree=.25,nn_pool=.75),
        script_sha256=u.sha(Path(__file__)),
        rationale='The independently verified equal six-NN pool improves its matched NN control, but its original half-tree mean misses the global best. Declare ONE fixed3:1 pool-to-tree follow-up to give the improved neural family more weight. No coefficient optimization/grid/subset search or qualifying outcome access. Adaptive known-CV iteration, not untouched Test.',
        qualifying_labels_read=False,production_started=False))
    cp=pool.REPORT/'independent-canonical-verification.json';checked=u.read(cp)
    if checked['state']!='complete_independent_canonical_fixed_six_native_nn_pool' or not checked['matched_nn_improves']:
        raise ValueError('Independently improved matched NN family required')
    original=u.read(pool.REPORT/'verification.json');bindings=dict(original['source_sha256'])
    p=pool.RUN/'nn-pool.f32'
    if u.sha(p)!=next(r for r in checked['records'] if r['arm']=='nn-pool')['prediction_sha256']:
        raise ValueError('Exact canonical source pool required')
    for path in [cp,p,pool.REPORT/'verification.json',REPORT/'plan.json',Path(__file__)]:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    tree=u.ROOT/'artifacts'/pool.TREE/'tree.f32';nn=np.fromfile(p,dtype='<f4');t=np.fromfile(tree,dtype='<f4')
    y=np.load(u.FRESH/'data/probex/ratings.npy');groups=np.load(u.ROOT/'artifacts'/pool.ORIGINS[0]/'groups.npy')
    previous=u.ROOT/'experiments/upstream-item154-tree-item154-mps/verification.json';prior=u.read(previous)
    independent=u.ROOT/'experiments/upstream-item154-tree-item154-mps/independent-mean-check.json'
    if u.read(independent)['verification_sha256']!=u.sha(previous) or not prior['fixed_average_bitwise_equal']:
        raise ValueError('Canonical previous global result required')
    for path in [previous,independent,tree,u.FRESH/'data/probex/ratings.npy']:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    values=(t.astype(np.float64)*.25+nn.astype(np.float64)*.75).astype('<f4')
    path=RUN/'fixed-tree-pool.f32';values.tofile(path)
    def score(a,b):return math.sqrt(math.fsum((float(x)-float(z))**2 for x,z in zip(np.clip(a.astype(np.float64),1.05,4.95),b))/len(b))
    rmse=score(values,y)
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original source/model evidence changed')
    u.write(REPORT/'verification.json',dict(state='complete_fixed_tree25_pool75_development',observed=pool.stamp(),
        rmse=rmse,previous_best_rmse=prior['rmse'],improves_previous=rmse<prior['rmse'],
        rows=len(y),weights=dict(tree=.25,nn_pool=.75),prediction_sha256=u.sha(path),source_sha256=bindings,
        fold_rmse=[score(values[groups//2==f],y[groups//2==f]) for f in range(5)],
        actual_origin_model_and_replay_proofs_bound=True,qualifying_labels_read=False,production_started=False,goal_complete=False))
    print(rmse,prior['rmse'],rmse-prior['rmse'],flush=True)

if __name__=='__main__':main()
