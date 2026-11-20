#!/usr/bin/env python3
"""Fixed original KNN3 terminal addition to own107/item154."""
from pathlib import Path
import math
import numpy as np
import upstream_item154_features as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-knn3pair2-screen'
REPORT=u.ROOT/'experiments/upstream-knn3pair2-screen'
NEW=['s3-dnn32nm__knn3','s3-nade12s__knn3']
PROOFS={name:u.ROOT/f'experiments/upstream-split3-knn3-chains/{name}-verification.json' for name in NEW}
PARENTS={'s3-dnn32nm__knn3':'s3-dnn32nm','s3-nade12s__knn3':'s3-nade12s'}

def verified_inputs():
    found={};sources={}
    for name,path in PROOFS.items():
        proof=u.read(path)
        if (proof['state']!='complete_original_two_phase_knn3_chain' or proof['source_pin']!=u.PIN or
            proof['job']!=name or proof['parent']!=PARENTS[name] or
            not proof['both_full_native_neighbor_builds_verified'] or not proof['original_default_configuration_verified'] or
            not proof['complete_parent_terminal_inputs_bound'] or
            not proof['full_known_projection_recomputed'] or not proof['known_training_disjointness_bound_to_unchanged_original_data'] or
            not proof['qualifying_targets_zero'] or proof['qualifying_labels_read'] is not False or proof['public_prediction_values_read'] is not False):
            raise ValueError('Complete original KNN3 builds, parents, configuration and every-row projection proof required')
        for rel,digest in proof['bindings'].items():
            if rel in sources and sources[rel]!=digest:raise ValueError('Conflicting original native lineage')
            sources[rel]=digest
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        for pair in proof['projected_pairs']:
            if pair['name'] in found:raise ValueError('Duplicate published native pair')
            paths={q:pair[q] for q in ['probex','qual']}
            for query,pathstr in paths.items():
                if proof['bindings'].get(pathstr)!=pair[query+'_sha256']:raise ValueError('Original native query binding differs')
            found[pair['name']]=dict(name=pair['name'],paired_files=paths,duplicate_of=None)
    if set(found)!=set(NEW):raise ValueError('Exactly two original KNN3 pairs required')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original completed native binding changed')
    return [found[name] for name in NEW],sources

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing KNN3 pair screen preserved')
    temporal.configure();parent=temporal.n.verify_prepared();rows,sources=verified_inputs()
    if parent['support_columns']!=154 or parent['prediction_columns']!=107:raise ValueError('Complete own107/item154 parent required')
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
        rule='Fixed control and both predeclared original KNN3 terminal pairs after both complete original native neighbor builds for each chain. Original user folds and ridge penalty. Native side outputs extend the released raw-column schema; no subset search or qualifying outcomes.',qualifying_labels_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]));records=[]
    for arm,result in scored.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Parent control differs')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(p),fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Known-ProbeX diagnostic on both fixed own KNN3 pairs, both exact declared published outputs. No nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':main()
