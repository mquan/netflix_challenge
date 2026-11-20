#!/usr/bin/env python3
"""Evaluate the separately predeclared known-only half mean of own107/50 tree and KNN3 virtual NN."""
import datetime as dt
import math
from pathlib import Path
import numpy as np
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-rbm50-tree-cf96-prefix2030-compact-mps'
RUN=u.ROOT/'artifacts/upstream-rbm50-tree-cf96-prefix2030-compact-mps'
ORIGINS=[('tree','upstream-rbm50-tree'),('nn','upstream-cf96-prefix2030-compact-mps-context')]


def evaluate():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing candidate evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
        script_sha256=u.sha(Path(__file__)),origins=ORIGINS,weights=[.5,.5],
        selection='Fixed known-only comparison declared in upstream-cf96-prefix2030-compact-queue/plan.json before this neural study started. Actual saved-model replays are required before evaluation. No weight search, no qualifying outcomes, no new source/NN fitting. Both existing families must preserve identical held-user row assignments.',
        production='No automatic production; requires a separately frozen heterogeneous-family production recipe if better.'))
    declaration=u.ROOT/'experiments/upstream-cf96-prefix2030-compact-queue/plan.json'
    if u.read(declaration)['weights']!=[.5,.5]:raise ValueError('Prior fixed mean declaration required')
    sources={str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),str(declaration.relative_to(u.ROOT)):u.sha(declaration)};families={};identity=None
    for family,name in ORIGINS:
        report=u.ROOT/'experiments'/name;cache=u.ROOT/'artifacts'/name
        manifest_path=report/'manifest.json';manifest=u.read(manifest_path)
        evaluation=report/'evaluation.json';proof_path=report/'independent-verification.json';proof=u.read(proof_path)
        if (manifest['source_commit']!=u.PIN or proof['state']!='complete' or
                proof['evaluation_sha256']!=u.sha(evaluation) or not proof['fixed_average_bitwise_equal']):
            raise ValueError('Verified own full development result required')
        for relative,digest in manifest['code_sha256'].items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Originating source code changed')
            sources[relative]=digest
        arrays={}
        for key in ['labels','users','groups']:
            path=cache/(key+'.npy')
            if u.sha(path)!=manifest['prepared_sha256'][path.name]:raise ValueError('Row identity changed')
            arrays[key]=np.load(path,allow_pickle=False);sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        if identity is None:identity=arrays
        elif any(not np.array_equal(identity[k],arrays[k]) for k in arrays):
            raise ValueError('Held-user row identities differ between families')
        replay_path=report/(family+'-replay.json');replay=u.read(replay_path)
        if ([r['fold'] for r in replay['records']]!=list(range(5)) or
                not all(r['bitwise_equal'] for r in replay['records'])):raise ValueError('Saved-model replay incomplete')
        for path in [manifest_path,evaluation,proof_path,replay_path]:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        row=next(r for r in u.read(evaluation)['records'] if r['arm']==family)
        checked=next(r for r in proof['records'] if r['arm']==family)
        path=cache/(family+'.f32');digest=u.sha(path)
        if digest!=row['prediction_sha256'] or digest!=checked['prediction_sha256']:
            raise ValueError('Originating prediction binding changed')
        values=np.fromfile(path,dtype='<f4');sources[str(path.relative_to(u.ROOT))]=digest
        if values.shape!=arrays['labels'].shape or not np.isfinite(values).all():raise ValueError('Invalid prediction vector')
        for fold in range(5):
            path=report/family/f'{fold}.json';receipt=u.read(path)
            if receipt['manifest_sha256']!=u.sha(manifest_path) or receipt['fold']!=fold or receipt['family']!=family:
                raise ValueError('Actual original fold receipt required')
            sources[str(path.relative_to(u.ROOT))]=u.sha(path)
            for filename,digest in receipt['outputs'].items():
                artifact=cache/family/str(fold)/filename
                if u.sha(artifact)!=digest:raise ValueError('Actual originating model changed')
                sources[str(artifact.relative_to(u.ROOT))]=digest
            actual=np.fromfile(cache/family/str(fold)/'prediction.f32',dtype='<f4')
            if not np.array_equal(actual,values[arrays['groups']//2==fold]):
                raise ValueError('Assembled vector differs from actual fold outputs')
        families[family]=values.astype(np.float64)
    native=u.FRESH/'data/probex'
    for key,name in [('labels','ratings.npy'),('users','user_idxs.npy')]:
        if not np.array_equal(identity[key],np.load(native/name,allow_pickle=False)):
            raise ValueError('Development identity differs from original known ProbeX')
        sources[str((native/name).relative_to(u.ROOT))]=u.sha(native/name)
    previous=u.ROOT/'experiments/upstream-item154-tree-item154-mps/verification.json'
    prior=u.read(previous)
    prior_vector=u.ROOT/'artifacts/upstream-item154-tree-item154-mps/fixed-half-tree-nn.f32'
    if (prior['state']!='complete_verified_fixed_combination' or not prior['fixed_average_bitwise_equal'] or
        u.sha(prior_vector)!=prior['prediction_sha256']):
        raise ValueError('Previous fixed-combination best not verified')
    values=np.fromfile(prior_vector,dtype='<f4').astype(np.float64)
    if values.shape!=identity['labels'].shape:raise ValueError('Previous population differs')
    best=math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(np.clip(values,1.05,4.95),identity['labels']))/len(values))
    if abs(best-prior['rmse'])>1e-12:raise ValueError('Previous complete score differs')
    for path in [previous,prior_vector,REPORT/'plan.json']:sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    RUN.mkdir();prediction=((families['tree']+families['nn'])/2).astype('<f4')
    path=RUN/'fixed-half-tree-nn.f32';prediction.tofile(path)
    stored=np.fromfile(path,dtype='<f4').astype(np.float64)
    if not np.array_equal(stored.astype('<f4'),prediction):raise ValueError('Stored mean differs')
    labels=identity['labels'];folds=identity['groups']//2
    def score(p,y):return math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(np.clip(p,1.05,4.95),y))/len(y))
    value=score(stored,labels)
    for relative,digest in sources.items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Candidate evidence changed during evaluation')
    result=dict(state='complete_verified_fixed_combination',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
        rmse=value,previous_best_rmse=best,delta_vs_previous_best=value-best,improves_previous=value<best,
        rows=len(labels),users=len(np.unique(identity['users'])),fold_rmse=[score(stored[folds==f],labels[folds==f]) for f in range(5)],
        origins=ORIGINS,weights=[.5,.5],source_sha256=sources,prediction_sha256=u.sha(path),
        all_ten_original_fold_receipts_and_models_rehashed=True,original_saved_model_replays_verified=True,
        fixed_average_bitwise_equal=True,qualifying_labels_read=False,goal_complete=False,
        scope='Adaptive known-ProbeX candidate. Fixed half mean of own107/50 tree and own109/item154 MPS context NN, using their actual original nested held-user predictions. Distinct family schemas/recipes remain explicit. No fresh fitting receipts, production predictions, qualifying score or official Test claim.')
    u.write(REPORT/'verification.json',result)
    print({k:result[k] for k in ['rmse','previous_best_rmse','delta_vs_previous_best','improves_previous','rows']})


if __name__=='__main__':evaluate()
