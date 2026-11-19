#!/usr/bin/env python3
"""Check fixed raw snapshot lineage and actual replay evidence; score known rows only."""
import argparse
import math
from pathlib import Path
import numpy as np
import upstream_mps_autorec as model

u=model.u

def verify(epoch):
    import torch
    torch.set_num_threads(2);plan=model.verify_plan();pp=model.REPORT/f'pair-{epoch:05}.json';pair=u.read(pp)
    if (epoch not in model.CONFIG['snapshot_epochs'] or pair['epoch']!=epoch or
        pair['state']!='complete_own_raw_snapshot_pair_with_actual_saved_model_replays' or
        pair['plan_sha256']!=u.sha(model.REPORT/'plan.json') or pair['qualifying_labels_read'] is not False):
        raise ValueError('Fixed complete paired snapshot required')
    bindings={str(pp.relative_to(u.ROOT)):u.sha(pp)};counts={}
    for phase,(query,_,rows) in model.data.PHASES.items():
        rp=model.REPORT/f'{phase}-snapshot-{epoch:05}.json';r=u.read(rp);root=model.RUN/phase/'snapshots'/f'epoch-{epoch:05}'
        if (not r['actual_saved_model_replay_bitwise'] or r['plan_sha256']!=pair['plan_sha256'] or
            r['epoch']!=epoch or r['phase']!=phase or r['rows']!=rows or pair['receipts'][phase]!=u.sha(rp) or
            u.sha(root/'model.pt')!=r['model_sha256'] or u.sha(root/'prediction.npy')!=r['prediction_sha256']):
            raise ValueError('Actual source model/query replay binding differs')
        state=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
        if state['config']!=model.CONFIG or state['phase']!=phase or state['epoch']!=epoch or state['plan_sha256']!=pair['plan_sha256']:
            raise ValueError('Actual snapshot configuration differs')
        if not all(torch.isfinite(v).all() for v in state['model'].values()):raise ValueError('Nonfinite saved model')
        values=np.load(root/'prediction.npy',mmap_mode='r')
        if values.shape!=(rows,) or values.dtype!=np.float32 or not np.isfinite(values).all() or np.any((values<1)|(values>5)):
            raise ValueError('Full query shape/finiteness/range differs')
        if pair['pairs'][query]!=str((root/'prediction.npy').relative_to(u.ROOT)):raise ValueError('Query path differs')
        for path in [rp,root/'model.pt',root/'prediction.npy']:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
        counts[query]=rows
    prediction=np.load(u.ROOT/pair['pairs']['probex']);yp=u.FRESH/'data/probex/ratings.npy';y=np.load(yp)
    rmse=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(prediction,y))/len(y))
    bindings[str(yp.relative_to(u.ROOT))]=u.sha(yp)
    for path in [Path(__file__),model.REPORT/'plan.json']:bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Snapshot evidence changed during verification')
    result=dict(state='complete_independent_raw_snapshot_bindings_and_known_score',observed=model.stamp(),epoch=epoch,
        known_component_rmse=rmse,counts=counts,bindings=bindings,actual_original_saved_query_replay_evidence_verified=True,
        independently_verified_csr_histories_bound=True,full50_epoch_model_complete=pair['full50_epoch_model_complete'],
        qualifying_labels_read=False,qualifying_accuracy_claimed=False,
        scope='Separate model/config/source/query hashes and actual saved-query replay evidence check, plus compensated known-only single-component score. This checker does not rerun inference or claim nonlinear/global/qualifying improvement. Original declared50/50 training continues.')
    u.write(model.REPORT/f'snapshot-{epoch:05}-verification.json',result);print(dict(epoch=epoch,known_component_rmse=rmse,counts=counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);verify(p.parse_args().epoch)
