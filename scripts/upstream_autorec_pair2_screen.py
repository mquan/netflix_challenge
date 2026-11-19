#!/usr/bin/env python3
"""Fixed known-only addition of own raw autoencoder snapshots5 and10."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import fcntl
import math
import subprocess
import sys
import time
import numpy as np
import upstream_latent90_features as temporal
import upstream_remote_batch as algebra
import upstream_mps_autorec as raw

u=raw.u
RUN=u.ROOT/'artifacts/upstream-autorec-pair2-screen'
REPORT=u.ROOT/'experiments/upstream-autorec-pair2-screen'
EPOCHS=[5,10]
NEW=[f'mps-autorec_ep{e:02}' for e in EPOCHS]

def selected():
    return [dict(name=name,duplicate_of=None,paired_files={query:str((raw.RUN/phase/'snapshots'/f'epoch-{epoch:05}'/'prediction.npy').relative_to(u.ROOT))
        for phase,(query,_,_) in raw.data.PHASES.items()}) for epoch,name in zip(EPOCHS,NEW)]

def verified_inputs():
    raw.verify_plan();rows=selected();sources={}
    for epoch,row in zip(EPOCHS,rows):
        path=raw.REPORT/f'snapshot-{epoch:05}-verification.json';proof=u.read(path)
        pairpath=raw.REPORT/f'pair-{epoch:05}.json';pair=u.read(pairpath)
        if (proof['state']!='complete_independent_raw_snapshot_bindings_and_known_score' or proof['epoch']!=epoch or
            not proof['actual_original_saved_query_replay_evidence_verified'] or proof['qualifying_labels_read'] is not False or
            pair['pairs']!=row['paired_files'] or pair['epoch']!=epoch):raise ValueError('Verified complete fixed snapshot pairs required')
        for name,digest in proof['bindings'].items():
            if name in sources and sources[name]!=digest:raise ValueError('Conflicting raw source lineage')
            sources[name]=digest
        for p in [path,pairpath,raw.REPORT/'plan.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Raw snapshot source changed')
    return rows,sources

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (REPORT/'plan.json').exists() or RUN.exists():raise ValueError('Existing pair2 evidence preserved')
        rows=selected()
        u.write(REPORT/'plan.json',dict(created=raw.stamp(),ready_unused=rows,fixed_additions=NEW,weights_selected_from_qualifying=False,
            rule='Fixed control/all-two first declared raw snapshots5/10, independent of their individual scores. Original known user folds and ridge penalty1000. No subset or qualifying-based selection.',
            complete_snapshot_pairs_required=True,full50_model_completion_claimed=False,qualifying_labels_read=False,script_sha256=u.sha(Path(__file__))))
        while not all((raw.REPORT/f'pair-{epoch:05}.json').exists() for epoch in EPOCHS):
            u.write(REPORT/'status.json',dict(state='waiting_for_two_complete_actual_replayed_snapshot_pairs',observed=raw.stamp()));time.sleep(15)
        for epoch in EPOCHS:
            proof=raw.REPORT/f'snapshot-{epoch:05}-verification.json'
            if not proof.exists():
                with (REPORT/f'raw-{epoch:05}-verification.log').open('x') as log:
                    subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_mps_autorec_snapshot.py'),'--epoch',str(epoch)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (u.ROOT/'experiments/upstream-source-extension/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX)
            temporal.configure();m=temporal.n.verify_prepared();rows,sources=verified_inputs()
            for p in [Path(__file__),REPORT/'plan.json',temporal.REPORT/'manifest.json',temporal.REPORT/'cache-column-verification.json',u.ROOT/'scripts/verify_upstream_autorec_pair2_screen.py']:
                sources[str(p.relative_to(u.ROOT))]=u.sha(p)
            for name,digest in m['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
            y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy');old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
            extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex']),0,6) for r in rows])
            RUN.mkdir();u.write(REPORT/'status.json',dict(state='fixed_all_two_known_ridge_diagnostic',observed=raw.stamp()))
            scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]));records=[]
            for arm,result in scored.items():
                path=RUN/(arm+'-ridge.npy');np.save(path,result['prediction'])
                score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
                if arm=='control' and abs(score-m['baseline_rmse'])>1e-12:raise ValueError('Original ridge control differs')
                records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(path),fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
            np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
            u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,script_sha256=u.sha(Path(__file__)),
                qualifying_labels_read=False,public_prediction_values_read=False,scope='Fixed own raw snapshot pair addition to own107. Full original50/50 trajectory continues. Known-only ridge diagnostic, not nonlinear or qualifying promotion.'))
            with (REPORT/'independent-ridge.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_autorec_pair2_screen.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            u.write(REPORT/'status.json',dict(state='complete_ridge_diagnostic_and_independent_replay',observed=raw.stamp(),records=records));print(records,flush=True)

if __name__=='__main__':run()
