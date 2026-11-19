#!/usr/bin/env python3
"""Fixed top32 unsupervised item components of our complete rank1000 native model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import datetime as dt
import numpy as np
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-rank1000-item32'
REPORT=u.ROOT/'experiments/upstream-rank1000-item32'
RANK=32
FACTOR=u.ROOT/'artifacts/upstream-parallel-mapped/source/preds_new/tsvdx4p-1000lm.ifeat.trainx.npy'

def inputs():
    path=u.ROOT/'experiments/upstream-rank1000-completion/verification.json';proof=u.read(path)
    if (proof['state']!='complete_original_two_phase_rank1000' or proof['source_pin']!=u.PIN or
        proof['factors']!=1000 or proof['epochs_per_phase']!=20 or not proof['both_full_epoch_traces'] or
        not proof['unchanged_original_data_aliases_verified'] or not proof['qualifying_targets_zero'] or
        proof['qualifying_labels_read'] is not False or proof['public_prediction_values_read'] is not False):
        raise ValueError('Complete original own rank1000 phases and source checks required')
    sources=dict(proof['bindings']);sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    if str(FACTOR.relative_to(u.ROOT)) not in sources:raise ValueError('Actual fresh factor is not source-bound')
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original own rank1000 binding changed')
    a=np.load(FACTOR,allow_pickle=False)
    if a.shape!=(17770,1000) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Factor schema differs')
    return a.astype(np.float64),sources

def run():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing item32 evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),rank=RANK,
        script_sha256=u.sha(Path(__file__)),rule='Fixed top32 centered covariance components of the own complete20/20 rank1000 trainx item factor table. Canonical maximum-absolute-loading signs, unit population variance. Same trainx projection for development/qualifying. No outcomes or component-count search. Query contexts are gathered virtually from canonical item identities.',qualifying_labels_read=False,public_numerical_features_read=False))
    a,sources=inputs();center=a.mean(axis=0);x=a-center;cov=x.T@x/len(x)
    spectrum,basis=np.linalg.eigh(cov);order=np.argsort(spectrum)[::-1][:RANK];values=spectrum[order];vectors=basis[:,order]
    if values.min()<=0:raise ValueError('Positive top spectrum required')
    vectors*=np.sign(vectors[np.abs(vectors).argmax(axis=0),np.arange(RANK)]);scales=np.sqrt(values)
    table=((x@vectors)/scales).astype(np.float32)
    RUN.mkdir();np.save(RUN/'item32.npy',table);np.savez(RUN/'projection.npz',center=center,vectors=vectors,scales=scales,eigenvalues=values)
    for p in [Path(__file__),REPORT/'plan.json']:sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    u.write(REPORT/'receipt.json',dict(state='complete_own_rank1000_item32_projection',source_sha256=sources,
        output_sha256={p.name:u.sha(p) for p in RUN.iterdir()},explained_variance_fraction=float(values.sum()/spectrum.sum()),
        support_columns=32,qualifying_labels_read=False,public_numerical_features_read=False,accuracy_claimed=False))

if __name__=='__main__':run()
