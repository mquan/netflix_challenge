#!/usr/bin/env python3
"""Reopen all seven preserved pipelines before the deferred numerical recovery."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import numpy as np
import upstream_split3_matched_mps as study

u=study.u
REPORT=u.ROOT/'experiments/upstream-split3-preserved-pipelines'

def main():
    import torch
    if REPORT.exists():raise ValueError('Existing actual saved-state preflight retained')
    REPORT.mkdir();torch.set_num_threads(2)
    manifest=study.REPORT/'manifest.json';frozen=u.read(manifest)
    if frozen['code_sha256']!=study.code_hashes() or frozen['config']!=study.CONFIG:raise ValueError('Original numerical recipe changed')
    mapping=np.load(study.data.RUN/'mapping.npy');groups=np.load(study.data.RUN/'groups.npy')[mapping]
    records=[]
    for arm,folds in [('control',range(5)),('expanded',range(2))]:
        for fold in folds:
            receipt=study.REPORT/arm/f'{fold}.json';r=u.read(receipt);root=study.RUN/arm/str(fold);counts=r['chosen_iterations']
            if r['manifest_sha256']!=u.sha(manifest) or len(counts)!=3 or any(type(c)!=int or not 0<=c<=40 for c in counts):raise ValueError('Actual original receipt seed budgets differ')
            expected={'ridge.npz','scalers.npz','prediction.f32'}|{f'nn-{seed}.pt' for seed,c in zip(study.CONFIG['seeds'],counts) if c}
            if set(r['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual complete original pipeline inventory differs')
            for name,h in r['outputs'].items():
                if u.sha(root/name)!=h:raise ValueError('Actual completed pipeline byte hash differs')
            for name in ['ridge.npz','scalers.npz']:
                with np.load(root/name) as saved:
                    if not all(np.isfinite(saved[k]).all() for k in saved.files):raise ValueError('Serialized original preprocessing not finite')
            reopened=0
            for seed,count in zip(study.CONFIG['seeds'],counts):
                if not count:continue
                saved=torch.load(root/f'nn-{seed}.pt',map_location='cpu',weights_only=True)
                if (saved['config']!=study.CONFIG or saved['feature_width']!=210 or saved['seed']!=seed+fold or saved['epochs']!=count):raise ValueError('Actual original saved model configuration differs')
                model=study.model_factory(210,study.CONFIG);model.load_state_dict(saved['state'],strict=True)
                if not all(torch.isfinite(t).all().item() for t in saved['state'].values()):raise ValueError('Actual saved model tensors not finite')
                del saved,model;reopened+=1
            prediction=np.fromfile(root/'prediction.f32',dtype='<f4')
            if prediction.shape!=(int((groups//2==fold).sum()),) or not np.isfinite(prediction).all():raise ValueError('Complete original canonical held vector absent')
            records.append(dict(arm=arm,fold=fold,actual_saved_models_reopened=reopened,receipt_sha256=u.sha(receipt),
                pipeline_root=str(root.relative_to(u.ROOT)),outputs=r['outputs'],held_rows=len(prediction),all_actual_tensors_finite=True))
    partial=study.RUN/'expanded/2'
    if {p.name for p in partial.iterdir()}!={'ridge.npz','scalers.npz'}:raise ValueError('Interrupted fold contains completed model requiring preservation')
    u.write(REPORT/'verification.json',dict(state='complete_actual_seven_preserved_pipeline_files_and_models_reopened',
        original_manifest_sha256=u.sha(manifest),verifier_sha256=u.sha(Path(__file__)),records=records,
        actual_saved_models_reopened=sum(r['actual_saved_models_reopened'] for r in records),
        interrupted_fold_completed_seed_models=0,all_original_outputs_unchanged=True,
        scope='Actual complete saved model/config/tensor/preprocessing/vector existence and hashes only. No forward inference or eligible scaler-statistic replay yet; those remain mandatory after the three missing fits. No completed comparison or promotion.',qualifying_labels_read=False))
    print('Seven complete original pipelines and',sum(r['actual_saved_models_reopened'] for r in records),'actual models reopened; full inference/statistic replay still pending',flush=True)

if __name__=='__main__':main()
