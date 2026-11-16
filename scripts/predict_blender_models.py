#!/usr/bin/env python3
"""Export qualifying predictions from models frozen before base exports existed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
from fit_qual_blend import nn_features
from round8_nn_blend import net
from streaming_blend import digest
from streaming_nn_blend import predict_device
from qual_matrix import verified_matrix,predict_matrix_ridge
from feature_adaptation import relative_unrated


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--device',choices=['cpu','mps'],default='cpu')
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads)
    models=Path(a.models_dir);m=json.loads((models/'report.json').read_text())
    if m['state']!='models_ready' or m['qualifying_inputs_provided'] is not False or m['qualifying_labels_read'] is not False:
        raise ValueError('models were not frozen using development alone')
    for name,expected in m['outputs'].items():
        if digest(models/name)!=expected:raise ValueError('trained model changed: '+name)
    bank=m['config']['bank'];b=json.loads(Path(bank+'.inputs.json').read_text())
    if digest(bank+'.inputs.json')!=m['bank_sha256']:raise ValueError('development bank changed')
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['state']!='complete' or q['qualifying_labels_read'] is not False or q['bank_sha256']!=m['bank_sha256']:
        raise ValueError('qualifying features are not verified against the frozen bank')
    if digest(a.qual+'.context.npy')!=q['context_sha256']:raise ValueError('qualifying context changed')
    raw=verified_matrix(a.qual,q);c=np.load(a.qual+'.context.npy',mmap_mode='r')
    expected_width=m.get('feature_width')
    if expected_width is None:
        first=torch.load(models/f"nn-{m['config']['seeds'][0]}.pt",map_location='cpu',weights_only=True)
        expected_width=first['state']['0.weight'].shape[1]-1
    if raw.shape!=(q['rows'],expected_width):raise ValueError('feature shape differs')
    model_report_sha=digest(models/'report.json')
    report=dict(state='predicting',config=vars(a),models_report_sha256=model_report_sha,
                bank_sha256=m['bank_sha256'],qual_manifest_sha256=digest(a.qual+'.json'),
                qualifying_labels_read=False,qualifying_rows=len(raw),metric=None,
                blend_rule=m['blend_rule'],nn_epochs=m['nn_epochs'],tree_rounds=m['tree_rounds'])
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    w=np.load(models/'ridge-weights.npz')['full'];base=predict_matrix_ridge(raw,c,b['columns'],w)
    if not m['config'].get('nn_only'):
        subprocess.run([sys.executable,str(Path(__file__).with_name('predict_tree_models.py')),
                    '--models-dir',str(models),'--qual',a.qual,'--output',str(root/'tree-correction.npy'),
                    '--threads',str(a.threads)],check=True)
        worker=json.loads((root/'tree-correction.npy.json').read_text())
        if worker['torch_loaded'] or worker['models_report_sha256']!=model_report_sha:raise ValueError('invalid tree inference')
        tree=np.load(root/'tree-correction.npy')
    else:tree=None
    scaler=np.load(models/'nn-scaler.npz');nn_models=[]
    for seed in m['config']['seeds']:
        saved=torch.load(models/f'nn-{seed}.pt',map_location='cpu',weights_only=True)
        cfg=SimpleNamespace(**saved['config'])
        model=net(raw.shape[1]+1,cfg.hidden,cfg.layers,cfg.dropout).to(a.device)
        model.load_state_dict(saved['state']);nn_models.append(model)
    neural=np.zeros(len(raw))
    for lo in range(0,len(raw),65536):
        hi=min(lo+65536,len(raw));block=np.array(raw[lo:hi],dtype=np.float32,order='F')
        f=nn_features(block,b['columns']).astype(np.float64)
        if m['config'].get('relative_unrated'):
            index,values=relative_unrated(f,b);f[:,index]=values
        f=np.ascontiguousarray((np.clip(f,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std'],dtype=np.float32)
        for model in nn_models:neural[lo:hi]+=predict_device(model,f,a.device)/len(nn_models)
    if tree is not None:(base+tree).astype('<f4').tofile(root/'tree.f32')
    (base+neural).astype('<f4').tofile(root/'nn.f32')
    final=np.clip(base+(neural if tree is None else (tree+neural)/2),1,5).astype('<f4')
    if not np.isfinite(final).all():raise ValueError('invalid predictions')
    final.tofile(root/'qualifying-final.f32')
    if digest(models/'report.json')!=model_report_sha:raise ValueError('models changed during inference')
    report.update(state='complete',frozen_prediction_sha256=digest(root/'qualifying-final.f32'),
                  prediction_min=float(final.min()),prediction_max=float(final.max()))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
