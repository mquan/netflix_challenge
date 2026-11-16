#!/usr/bin/env python3
"""Freeze an equal-weight development snapshot predictor and its provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--source-dir',required=True)
    ap.add_argument('--epochs',type=int,nargs='+',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.source_dir)
    parent=json.loads((root/'A2h-refit.json').read_text())
    progress=json.loads((root/'A2h-refit.refit.progress.json').read_text())
    if parent['data_sha256']!=digest(a.data) or parent['config']!=progress['config']:
        raise ValueError('development snapshot history differs')
    selected=root/f"A2h-refit.refit.e{parent['selected_epoch']}.f32"
    if digest(selected)!=parent['predictions_sha256']:raise ValueError('retained final snapshot differs')
    if max(a.epochs)>progress['selected_epoch']:raise ValueError('requested epoch not trained')
    d=load(a.data);y=d['probe']['rating'].astype(np.float64);result=np.zeros(len(y));inputs={}
    for epoch in a.epochs:
        p=root/f'A2h-refit.refit.e{epoch}.f32';v=np.fromfile(p,dtype='<f4')
        if len(v)!=len(y) or not np.isfinite(v).all():raise ValueError('invalid development snapshot')
        inputs[str(p)]=digest(p);result+=v.astype(np.float64)/len(a.epochs)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:result.astype('<f4').tofile(f)
    r=dict(kind='snapshot_reference',epochs=a.epochs,weights='equal',path=str(path.resolve()),
           column_sha256=digest(path),rows=len(y),data_sha256=parent['data_sha256'],config=parent['config'],
           parent_retained_column_sha256=parent['predictions_sha256'],
           parent_report_sha256=digest(root/'A2h-refit.json'),progress_sha256=digest(root/'A2h-refit.refit.progress.json'),
           inputs=inputs,qualifying_labels_read=False,
           development_rmse=float(np.sqrt(np.mean((np.clip(result,1,5)-y)**2))),
           provenance_note='Archived run config/history and retained final digest verified; earlier snapshot hashes frozen now',
           columns=[dict(source='local/'+path.stem+'.probe.npy',path=str(path.resolve()),sha256=digest(path))])
    Path(a.output+'.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


if __name__=='__main__':main()
