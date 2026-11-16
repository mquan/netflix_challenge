#!/usr/bin/env python3
"""Keep individual predictors alongside audited known-development blend outputs."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
from gbdt_blend import load,blend_fold
from streaming_blend import digest,grouped_statistics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-bank',required=True);ap.add_argument('--base-qual',required=True)
    ap.add_argument('--meta-bank',required=True);ap.add_argument('--meta-qual',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(a.base_bank+'.inputs.json').read_text())
    meta=json.loads(Path(a.meta_bank+'.inputs.json').read_text())
    for key in ['data_sha256','meta_sha256','support','extra_meta_names']:
        if base[key]!=meta[key]:raise ValueError('different development schema '+key)
    n,k=base['columns'],meta['columns']
    if base['names'][n:]!=meta['names'][k:]:raise ValueError('support columns differ')
    raw=np.load(a.meta_bank+'.x.npy',mmap_mode='r')
    names=base['names'][:n]+meta['names'];inputs={}
    for j,name in enumerate(names):
        expected=(base['inputs'] if j<n else meta['inputs'])[name]
        if digest(name)!=expected:raise ValueError('source column changed')
        inputs[name]=expected
    bank=str(root/'bank');x=np.lib.format.open_memmap(bank+'.x.npy',mode='w+',dtype='<f4',
        shape=(len(raw),n+raw.shape[1]),fortran_order=True)
    for j,name in enumerate(base['names'][:n]):
        values=np.fromfile(name,dtype='<f4')
        if values.shape!=(len(raw),) or not np.isfinite(values).all():raise ValueError('invalid raw prediction')
        x[:,j]=values
    for j in range(raw.shape[1]):x[:,n+j]=raw[:,j]
    x.flush()
    b=dict(meta,inputs=inputs,names=names,columns=n+k)
    b['config']=dict(meta['config'],output=bank)
    b.update(raw_source_bank_sha256=digest(a.base_bank+'.inputs.json'),
        meta_source_bank_sha256=digest(a.meta_bank+'.inputs.json'),
        derivation='Individual raw CF predictors followed by audited Probe-fitted blender outputs, identical support/context')
    Path(bank+'.inputs.json').write_text(json.dumps(b,indent=2)+'\n')
    print(json.dumps(dict(stage='statistics',columns=n+k,width=x.shape[1])),flush=True)
    if digest(b['config']['data'])!=b['data_sha256'] or digest(b['config']['meta'])!=b['meta_sha256']:
        raise ValueError('data changed')
    d=load(b['config']['data']);c=np.load(b['config']['meta'])['context']
    inner=blend_fold(d['probe']['user'].astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    stats=grouped_statistics(x[:,:n+k],c,d['probe']['rating'].astype(np.float64),blend_fold(d['probe']['user'])*2+inner)
    np.savez(bank+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
    q=json.loads(Path(a.base_qual+'.json').read_text());mq=json.loads(Path(a.meta_qual+'.json').read_text())
    if q['bank_sha256']!=digest(a.base_bank+'.inputs.json') or mq['bank_sha256']!=digest(a.meta_bank+'.inputs.json'):
        raise ValueError('unverified query schemas')
    for key in ['data_sha256','dataset_manifest_sha256','context_sha256','qualifying_labels_read']:
        if q[key]!=mq[key]:raise ValueError('query data differs')
    if q['qualifying_labels_read'] is not False or mq['matrix_layout']!='column_block_insertion':
        raise ValueError('expected unlabeled meta column block')
    if digest(a.meta_qual+'.matrix.json')!=mq['composite_spec_sha256']:raise ValueError('meta block changed')
    ms=json.loads(Path(a.meta_qual+'.matrix.json').read_text())
    if ms['insert_at']!=0 or len(ms['columns'])!=k:raise ValueError('meta prediction block differs')
    qual=str(root/'qual');spec=dict(base_prefix=str(Path(a.base_qual).resolve()),
        base_manifest_sha256=digest(a.base_qual+'.json'),columns=ms['columns'],insert_at=n,
        shape=[q['rows'],x.shape[1]])
    Path(qual+'.matrix.json').write_text(json.dumps(spec,indent=2)+'\n')
    os.link(a.base_qual+'.context.npy',qual+'.context.npy')
    q.update(columns=n+k,width=x.shape[1],bank_sha256=digest(bank+'.inputs.json'),
        matrix_layout='column_block_insertion',composite_spec_sha256=digest(qual+'.matrix.json'),
        derivation=b['derivation'])
    q.pop('x_sha256',None)
    Path(qual+'.json').write_text(json.dumps(q,indent=2)+'\n')
    (root/'report.json').write_text(json.dumps(dict(state='complete',columns=n+k,width=x.shape[1],
        qualifying_labels_read=False,independent_development_cv=False),indent=2)+'\n')
    print(json.dumps(dict(state='complete',columns=n+k)),flush=True)


if __name__=='__main__':main()
