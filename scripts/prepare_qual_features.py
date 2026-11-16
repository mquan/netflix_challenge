#!/usr/bin/env python3
"""Materialize a production blender's qualifying features, without labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from gbdt_blend import load
from production_neural import validate_dataset
from round4_boost import extra_meta
from streaming_blend import digest


def counterpart(source):
    return source.replace('.probe_3.npy','.qual.npy').replace('.probex.npy','.qual.npy').replace('.probe.npy','.qual.npy')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True)
    ap.add_argument('--dataset-manifest',required=True)
    ap.add_argument('--meta',required=True)
    ap.add_argument('--qual-manifests',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    if Path(a.output+'.json').exists(): ap.error('output exists')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    bank=json.loads(Path(a.bank+'.inputs.json').read_text()); config=bank['config']
    if config['include_local'] or not config.get('no_local_support') or config.get('extra'):
        raise ValueError('this exporter accepts only audited public inputs')
    source_by_path={}
    for path in [config['public'],config['support_manifest']]:
        for c in json.loads(Path(path).read_text())['columns']:
            source_by_path[str(Path(c['path']).resolve())]=c['source']
    qualifying={}
    for path in a.qual_manifests:
        manifest=json.loads(Path(path).read_text())
        if manifest['plan'].get('labels_requested') is not False: raise ValueError('unverified qualifying manifest')
        for c in manifest['columns']: qualifying[c['source']]=c
    data,manifest=validate_dataset(a.dataset_manifest)
    d=load(data)
    if not np.all(d['probe']['rating']==0): raise ValueError('qualifying targets must be unknown sentinels')
    cache=np.load(a.meta)
    if str(cache['data_sha256'].item())!=manifest['outputs'][data.name]: raise ValueError('wrong metadata dataset')
    meta,context=cache['meta'],cache['context']
    more,names=extra_meta(d)
    if names!=bank['extra_meta_names']: raise ValueError('metadata schema changed')
    width=len(bank['names'])+meta.shape[1]+more.shape[1]
    dev=np.load(a.bank+'.x.npy',mmap_mode='r')
    if width!=dev.shape[1] or len(meta)!=len(d['probe']): raise ValueError('feature shape mismatch')
    x=np.lib.format.open_memmap(a.output+'.x.npy',mode='w+',dtype='<f4',
                               shape=(len(d['probe']),width),fortran_order=True)
    inputs={}
    for j,path in enumerate(bank['names']):
        source=source_by_path[str(Path(path).resolve())]
        col=qualifying[counterpart(source)]
        if digest(col['path'])!=col['sha256']: raise ValueError('qualifying input hash changed')
        values=np.load(col['path'],allow_pickle=False)
        if values.shape!=(len(x),) or not np.isfinite(values).all(): raise ValueError('bad qualifying column')
        x[:,j]=values
        inputs[col['path']]=col['sha256']
        if (j+1)%100==0: print('prepared qualifying column',j+1,'of',len(bank['names']),flush=True)
    x[:,len(bank['names']):len(bank['names'])+meta.shape[1]]=meta
    x[:,-more.shape[1]:]=more
    x.flush()
    np.save(a.output+'.context.npy',context,allow_pickle=False)
    report=dict(state='complete',purpose='qualifying_features_without_target_labels',rows=len(x),width=width,
                columns=bank['columns'],bank_sha256=digest(a.bank+'.inputs.json'),
                dataset_manifest_sha256=digest(a.dataset_manifest),data_sha256=digest(data),
                x_sha256=digest(a.output+'.x.npy'),context_sha256=digest(a.output+'.context.npy'),
                qualifying_inputs=inputs,qualifying_labels_read=False,
                query_order='Original qualifying.txt parse order, bound by verified StageC dataset')
    Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='qualifying_inputs'}),flush=True)


if __name__=='__main__': main()
