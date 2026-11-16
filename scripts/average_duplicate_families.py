#!/usr/bin/env python3
"""Remove development contrasts between bases identical on every query row.

Only unlabeled query predictions identify families. Each development family is
replaced by its row mean, retaining feature width and multiplicity. The query
matrix is unchanged, after checking equality on ALL rows, not a sample.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
from gbdt_blend import load,blend_fold
from qual_matrix import verified_matrix
from streaming_blend import digest,grouped_statistics


def average_groups(block,groups):
    result=np.array(block,copy=True,order='F')
    for group in groups:
        result[:,group]=np.mean(block[:,group],axis=1,dtype=np.float64).astype(np.float32)[:,None]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--qual-output',required=True)
    a=ap.parse_args()
    if Path(a.output+'.inputs.json').exists() or Path(a.qual_output+'.json').exists():
        raise ValueError('output exists')
    b=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.qual+'.json').read_text())
    if digest(b['config']['data'])!=b['data_sha256'] or digest(b['config']['meta'])!=b['meta_sha256']:
        raise ValueError('development data or metadata changed')
    if q['bank_sha256']!=digest(a.bank+'.inputs.json') or q['qualifying_labels_read'] is not False:
        raise ValueError('unverified query bank')
    raw=np.load(a.bank+'.x.npy',mmap_mode='r');query=verified_matrix(a.qual,q)
    # Resolve insertion overlays to read individual columns without repeatedly
    # materializing the full query matrix.
    def column(matrix,j):
        if hasattr(matrix,'base') and hasattr(matrix,'index'):
            if j==matrix.index:return matrix.column
            return column(matrix.base,j-(j>matrix.index))
        return matrix[:,j]
    by_hash={}
    for j in range(b['columns']):
        v=column(query,j)
        h=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        by_hash.setdefault(h,[]).append(j)
    groups=[g for g in by_hash.values() if len(g)>1]
    for g in groups:
        for j in g[1:]:
            if not np.array_equal(column(query,g[0]),column(query,j)):
                raise ValueError('hash collision or changed query columns')
    if not groups:raise ValueError('no duplicate families')
    print(json.dumps(dict(stage='families',groups=groups)),flush=True)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    x=np.lib.format.open_memmap(a.output+'.x.npy',mode='w+',dtype='<f4',shape=raw.shape,fortran_order=True)
    for lo in range(0,len(raw),16384):x[lo:lo+16384]=average_groups(raw[lo:lo+16384],groups)
    x.flush()
    derived=dest.parent/(dest.name+'-columns');derived.mkdir(exist_ok=False)
    transformed={}
    for number,g in enumerate(groups):
        path=derived/f'family{number}.f32';np.asarray(x[:,g[0]],dtype='<f4').tofile(path)
        path=str(path.resolve());h=digest(path)
        for j in g:transformed[j]=(path,h)
    names=list(b['names']);inputs={}
    for j,name in enumerate(names):
        if j in transformed:names[j],h=transformed[j]
        else:h=b['inputs'][name]
        inputs[names[j]]=h
    derivation=dict(kind='average_development_families_identical_on_all_unlabeled_queries',
        source_bank=a.bank,source_bank_sha256=digest(a.bank+'.inputs.json'),
        source_qual=a.qual,source_qual_sha256=digest(a.qual+'.json'),groups=groups,
        qualifying_labels_read=False,unlabeled_target_predictions_used=True,
        query_matrix_unchanged=True,development_x_sha256=digest(a.output+'.x.npy'))
    b.update(names=names,inputs=inputs,derivation=derivation)
    b['config']=dict(b['config'],output=a.output)
    Path(a.output+'.inputs.json').write_text(json.dumps(b,indent=2)+'\n')
    d=load(b['config']['data']);y=d['probe']['rating'].astype(np.float64);users=d['probe']['user'];del d
    context=np.load(b['config']['meta'])['context']
    inner=blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    stats=grouped_statistics(x[:,:b['columns']],context,y,blend_fold(users)*2+inner)
    np.savez(a.output+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
    q.update(bank_sha256=digest(a.output+'.inputs.json'),derivation=derivation)
    Path(a.qual_output).parent.mkdir(parents=True,exist_ok=True)
    # Matrix specification binds its original immutable parent; its contents
    # equal the transformed matrix because each family was checked above.
    if q.get('matrix_layout')=='column_insertion':
        os.link(a.qual+'.matrix.json',a.qual_output+'.matrix.json')
    else:os.link(a.qual+'.x.npy',a.qual_output+'.x.npy')
    os.link(a.qual+'.context.npy',a.qual_output+'.context.npy')
    Path(a.qual_output+'.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(dict(state='complete',derivation=derivation)),flush=True)


if __name__=='__main__':main()
