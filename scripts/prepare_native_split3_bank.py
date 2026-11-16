#!/usr/bin/env python3
"""Use the public split_3 family's full held-out population for meta training.

All original Probe rows plus 352099 further known ratings were excluded by
these base learners. No old/local/new-split predictor is eligible on the added
rows. This is an alternative development population, not original-Probe CV.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from gbdt_blend import load,HEADER,meta_features,blend_fold
from moonshot_nn_blend import contexts
from round4_boost import extra_meta
from streaming_blend import digest,grouped_statistics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='data/processed/netflix.bin')
    ap.add_argument('--mapping-prefix',default='artifacts/public-bank/split3-map')
    ap.add_argument('--raw-user-map',default='artifacts/public-bank/dense-to-raw-user.u32')
    ap.add_argument('--base-bank',default='artifacts/qualifying/dev-public385')
    ap.add_argument('--base-qual',default='artifacts/qualifying/public385/qual')
    ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    original=load(a.source);mapping=np.fromfile(a.mapping_prefix+'.u32',dtype='<u4')
    moved=np.fromfile(a.mapping_prefix+'.moved.u32',dtype='<u4')
    source_items=np.fromfile(a.mapping_prefix+'.source-items.u32',dtype='<u4')
    source_users=np.fromfile(a.mapping_prefix+'.source-users.u32',dtype='<u4')
    raw=np.fromfile(a.raw_user_map,dtype='<u4');dense_order=np.argsort(raw)
    if len(mapping)!=len(original['probe']) or len(np.unique(mapping))!=len(mapping):raise ValueError('invalid Probe mapping')
    if len(moved)!=352099 or len(mapping)!=1408395 or len(source_items)!=1760494 or len(raw)!=original['users']:
        raise ValueError('unexpected published split sizes')
    if len(np.unique(moved))!=len(moved) or np.any(moved>=len(original['train_ratings'])):raise ValueError('invalid held rows')
    q=np.zeros(len(source_items),dtype=original['probe'].dtype);q[mapping]=original['probe']
    remaining=np.ones(len(q),dtype=bool);remaining[mapping]=False;remaining=np.flatnonzero(remaining)
    owner=np.searchsorted(original['train_offsets'],moved,side='right')-1
    source_user_rank=np.argsort(dense_order)
    order=np.lexsort((source_user_rank[owner],original['train_items'][moved]))
    selected=moved[order];owners=owner[order]
    q['user'][remaining]=owners;q['item'][remaining]=original['train_items'][selected]
    q['day'][remaining]=original['train_days'][selected];q['rating'][remaining]=original['train_ratings'][selected]
    if not np.array_equal(q['item'],source_items) or not np.array_equal(q['user'],dense_order[source_users]):
        raise ValueError('full source row mapping differs')
    keep=np.ones(len(original['train_ratings']),dtype=bool);keep[moved]=False
    count=np.diff(original['train_offsets'])-np.bincount(owner,minlength=original['users'])
    offsets=np.r_[0,np.cumsum(count)].astype('<u8')
    h=np.fromfile(a.source,dtype=HEADER,count=1);h['train_count']=keep.sum();h['probe_count']=len(q)
    h['mean']=(np.sum(original['train_ratings'],dtype=np.float64)-np.sum(original['train_ratings'][moved],dtype=np.float64))/keep.sum()
    data=root/'train3-probe3.bin'
    with data.open('xb') as f:
        for values in [h,offsets]:values.tofile(f)
        for key in ['train_items','train_days','train_ratings']:original[key][keep].tofile(f)
        for values in [q,original['implicit_offsets'],original['implicit_items'],original['implicit_days']]:values.tofile(f)
    proof=dict(original_data_sha256=digest(a.source),mapping_files={str(p):digest(p) for p in
        [Path(a.mapping_prefix+s) for s in ['.u32','.moved.u32','.source-items.u32','.source-users.u32']]+[Path(a.raw_user_map)]},
        original_probe_rows=len(mapping),added_held_original_training_rows=len(moved),development_rows=len(q),
        base_training_rows=int(keep.sum()),original_probe_entirely_excluded=True,
        query_order='Exact published numeric item/source-user order',qualifying_labels_read=False,
        limitation='Alternative full native split3 development population; not original-Probe CV')
    del original,keep,q,raw,source_items,source_users
    print(json.dumps(dict(stage='dataset',proof=proof)),flush=True)
    d=load(data);meta=np.column_stack(list(meta_features(d).values())).astype(np.float32)
    context=contexts(d);more,more_names=extra_meta(d)
    metapath=root/'meta-context.npz';np.savez(metapath,meta=meta,context=context,data_sha256=digest(data))
    base=json.loads(Path(a.base_bank+'.inputs.json').read_text())
    lookup={}
    for folder in ['split3-columns','split3-support']:
        manifest_path=Path('artifacts/public-bank')/folder/'manifest.json';m=json.loads(manifest_path.read_text())
        if m['plan']['mapping_sha256']!=digest(a.mapping_prefix+'.u32'):raise ValueError('source mapping provenance differs')
        for entry in m['columns']:
            source=manifest_path.parent/Path(entry['source']).name
            lookup[entry['path']]=(source,entry)
    columns=base['columns']-122;support=base['support']
    names=base['names'][122:];bank=str(root/'bank')
    if columns!=263 or support!=220 or len(names)!=columns+support:raise ValueError('unexpected family bank')
    x=np.lib.format.open_memmap(bank+'.x.npy',mode='w+',dtype='<f4',shape=(len(d['probe']),len(names)+meta.shape[1]+more.shape[1]),fortran_order=True)
    inputs={};full_names=[];column_proofs=[]
    for j,name in enumerate(names):
        source,entry=lookup[name]
        if hashlib.md5(source.read_bytes()).hexdigest()!=entry['md5'] or digest(name)!=entry['sha256']:
            raise ValueError('source column changed')
        values=np.load(source,allow_pickle=False)
        if values.shape!=(len(x),) or not np.isfinite(values).all() or not np.array_equal(values[mapping],np.fromfile(name,dtype='<f4')):
            raise ValueError('native column does not project to verified original Probe')
        x[:,j]=values;full_names.append(str(source.resolve()));inputs[full_names[-1]]=digest(source)
        column_proofs.append(dict(source=str(source),sha256=inputs[full_names[-1]],original_projected_column=name,projected_sha256=entry['sha256']))
    x[:,len(names):len(names)+meta.shape[1]]=meta;x[:,len(names)+meta.shape[1]:]=more;x.flush()
    config=dict(data=str(data),meta=str(metapath),output=bank,extra=[],sample_weights=None,
                metric_label='Full native split3 holdout; alternative population, not original-Probe CV')
    record=dict(config=config,inputs=inputs,names=full_names,columns=columns,support=support,
        extra_meta_names=more_names,data_sha256=digest(data),meta_sha256=digest(metapath),
        derivation=proof,column_projection_proofs=column_proofs,x_sha256=digest(bank+'.x.npy'))
    Path(bank+'.inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    inner=blend_fold(d['probe']['user'].astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    stats=grouped_statistics(x[:,:columns],context,d['probe']['rating'].astype(np.float64),blend_fold(d['probe']['user'])*2+inner)
    np.savez(bank+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
    parent=json.loads(Path(a.base_qual+'.json').read_text())
    if parent['bank_sha256']!=digest(a.base_bank+'.inputs.json') or parent['qualifying_labels_read'] is not False:
        raise ValueError('unverified qualifying parent')
    prefix=str(root/'qual');selection=list(range(122,parent['width']))
    spec=dict(base_prefix=str(Path(a.base_qual).resolve()),base_manifest_sha256=digest(a.base_qual+'.json'),
        columns=selection,shape=[parent['rows'],len(selection)])
    Path(prefix+'.matrix.json').write_text(json.dumps(spec,indent=2)+'\n')
    os.link(a.base_qual+'.context.npy',prefix+'.context.npy')
    parent.update(bank_sha256=digest(bank+'.inputs.json'),columns=columns,width=len(selection),
        matrix_layout='column_selection',composite_spec_sha256=digest(prefix+'.matrix.json'),derivation=proof)
    parent.pop('x_sha256',None)
    Path(prefix+'.json').write_text(json.dumps(parent,indent=2)+'\n')
    (root/'report.json').write_text(json.dumps(dict(state='complete',bank=bank,qual=prefix,proof=proof),indent=2)+'\n')
    print(json.dumps(dict(state='complete',bank=bank,qual=prefix)),flush=True)


if __name__=='__main__':main()
