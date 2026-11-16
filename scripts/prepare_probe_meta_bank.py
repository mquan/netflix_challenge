#!/usr/bin/env python3
"""Compact final stacking bank from audited Probe-fitted public blenders."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
from gbdt_blend import load,blend_fold
from round4_boost import extra_meta
from streaming_blend import digest,grouped_statistics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    imported=json.loads(Path(a.manifest).read_text())
    if imported['state']!='complete' or imported['policy']['qualifying_outcomes_used_for_weight_fitting'] is not False:
        raise ValueError('blender source policy differs')
    b0=json.loads(Path('artifacts/public-new/ensemble-tree-s1.inputs.json').read_text())
    parent_prefix='artifacts/qualifying/public536/qual';parent=json.loads(Path(parent_prefix+'.json').read_text())
    rows=np.fromfile('artifacts/public-new/dataset/original-probe-rows.u32',dtype='<u4')
    split=json.loads(Path('artifacts/public-new/dataset/manifest.json').read_text())
    if digest('artifacts/public-new/dataset/original-probe-rows.u32')!=split['original_rows_sha256']:
        raise ValueError('retained-row map differs')
    names=[];inputs={};query_columns=[];projections=[]
    def add(dev,expected,query,qhash,format):
        if digest(dev)!=expected or digest(query)!=qhash:raise ValueError('stacking input changed')
        values=np.fromfile(dev,dtype='<f4')
        if values.shape!=(len(rows),) or not np.isfinite(values).all():raise ValueError('invalid development column')
        names.append(str(Path(dev).resolve()));inputs[names[-1]]=expected
        query_columns.append(dict(path=str(Path(query).resolve()),sha256=qhash,format=format))
    for item in imported['columns']:
        add(item['path'],item['sha256'],item['qualifying_path'],item['qualifying_sha256'],'npy')
    extras=root/'extra-columns';extras.mkdir()
    original='artifacts/qualifying/dev-public385-NN2'
    cv=json.loads(Path(original+'.json').read_text())
    neural=Path('artifacts/qualifying/public385/nn-only');nr=json.loads((neural/'report.json').read_text())
    if cv['config']['bank']!=nr['config']['bank'] or cv['config']['seeds']!=[2026,7,13] or nr['qualifying_labels_read'] is not False or nr['state']!='complete':
        raise ValueError('neural reference recipe differs')
    if cv['bank_input_sha256']!=nr['bank_sha256']:raise ValueError('neural reference bank digest differs')
    oldq='artifacts/qualifying/public385/qual'
    if digest(oldq+'.json')!=nr['qual_manifest_sha256']:raise ValueError('neural query manifest changed')
    oldq_report=json.loads(Path(oldq+'.json').read_text())
    if oldq_report['data_sha256']!=parent['data_sha256']:raise ValueError('neural query dataset differs')
    value=np.fromfile(original+'.nn.f32',dtype='<f4')
    if len(value)!=1408395:raise ValueError('original neural OOF shape differs')
    path=extras/'NN385-oof.f32';np.clip(value[rows],1,5).astype('<f4').tofile(path)
    add(path,digest(path),neural/'qualifying-final.f32',nr['frozen_prediction_sha256'],'f32')
    projections.append(dict(kind='Projected original-Probe OOF NN385',original_sha256=digest(original+'.nn.f32'),
        cv_report_sha256=digest(original+'.json'),production_report_sha256=digest(neural/'report.json'),
        clipped_both_development_and_qualifying_inputs=True))
    for name,reference,producer in [
        ('A2h-mid','A2h-snapshot-4-8.f32','A2h-snapshot-4-8'),
        ('MT3','MT3-refit.refit.e7.f32','MT3-fixed7'),
        ('MT8','MT8-refit.refit.e7.f32','MT8-fixed7'),
        ('MT1','MT1-refit.refit.e10.f32','MT1-fixed10'),
        ('MH3','MH3-categorical-mse-refit.f32','MH3-fixed4')]:
        reference=Path('artifacts/retained-neural')/reference;producer=Path('artifacts/qualifying')/producer
        report=json.loads((producer/'report.json').read_text())
        if report['state']!='complete' or report['metric'] is not None or report['training_requested'] is not True or len(report['records'])!=report['fixed_epochs']:
            raise ValueError('local production budget incomplete')
        if digest(reference)!=report['column_sha256'] or digest(producer/'model.pt')!=report['checkpoint_sha256']:
            raise ValueError('local family checkpoint or reference changed')
        if report['data_sha256']!=parent['data_sha256'] or report['dataset_manifest_sha256']!=parent['dataset_manifest_sha256']:
            raise ValueError('local qualifying dataset or row order differs')
        values=np.fromfile(reference,dtype='<f4');dest=extras/(name+'.f32')
        if len(values)!=1408395:raise ValueError('local development row count differs')
        values[rows].astype('<f4').tofile(dest)
        add(dest,digest(dest),producer/'model.f32',report['predictions_sha256'],'f32')
        projections.append(dict(kind=name,original_sha256=digest(reference),production_report_sha256=digest(producer/'report.json')))
    columns=len(names);supports=b0['names'][b0['columns']:]
    if b0['support']!=220 or len(supports)!=220:raise ValueError('unexpected support schema')
    for name in supports:
        if digest(name)!=b0['inputs'][name]:raise ValueError('support changed')
        names.append(name);inputs[name]=b0['inputs'][name]
    data=b0['config']['data'];meta=b0['config']['meta']
    if digest(data)!=b0['data_sha256'] or digest(meta)!=b0['meta_sha256']:raise ValueError('development data changed')
    d=load(data);cache=np.load(meta);more,more_names=extra_meta(d)
    if more_names!=b0['extra_meta_names']:raise ValueError('metadata order differs')
    bank=str(root/'bank');width=len(names)+cache['meta'].shape[1]+more.shape[1]
    x=np.lib.format.open_memmap(bank+'.x.npy',mode='w+',dtype='<f4',shape=(len(rows),width),fortran_order=True)
    for j,name in enumerate(names):
        values=np.fromfile(name,dtype='<f4')
        if values.shape!=(len(rows),) or not np.isfinite(values).all():raise ValueError('invalid feature column')
        x[:,j]=values
    x[:,len(names):len(names)+cache['meta'].shape[1]]=cache['meta'];x[:,len(names)+cache['meta'].shape[1]:]=more;x.flush()
    config=dict(data=data,meta=meta,output=bank,extra=[],sample_weights=None,
        metric_label='ProbeX stacking development; upstream row crossfits/legacy epoch weights are not nested within user folds, not independent CV')
    record=dict(config=config,inputs=inputs,names=names,columns=columns,support=220,extra_meta_names=more_names,
        data_sha256=b0['data_sha256'],meta_sha256=b0['meta_sha256'],public_manifest_sha256=digest(a.manifest),
        pretrained_blender_policy=imported['policy'],local_projection_proofs=projections,
        original_probe_row_map_sha256=digest('artifacts/public-new/dataset/original-probe-rows.u32'))
    Path(bank+'.inputs.json').write_text(json.dumps(record,indent=2)+'\n')
    inner=blend_fold(d['probe']['user'].astype(np.uint64)*np.uint64(7)+np.uint64(1))==0
    stats=grouped_statistics(x[:,:columns],cache['context'],d['probe']['rating'].astype(np.float64),blend_fold(d['probe']['user'])*2+inner)
    np.savez(bank+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
    if parent['bank_sha256']!=digest('artifacts/public-new/ensemble-tree-s1.inputs.json') or parent['qualifying_labels_read'] is not False:
        raise ValueError('query parent differs')
    generic=str(root/'query-support')
    selection=dict(base_prefix=str(Path(parent_prefix).resolve()),base_manifest_sha256=digest(parent_prefix+'.json'),
        columns=list(range(b0['columns'],parent['width'])),shape=[parent['rows'],width-columns])
    Path(generic+'.matrix.json').write_text(json.dumps(selection,indent=2)+'\n')
    generic_report=dict(parent,state='complete',columns=0,width=width-columns,bank_sha256=None,
        purpose='Unlabeled support block only; not a standalone model bank',matrix_layout='column_selection',
        composite_spec_sha256=digest(generic+'.matrix.json'),qualifying_labels_read=False)
    generic_report.pop('x_sha256',None)
    Path(generic+'.json').write_text(json.dumps(generic_report,indent=2)+'\n')
    qual=str(root/'qual');spec=dict(base_prefix=str(Path(generic).resolve()),base_manifest_sha256=digest(generic+'.json'),
        columns=query_columns,insert_at=0,shape=[parent['rows'],width])
    Path(qual+'.matrix.json').write_text(json.dumps(spec,indent=2)+'\n')
    os.link(parent_prefix+'.context.npy',qual+'.context.npy')
    parent.update(columns=columns,width=width,bank_sha256=digest(bank+'.inputs.json'),
        matrix_layout='column_block_insertion',composite_spec_sha256=digest(qual+'.matrix.json'),
        derivation='Audited Probe-only source blenders plus projected local references; existing verified source-new supports/context, no qualifying labels')
    parent.pop('x_sha256',None)
    Path(qual+'.json').write_text(json.dumps(parent,indent=2)+'\n')
    (root/'report.json').write_text(json.dumps(dict(state='complete',bank=bank,qual=qual,columns=columns,width=width,
        qualifying_labels_read=False,independent_development_cv=False),indent=2)+'\n')
    print(json.dumps(dict(state='complete',columns=columns,width=width)),flush=True)


if __name__=='__main__':main()
