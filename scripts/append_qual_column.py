#!/usr/bin/env python3
"""Attach a verified full-training local base to an audited qualifying bank."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from streaming_blend import digest
from qual_matrix import verified_matrix


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-bank',required=True);ap.add_argument('--base-qual',required=True)
    ap.add_argument('--bank',required=True);ap.add_argument('--local-report',required=True)
    ap.add_argument('--local-column',required=True);ap.add_argument('--dataset-manifest',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--reference-column',help='original full-Probe column, if the development bank uses its projection')
    ap.add_argument('--row-map',help='audited original-Probe to retained-row projection')
    a=ap.parse_args()
    base=json.loads(Path(a.base_bank+'.inputs.json').read_text())
    bank=json.loads(Path(a.bank+'.inputs.json').read_text())
    q=json.loads(Path(a.base_qual+'.json').read_text())
    local=json.loads(Path(a.local_report).read_text())
    if local['state']!='complete' or local['metric'] is not None:
        raise ValueError('local base has not completed label-isolated production training')
    budget=local.get('fixed_training_steps',local.get('fixed_epochs'))
    if local.get('training_requested') is not True or len(local['records'])!=budget:
        raise ValueError('local base has not completed its fixed training budget')
    if local['dataset_manifest_sha256']!=digest(a.dataset_manifest):raise ValueError('local query dataset differs')
    if local['predictions_sha256']!=digest(a.local_column):raise ValueError('local prediction changed')
    checkpoint=local.get('checkpoint_path',str(Path(a.local_column).with_suffix('.pt')))
    if local['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('local checkpoint changed')
    if q['qualifying_labels_read'] is not False or q['bank_sha256']!=digest(a.base_bank+'.inputs.json'):
        raise ValueError('unverified qualifying bank')
    if q['dataset_manifest_sha256']!=local['dataset_manifest_sha256'] or q['data_sha256']!=local['data_sha256']:
        raise ValueError('different qualifying row order/data')
    n=base['columns']
    if bank['columns']!=n+1 or bank['names'][:n]+bank['names'][n+1:]!=base['names']:
        raise ValueError('bank differs by more than one trailing base')
    retained_hash=bank['inputs'][bank['names'][n]]
    projection=None
    if retained_hash!=local['column_sha256']:
        if not a.reference_column or not a.row_map or digest(a.reference_column)!=local['column_sha256']:
            raise ValueError('local production recipe does not match the retained development base')
        source_split=json.loads(Path(bank['config']['data']).with_name('manifest.json').read_text())
        if source_split['original_rows_sha256']!=digest(a.row_map):raise ValueError('projection map differs')
        rows=np.fromfile(a.row_map,dtype='<u4');original=np.fromfile(a.reference_column,dtype='<f4')
        retained_path=bank['names'][n]
        if digest(retained_path)!=retained_hash or not np.array_equal(original[rows],np.fromfile(retained_path,dtype='<f4')):
            raise ValueError('retained local predictions do not match the audited projection')
        projection=dict(original_sha256=local['column_sha256'],rows_sha256=digest(a.row_map),
                        retained_sha256=retained_hash)
    for key in ('data_sha256','meta_sha256','extra_meta_names'):
        if bank[key]!=base[key]:raise ValueError('development schema differs: '+key)
    for name in base['names']:
        if bank['inputs'][name]!=base['inputs'][name]:raise ValueError('development column changed')
    if digest(a.base_qual+'.context.npy')!=q['context_sha256']:raise ValueError('qualifying context changed')
    raw=verified_matrix(a.base_qual,q)
    values=np.fromfile(a.local_column,dtype='<f4')
    if len(values)!=len(raw) or len(values)!=local['prediction_count'] or not np.isfinite(values).all():
        raise ValueError('local prediction rows differ')
    if Path(a.output+'.json').exists() or Path(a.output+'.matrix.json').exists():raise ValueError('output exists')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    shape=(len(raw),raw.shape[1]+1)
    spec=dict(base_prefix=str(Path(a.base_qual).resolve()),base_manifest_sha256=digest(a.base_qual+'.json'),
              column_path=str(Path(a.local_column).resolve()),column_sha256=local['predictions_sha256'],
              insert_at=n,shape=list(shape))
    Path(a.output+'.matrix.json').write_text(json.dumps(spec,indent=2)+'\n')
    os.link(a.base_qual+'.context.npy',a.output+'.context.npy')
    q.pop('x_sha256',None)
    q.update(columns=bank['columns'],width=shape[1],bank_sha256=digest(a.bank+'.inputs.json'),
             parent_qual_manifest_sha256=digest(a.base_qual+'.json'),
             local_production_report_sha256=digest(a.local_report),
             local_execution_overrides=local.get('execution_overrides'),
             local_prediction_sha256=local['predictions_sha256'],
             matrix_layout='column_insertion',composite_spec_sha256=digest(a.output+'.matrix.json'),
             local_development_projection=projection,qualifying_labels_read=False,
             derivation='Insert retained local base at the exact development column; same verified query dataset/order')
    q['qualifying_inputs'][a.local_column]=local['predictions_sha256']
    Path(a.output+'.json').write_text(json.dumps(q,indent=2)+'\n')
    print(json.dumps(dict(state='complete',output=a.output,rows=len(raw),columns=bank['columns'])),flush=True)


if __name__=='__main__': main()
