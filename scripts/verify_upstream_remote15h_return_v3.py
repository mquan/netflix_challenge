#!/usr/bin/env python3
"""Check frozen remote transfer, original recipe/traces and lossless local pairs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
import upstream_remote15h as helper
import upstream_repro as u

def main(name):
    report=u.ROOT/'experiments/upstream-remote15h-return'/(name+'-v3')
    target=u.ROOT/'artifacts/remote15h-return'/(name+'-v3')
    if report.exists() or target.exists():raise ValueError('Existing imported return preserved')
    report.mkdir(parents=True);revision=subprocess.check_output(['git','rev-parse','origin/research/remote15h-handoff'],text=True).strip()
    prefix='experiments/upstream-remote15h/execution/'+name+'/'
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,prefix],text=True).splitlines()
    if not files:raise ValueError('Committed remote return evidence required')
    for rel in files:
        blob=subprocess.check_output(['git','show',revision+':'+rel]);(report/Path(rel).name).write_bytes(blob)
    transfer=helper.read(report/'transfer.json');archive=u.ROOT/'artifacts/remote15h-return'/transfer['asset_name']
    if not transfer['uploaded'] or archive.stat().st_size!=transfer['archive_bytes'] or helper.sha(archive)!=transfer['archive_sha256']:
        raise ValueError('Complete immutable draft-release archive differs')
    extracted=target/'export';extracted.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>100*2**20:raise ValueError('Unexpected export size')
        allowed=set(helper.read(report/'verification.json')['files_sha256'])|{'verification.json'}
        accepted=[];ignored=[]
        for m in members:
            p=Path(m.name)
            if p.name.startswith('._') and m.isfile() and m.size<4096 and '..' not in p.parts and not p.is_absolute():
                if (len(p.parts)==1 and p.name=='._'+name) or (len(p.parts)==2 and p.parts[0]==name and p.name[2:] in allowed):
                    ignored.append(m.name);continue
            if p.is_absolute() or '..' in p.parts or len(p.parts)>2 or p.parts[0]!=name or not(m.isfile() or m.isdir()):
                raise ValueError('Unsafe export member')
            if m.isfile() and p.name not in allowed:raise ValueError('Undeclared payload member')
            accepted.append(m)
        tar.extractall(extracted,members=accepted)
    bundle=extracted/name;remote=helper.read(bundle/'verification.json');plan=helper.read(helper.PLAN)
    job=next(j for j in plan['jobs'] if j['name']==name)
    if (remote['state']!='verified_complete_original_remote15h_producer' or remote['job']!=name or
        remote['source_pin']!=u.PIN or not remote['both_full_original_epoch_traces'] or
        not remote['independent_known_projection'] or not remote['source_data_binary_rehashed'] or
        remote['qualifying_labels_read'] is not False or remote['public_prediction_values_read'] is not False):
        raise ValueError('Remote separate source/export verification required')
    helper.check_files(bundle,remote['files_sha256'])
    if helper.sha(bundle/'plan.json')!=helper.sha(helper.PLAN) or helper.sha(bundle/'upstream_remote15h.py')!=helper.sha(Path(helper.__file__)):
        raise ValueError('Frozen original plan/helper differs')
    launch=helper.read(bundle/(name+'-launch.json'));receipt=helper.read(bundle/(name+'-receipt.json'));prep=helper.read(bundle/'preparation.json')
    if (launch['command']!=job['command'] or launch['environment']!=job['environment'] or
        launch['plan_sha256']!=helper.sha(helper.PLAN) or receipt['launch_sha256']!=helper.sha(bundle/(name+'-launch.json')) or
        receipt['log_sha256']!=helper.sha(bundle/(name+'.log')) or launch['preparation_sha256']!=helper.sha(bundle/'preparation.json') or
        prep['row_map_sha256']!=plan['row_map_sha256'] or not prep['known_rows_absent_from_training'] or not prep['qualifying_targets_zero']):
        raise ValueError('Original recipe/data/row projection launch lineage differs')
    helper.trace_check((bundle/(name+'.log')).read_text(),job['epochs_per_phase'])
    local=u.ROOT/'artifacts/upstream-split3-pilot/source/data'
    helper.check_files(local,plan['split3_sha256']);helper.check_files(u.FRESH/'data',plan['data_sha256'])
    known=helper.arrays(u.FRESH/'data/probex',labels=True)
    mapping=helper.projection(helper.arrays(local/'probe_3',labels=True),known)
    helper.assert_disjoint(np.sort(helper.keys(*known[:3])),helper.arrays(local/'train_3'))
    np.save(target/'probex-row-map.npy',mapping)
    if helper.sha(target/'probex-row-map.npy')!=plan['row_map_sha256']:raise ValueError('Independent local full query map differs')
    if [r['column'] for r in remote['records']]!=job['published_columns']:raise ValueError('Exactly declared published pairs required')
    template=u.ROOT/'artifacts/upstream-split3-diversity/source/preds_3/s3-dnn32nm.qual.npy'
    with template.open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        shape,order,dtype=np.lib.format._read_array_header(stream,version)
        header_bytes=stream.tell();stream.seek(0);native_header=stream.read(header_bytes)
    if shape!=(2817131,) or order or dtype!=np.float32:raise ValueError('Original native NPY header template differs')
    rows=[];bindings={str(template.relative_to(u.ROOT)):helper.sha(template)}
    for row in remote['records']:
        path=bundle/row['file'];pairs={}
        if helper.sha(path)!=row['sha256']:raise ValueError('Transferred payload differs')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'probex','qual'}:raise ValueError('Only declared own paired vectors permitted')
            for query,count in [('probex',1128164),('qual',2817131)]:
                a=data[query]
                if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Full own paired vector shape/finiteness differs')
                p=target/(row['column']+'.'+query+'.npy')
                if query=='qual':
                    with p.open('xb') as stream:stream.write(native_header);stream.write(a.tobytes(order='C'))
                else:np.save(p,a)
                if not np.array_equal(np.load(p).view(np.uint32),a.view(np.uint32)):raise ValueError('Lossless local vector replay differs')
                pairs[query]=str(p.relative_to(u.ROOT));bindings[pairs[query]]=helper.sha(p)
            qual_path=target/(row['column']+'.qual.npy')
            if helper.sha(qual_path)!=row['native_qual_sha256']:raise ValueError('Native qualifying NPY byte reconstruction differs')
        for query,suffix in [('probex','probe_3'),('qual','qual')]:
            if receipt['outputs']['preds_3/'+row['column']+'.'+suffix+'.npy']!=row['native_probe_sha256' if query=='probex' else 'native_qual_sha256']:
                raise ValueError('Native query-to-transfer binding differs')
        rows.append(dict(name=row['column'],paired_files=pairs,duplicate_of=None))
    for p in [archive,Path(__file__),*bundle.iterdir(),*report.iterdir()]:
        if p.is_file():bindings[str(p.relative_to(u.ROOT))]=helper.sha(p)
    helper.write(report/'local-verification.json',dict(state='complete_local_compact_remote15h_return_check',observed=helper.stamp(),job=name,
        remote_commit=revision,source_pin=u.PIN,rows=rows,bindings=bindings,ignored_archive_appledouble_metadata=ignored,
        preserved_prior_import_failure='experiments/upstream-remote15h-return/'+name+'/import-attempt1-failure.json',
        original_full_epochs_each=job['epochs_per_phase'],original_recipe_trace_and_immutable_export_verified=True,
        local_full_known_map_and_training_disjointness_recomputed=True,native_qualifying_npy_bytes_reconstructed_exactly=True,native_npy_header_template=str(template.relative_to(u.ROOT)),
        original_numpy_header_attempt_preserved='experiments/upstream-remote15h-return/'+name+'-v2/import-attempt2-failure.json',
        native_full_probe_projection_recomputed_by_separate_remote_export=True,
        native_full_probe_vectors_present_locally=False,qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Locally verified original plan/helper/source-export lineage, both full epoch traces, canonical datasets/map/disjointness and complete lossless paired vectors. Remote separate exporter rehashed original binaries/source/data/native outputs and projected every known row; native full-probe vectors remain on remote pending full-output transfer. No new local full-probe inference replay or ensemble/qualifying score claimed.'))
    print(name,'verified',len(rows),'own full prediction pairs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');args=p.parse_args();main(args.job)
