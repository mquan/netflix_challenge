#!/usr/bin/env python3
"""Independently admit original1200, then every original two-phase output."""
import argparse
from pathlib import Path
import re
import subprocess
import numpy as np
import upstream_original1200 as run

u=run.u


def admit():
    plan=u.read(run.REPORT/'execution-plan.json')
    preflight=u.read(u.ROOT/'experiments/upstream-original1200-preflight/preflight.json')
    build_path=run.REPORT/'original-build-receipt.json';build=u.read(build_path)
    binary=run.RUN/'target/release/tsvdx5-new'
    if (plan['state']!='complete_isolated_original_preparation_no_training_launched' or
        plan['source_pin']!=u.PIN or set(plan['jobs'])!={run.NAME} or plan['neighbors']!=100 or
        plan['original1200_wrapper_sha256']!=u.sha(Path(run.__file__)) or
        plan['exact_expected_outputs']!=preflight['expected_outputs'] or len(plan['exact_expected_outputs'])!=58 or
        build['source_commit']!=u.PIN or build['builder_sha256']!=u.sha(Path(run.__file__)) or
        build['cwd']!=str(run.TREE) or
        build['environment'].get('CARGO_TARGET_DIR')!=str(run.RUN/'target') or
        build['environment'].get('CARGO_BUILD_JOBS')!='1' or
        build['preparation_sha256']!=u.sha(run.REPORT/'execution-plan.json') or
        build['binary_path']!=str(binary) or build['binaries']!={'tsvdx5-new':u.sha(binary)} or
        not build['original_source_and_complete_data_unchanged'] or build['training_launched'] is not False):
        raise ValueError('Exact isolated original1200 source/build/output plan required')
    for relative,digest in plan['bindings'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Complete original1200 source/data changed')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=run.TREE,text=True).strip()!=u.PIN:
        raise ValueError('Isolated source pin differs')
    changes=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=run.TREE,text=True).splitlines()
    if set(changes)-{'data/qual_ratings/qual_ratings.csv.gz'}:raise ValueError('Original model/config changed')
    job=plan['jobs'][run.NAME]
    if (job['factors']!=1200 or job['epochs_per_phase']!=23 or
        job['original_registry_job']!=u.read(u.BASE/'registry.json')['jobs'][run.NAME]):
        raise ValueError('Original full1200/23/100-neighbor preset required')
    if len(plan['full_identity_disjointness'])!=2:raise ValueError('Exactly two original identity scans required')
    for name,digest in preflight['original_git_blob_sha256'].items():
        if u.sha(run.TREE/name)!=digest:raise ValueError('Original preflight model/dispatcher/driver changed')
    for index,(train,query,nt,nq) in enumerate([('trainx','probex',99352343,1128164),('fulltrain','qual',100480507,2817131)]):
        if plan['full_identity_disjointness'][index]!=dict(training_rows=nt,query_rows=nq,all_training_identities_scanned=True,overlap=0):
            raise ValueError('Both full identity-disjointness scans required')
        for split,count in [(train,nt),(query,nq)]:
            for key in ['user_idxs','item_idxs','ratings','dates']:
                if np.load(run.TREE/f'data/{split}/{key}.npy',mmap_mode='r').shape!=(count,):
                    raise ValueError('Complete original dataset required')
    if np.any(np.load(run.TREE/'data/qual/ratings.npy',mmap_mode='r')):
        raise ValueError('Qualifying fitting targets must be zero')
    fresh=u.read(u.BASE/'fresh-training-plan.json')
    if u.sha(run.TREE/'data/qual_ratings/qual_ratings.csv.gz')!=fresh['synthetic_qualifying_payload_sha256']:
        raise ValueError('Original synthetic qualifying CSV differs')
    result=dict(state='complete_original1200_isolated_source_data_build_and_full_output_admission',
        preparation_sha256=u.sha(run.REPORT/'execution-plan.json'),build_sha256=u.sha(build_path),
        verifier_sha256=u.sha(Path(__file__)),binary_sha256=u.sha(binary),native_command=[str(binary),run.NAME],
        cwd=str(run.TREE),factors=1200,neighbors=100,epochs_each=23,qualifying_labels_read=False)
    return result


def complete():
    admission=admit();plan=u.read(run.REPORT/'execution-plan.json');actual=u.read(run.REPORT/'native-run.json')
    if (actual['command']!=admission['native_command'] or actual['cwd']!=str(run.TREE) or actual['returncode']!=0 or
        actual['environment']!={'RAYON_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}):
        raise ValueError('Complete actual original native command/environment required')
    log=run.REPORT/'native.log';text=log.read_text()
    if ([int(i) for i in re.findall(r'^Epoch\s+(\d+)\s*\|',text,re.M)]!=list(range(1,24))*2 or
        'trainx => probex' not in text or 'fulltrain => qual' not in text or
        any(token not in text for token in ['n_feat: 1200,','n_epochs: 23,','max_neighbors: 100,','low_memory: true,','full_su: true,'])):
        raise ValueError('Complete exact original1200 two-phase traces required')
    out=run.TREE/'preds_new';expected=plan['exact_expected_outputs']
    if {p.name for p in out.glob('*.npy')}!={r['name'] for r in expected}:
        raise ValueError('All58 original query/component/training outputs required')
    outputs={}
    for row in expected:
        path=out/row['name'];values=np.load(path,mmap_mode='r',allow_pickle=False)
        if values.shape!=(row['rows'],) or values.dtype!=np.dtype('<f4'):
            raise ValueError('Original complete numeric output schema differs')
        for lo in range(0,len(values),262144):
            if not np.isfinite(values[lo:lo+262144]).all():raise ValueError('Nonfinite original output')
        outputs[str(path.relative_to(u.ROOT))]=u.sha(path)
    native_log=out/(run.NAME+'.out')
    if not native_log.exists():raise ValueError('Native original log missing')
    native_text=native_log.read_text()
    if [int(i) for i in re.findall(r'^Epoch\s+(\d+)\s*\|',native_text,re.M)]!=list(range(1,24))*2:
        raise ValueError('Native persisted log must independently contain both23-epoch traces')
    outputs[str(native_log.relative_to(u.ROOT))]=u.sha(native_log)
    result=dict(state='complete_independent_original1200_both23_phases_and_all58_numeric_outputs',
        admission=admission,native_run_sha256=u.sha(run.REPORT/'native-run.json'),log_sha256=u.sha(log),
        factors=1200,neighbors=100,epochs_each=23,outputs=outputs,qualifying_labels_read=False,
        no_blend_fit_or_score_promotion=True)
    u.write(run.REPORT/'independent-verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--complete',action='store_true');args=parser.parse_args()
    result=complete() if args.complete else admit()
    if not args.complete:u.write(run.REPORT/'independent-admission.json',result)
    print(result['state'],flush=True)
