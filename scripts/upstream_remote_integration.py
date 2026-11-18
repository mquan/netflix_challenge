#!/usr/bin/env python3
"""Verify remote fresh transfers, then refit the frozen nonlinear recipe."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_nonlinear as n
import verify_upstream_nonlinear as v

u, x = n.u, n.x
PARENT_RUN, PARENT_REPORT = n.RUN, n.REPORT
RUN = u.ROOT/'artifacts/upstream-remote-integration'
REPORT = u.ROOT/'experiments/upstream-remote-integration'
TRANSFER = u.ROOT/'experiments/upstream-remote-12h/bknbrx-100/transfer-manifest.json'
BASE_CODE_HASHES = n.code_hashes


def code_hashes():
    return {**BASE_CODE_HASHES(), str(Path(__file__).relative_to(u.ROOT)):u.sha(Path(__file__)),
            'scripts/verify_upstream_nonlinear.py':u.sha(u.ROOT/'scripts/verify_upstream_nonlinear.py')}


def configure_engine():
    # Reuse the tested fitting engine without editing its frozen original source.
    n.RUN, n.REPORT, n.code_hashes = RUN, REPORT, code_hashes


def receive(path=TRANSFER, root=u.ROOT, data=u.FRESH/'data/probex'):
    root, data = Path(root), Path(data)
    manifest = u.read(path)
    if manifest['source_commit'] != u.PIN or manifest['job'] != 'bknbrx-100':
        raise ValueError('Incorrect remote job/source')
    receipt_path = root/manifest['native_receipt']
    receipt = u.read(receipt_path); receipt_sha = u.sha(receipt_path)
    if receipt_sha != manifest['native_receipt_sha256'] or receipt['source_commit'] != u.PIN or receipt['job'] != manifest['job']:
        raise ValueError('Remote native receipt mismatch')
    if receipt['command'] != manifest['native_command']:
        raise ValueError('Remote command mismatch')
    expected_alignment = {'ratings.npy','user_idxs.npy','item_idxs.npy','dates.npy'}
    if set(manifest['alignment_sha256']) != expected_alignment:
        raise ValueError('Incomplete alignment evidence')
    for name, digest in manifest['alignment_sha256'].items():
        if u.sha(data/name) != digest: raise ValueError('Remote row alignment differs: '+name)
    dr = manifest['fresh_data_receipt']
    if dr['source_commit'] != u.PIN or dr['qualifying_outcomes_zero'] is not True or dr['ratings_sha256'] != manifest['alignment_sha256']['ratings.npy']:
        raise ValueError('Invalid remote data receipt')
    arrays, seen, duplicate = {}, {}, {}
    expected_sources = {f'preds_new/bknbrx-100{suffix}.probex.npy' for suffix in ['', '_ep01', '_ep05']}
    if {r['source'] for r in manifest['outputs']} != expected_sources or len(manifest['outputs']) != 3:
        raise ValueError('Unexpected remote output set')
    for entry in manifest['outputs']:
        source = entry['source']; npz = root/entry['npz']
        if not npz.resolve().is_relative_to((root/'experiments/upstream-remote-12h/outputs').resolve()):
            raise ValueError('Unexpected transfer location')
        if (receipt['outputs'].get(source) != entry['source_npy_sha256'] or
                entry['receipt_bound_sha256'] != entry['source_npy_sha256'] or
                entry['owning_native_receipt_sha256'] != receipt_sha):
            raise ValueError('Export is not bound to native receipt')
        if u.sha(npz) != entry['npz_sha256']: raise ValueError('Changed compressed remote output')
        with np.load(npz, allow_pickle=False) as saved:
            if saved.files != ['prediction']: raise ValueError('Unexpected transfer payload')
            value = saved['prediction']
        if value.dtype != np.float32 or value.shape != (manifest['probex_rows'],) or not np.isfinite(value).all():
            raise ValueError('Malformed remote predictions')
        digest = x.array_sha(value)
        if digest != entry['prediction_content_sha256']: raise ValueError('Remote content mismatch')
        name = Path(source).name.removesuffix('.probex.npy')
        if digest in seen: duplicate[name] = seen[digest]
        else:
            arrays[name] = value; seen[digest] = name
    if manifest['probex_rows'] != dr['rows'] or manifest['probex_rows'] != len(np.load(data/'ratings.npy', mmap_mode='r')):
        raise ValueError('Remote row count mismatch')
    return arrays, dict(transfer_sha256=u.sha(path), receipt_sha256=receipt_sha,
        verified_exports=3, unique_columns=list(arrays), duplicate_of=duplicate,
        alignment_sha256=manifest['alignment_sha256'],
        scope='Receiver verifies NPZ/content/alignment and receipt bindings. Sender verified all native output hashes; fulltrain/qual native files remain on sender.')


def verify_parent():
    manifest = u.read(PARENT_REPORT/'manifest.json')
    if manifest['config'] != n.CONFIG or manifest['versions'] != n.versions() or manifest['code_sha256'] != BASE_CODE_HASHES():
        raise ValueError('Parent code/config/environment changed')
    for name, digest in manifest['prepared_sha256'].items():
        if u.sha(PARENT_RUN/name) != digest: raise ValueError('Parent prepared input changed')
    for name, digest in manifest['source_manifests_sha256'].items():
        if u.sha(u.ROOT/name) != digest: raise ValueError('Parent provenance changed')
    final = u.read(PARENT_REPORT/'evaluation.json')
    verification = u.read(PARENT_REPORT/'independent-verification.json')
    if verification['evaluation_sha256'] != u.sha(PARENT_REPORT/'evaluation.json'):
        raise ValueError('Parent verification changed')
    for row in final['records']:
        if u.sha(PARENT_RUN/(row['arm']+'.f32')) != row['prediction_sha256']:
            raise ValueError('Parent predictions changed')
    return manifest, final


def crossfit(z, y, groups):
    grams, rhs = n.grouped_stats(z, y, groups)
    out = np.empty(len(y)); folds = groups//2
    for fold in range(5):
        w = n.ridge_weights(grams, rhs, np.arange(10)//2 != fold)
        held = folds == fold; out[held] = np.clip(z[held]@w, 1.05, 4.95)
    return out, grams, rhs


def prepare():
    configure_engine()
    if (REPORT/'manifest.json').exists(): return n.verify_prepared()
    if RUN.exists(): raise ValueError('Partial integration preparation exists; preserve before recovery')
    parent, final = verify_parent()
    arrays, transfer = receive()
    if len(arrays) != 2: raise ValueError('Expected two unique received vectors')
    y, users, groups = [np.load(PARENT_RUN/(name+'.npy')) for name in ['labels','users','groups']]
    if len(y) != 1128164: raise ValueError('Wrong ProbeX population')
    old_z = np.load(PARENT_RUN/'design.npy', mmap_mode='r')
    # Explicitly reproduce all three predeclared historical controls.
    data = n.e.collect_expansion()
    fourteen, _ = n.b.evaluate([data[2][name] for name in n.e.ARMS['all-fourteen']], [True]*14, y, groups//2)
    if abs(x.metric(fourteen,y)-0.8677119362283443)>1e-9: raise ValueError('Fourteen control mismatch')
    prior_z = np.delete(old_z, 61, axis=1)  # remove only the later RBM chain rating column
    prior, _, _ = crossfit(prior_z,y,groups); del prior_z
    if abs(x.metric(prior,y)-0.8663200739003126)>1e-9: raise ValueError('Snapshot/context control mismatch')
    old, _, _ = crossfit(old_z,y,groups)
    if abs(x.metric(old,y)-0.866286461848646)>1e-9: raise ValueError('Latest ridge control mismatch')
    RUN.mkdir(parents=True)
    added = np.clip(np.column_stack(list(arrays.values())),0,6)
    design = np.lib.format.open_memmap(RUN/'design.npy',mode='w+',dtype=np.float64,
                                     shape=(len(y),old_z.shape[1]+added.shape[1]))
    for lo in range(0,len(y),8192):
        hi=min(lo+8192,len(y))
        design[lo:hi]=np.column_stack([old_z[lo:hi,:-1],added[lo:hi],old_z[lo:hi,-1:]])
    design.flush()
    baseline, grams, rhs = crossfit(design,y,groups)
    features = np.load(PARENT_RUN/'features.npy',mmap_mode='r')
    # Keep every existing nonlinear input unchanged; append the two new rating features.
    np.save(RUN/'features.npy',np.column_stack([features,added-3.6]).astype(np.float32))
    np.savez(RUN/'stats.npz',gram=grams,rhs=rhs)
    for name,value in [('labels',y),('users',users),('groups',groups),('baseline',baseline)]:
        np.save(RUN/(name+'.npy'),value)
    fold_delta=[x.metric(baseline[groups//2==f],y[groups//2==f])-x.metric(old[groups//2==f],y[groups//2==f]) for f in range(5)]
    u.write(REPORT/'ridge-screen.json',dict(receiving_verification=transfer,
        parent_control_rmse=x.metric(old,y),rmse=x.metric(baseline,y),
        delta_vs_parent=x.metric(baseline,y)-x.metric(old,y),fold_delta=fold_delta,
        controls=[x.metric(fourteen,y),x.metric(prior,y),x.metric(old,y)],
        scope='Fixed fresh bknbrx terminal/ep01 addition, duplicate ep05 excluded. Global rating weights; original fourteen-only interactions unchanged.'))
    sources=[TRANSFER,u.ROOT/u.read(TRANSFER)['native_receipt'],PARENT_REPORT/'manifest.json',
             PARENT_REPORT/'evaluation.json',PARENT_REPORT/'independent-verification.json']
    manifest=dict(config=n.CONFIG,versions=n.versions(),code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in sources},
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        baseline_rmse=x.metric(baseline,y),rows=len(y),prediction_columns=64,design_columns=140,
        parent_nonlinear_rmse=final['records'][-1]['rmse'],remote_columns=list(arrays),
        isolation=parent['isolation'],scope=parent['scope'],arms=parent['arms'])
    u.write(REPORT/'manifest.json',manifest)
    n.event(stage='remote_ridge_complete',rmse=manifest['baseline_rmse'],fold_delta=fold_delta)
    return manifest


def summarize():
    result=n.summarize(); old=u.read(PARENT_REPORT/'evaluation.json')
    old_rows={row['arm']:row for row in old['records']}
    for row in result['records']:
        reference=old_rows[row['arm']]
        row['delta_vs_previous_same_family']=row['rmse']-reference['rmse']
        row['fold_delta_vs_previous_same_family']=[a-b for a,b in zip(row['fold_rmse'],reference['fold_rmse'])]
        row['delta_vs_previous_best']=row['rmse']-old['records'][-1]['rmse']
    result['parent_evaluation_sha256']=u.sha(PARENT_REPORT/'evaluation.json')
    u.write(REPORT/'evaluation.json',result)
    return result


def run():
    def stop(signum,frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(); children=[]; logs=[]
        try:
            for family in ['tree','nn']:
                log=(REPORT/(family+'.log')).open('a'); logs.append(log)
                children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',family],
                    cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in [None,0] for p in children): raise RuntimeError('Integration worker failed')
                u.write(REPORT/'status.json',dict(state='running',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    workers={family:dict(pid=p.pid,returncode=p.poll()) for family,p in zip(['tree','nn'],children)}))
                time.sleep(30)
            if any(p.returncode for p in children):raise RuntimeError('Integration worker failed')
            summarize();u.write(REPORT/'status.json',dict(state='complete'))
        except BaseException as error:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p in children:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            u.write(REPORT/'status.json',dict(state='failed',error=repr(error)));raise
        finally:
            for log in logs:log.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',choices=['tree','nn'])
    ap.add_argument('--verify-family',choices=['tree','nn']);ap.add_argument('--verify-scores',action='store_true')
    args=ap.parse_args();os.nice(10);configure_engine()
    if args.worker:n.worker(args.worker)
    elif args.prepare:prepare()
    elif args.verify_family:v.replay(args.verify_family)
    elif args.verify_scores:v.score()
    else:run()
