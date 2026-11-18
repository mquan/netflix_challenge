#!/usr/bin/env python3
"""Prepare a fixed six-model fresh source extension on known ProbeX only."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import upstream_voting_blend as parent
import upstream_remote_batch as algebra
from upstream_full_reproduce import receipt_valid

u, n, x = parent.u, parent.n, parent.x
RUN = u.ROOT/'artifacts/upstream-source-extension'
REPORT = u.ROOT/'experiments/upstream-source-extension'
NEW = ['cfnade-12', 'attn-32', 'attn-32-t', 'aex-16', 'aex-64', 'aex-128']
ARMS = dict(control=[], cfnade=[0], attention=[1, 2], autoencoder=[3, 4, 5],
            main=list(range(6)))


def code_hashes():
    return {**parent.code_hashes(),
            str(Path(__file__).relative_to(u.ROOT)): u.sha(Path(__file__))}


def configure():
    n.RUN, n.REPORT, n.code_hashes = RUN, REPORT, code_hashes


def inventory(verify=False):
    jobs = u.read(u.BASE/'registry.json')['jobs']
    result = {}
    for name in NEW:
        path = u.BASE/'fresh-receipts'/(name+'.json')
        if verify or not path.exists():
            result[name] = receipt_valid(name,jobs) if verify else False
            continue
        record = u.read(path)
        if (record['source_commit'] != u.PIN or record['job'] != name or
                record['command'] != jobs[name]['command']):
            raise ValueError('Wrong source extension receipt: '+name)
        result[name] = (set(jobs[name]['outputs']).issubset(record['outputs']) and
                        all((u.FRESH/p).is_file() for p in record['outputs']))
    return result


def added_values(rows):
    """Only complete, hash-verified two-phase native jobs are admissible."""
    ready = inventory(verify=True)
    if not all(ready.values()):
        raise ValueError('Missing complete paired source jobs: '+str([k for k,v in ready.items() if not v]))
    values = []
    sources = {}
    for name in NEW:
        receipt = u.BASE/'fresh-receipts'/(name+'.json')
        sources[str(receipt.relative_to(u.ROOT))] = u.sha(receipt)
        for split, count in [('probex', rows), ('qual', 2817131)]:
            path = u.FRESH/'preds_new'/(name+'.'+split+'.npy')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.shape != (count,) or a.dtype != np.dtype('float32') or not np.isfinite(a).all():
                raise ValueError('Invalid paired extension prediction: '+name+'.'+split)
            sources[str(path.relative_to(u.ROOT))] = u.sha(path)
            if split == 'probex':
                values.append(np.clip(a, 0, 6))
    return np.column_stack(values), sources


def prepare():
    configure()
    if (REPORT/'manifest.json').exists():
        return n.verify_prepared()
    if RUN.exists():
        raise ValueError('Partial extension preparation preserved')
    disk = os.statvfs(u.ROOT)
    if disk.f_bavail*disk.f_frsize < 20*2**30:
        raise ValueError('Source extension needs at least 20 GiB free disk before preparation')
    parent.configure()
    pm = n.verify_prepared()
    configure()
    evaluation = parent.REPORT/'evaluation.json'
    if u.read(parent.REPORT/'independent-verification.json')['evaluation_sha256'] != u.sha(evaluation):
        raise ValueError('Parent development result has not been independently verified')
    u.validate_fresh()
    old = np.load(parent.RUN/'design.npy', mmap_mode='r')
    old_features = np.load(parent.RUN/'features.npy', mmap_mode='r')
    y, users, groups = [np.load(parent.RUN/(key+'.npy')) for key in ['labels','users','groups']]
    native = u.FRESH/'data/probex'
    if not np.array_equal(y, np.load(native/'ratings.npy')) or not np.array_equal(users, np.load(native/'user_idxs.npy')):
        raise ValueError('Extension development identities differ')
    added, sources = added_values(len(y))
    for key in ['ratings', 'user_idxs', 'item_idxs', 'dates']:
        path = native/(key+'.npy')
        sources[str(path.relative_to(u.ROOT))] = u.sha(path)
    REPORT.mkdir(parents=True, exist_ok=True)
    u.write(REPORT/'plan.json', dict(new_predictions=NEW, screen_arms=ARMS,
        parent_manifest_sha256=u.sha(parent.REPORT/'manifest.json'),
        config=n.CONFIG, code_sha256=code_hashes(),
        main_rule='Append all six fixed published models; no post-score feature subset selection. Screen family bundles are diagnostics.',
        qualifying_outcomes_read=False,
        scope='New development experiment; existing 85-column production recipe remains frozen. No nonlinear worker starts automatically.'))
    scored = algebra.screen(old, added, y, groups, ARMS)
    if abs(x.metric(scored['control']['prediction'],y)-pm['baseline_rmse']) > 1e-9:
        raise ValueError('Extension parent ridge control differs')
    RUN.mkdir(parents=True)
    z = np.lib.format.open_memmap(RUN/'design.npy', mode='w+', dtype=np.float64,
                                  shape=(len(y),old.shape[1]+len(NEW)))
    features = np.lib.format.open_memmap(RUN/'features.npy', mode='w+', dtype=np.float32,
                                         shape=(len(y),old_features.shape[1]+len(NEW)))
    for lo in range(0,len(y),8192):
        hi = min(lo+8192,len(y))
        z[lo:hi] = algebra.augmented_block(old,added,lo,hi)
        features[lo:hi] = np.column_stack([old_features[lo:hi],added[lo:hi]-3.6])
    z.flush(); features.flush()
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    for name,value in [('labels',y),('users',users),('groups',groups),('baseline',scored['main']['prediction'])]:
        np.save(RUN/(name+'.npy'),value)
    scores = []
    for arm,result in scored.items():
        pred = result['prediction']
        rmse = math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(pred,y))/len(y))
        if abs(rmse-x.metric(pred,y)) > 1e-12:
            raise ValueError('Independent source-screen score differs')
        path = RUN/(arm+'-ridge.npy'); np.save(path,pred)
        scores.append(dict(arm=arm,rmse=rmse,prediction_sha256=u.sha(path),
            fold_rmse=[x.metric(pred[groups//2==f],y[groups//2==f]) for f in range(5)]))
    u.write(REPORT/'ridge-screen.json',dict(records=scores,qualifying_outcomes_read=False,
        scope='Fixed five-user-fold ridge diagnostics on known ProbeX; not a qualifying or nonlinear result.'))
    for path in [parent.REPORT/'manifest.json', evaluation, REPORT/'plan.json', REPORT/'ridge-screen.json']:
        sources[str(path.relative_to(u.ROOT))] = u.sha(path)
    result = dict(config=n.CONFIG, versions=n.versions(), code_sha256=code_hashes(),source_commit=u.PIN,
        source_manifests_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},
        baseline_rmse=x.metric(scored['main']['prediction'],y),parent_baseline_rmse=pm['baseline_rmse'],
        rows=len(y),prediction_columns=pm['prediction_columns']+len(NEW),design_columns=z.shape[1],
        nonlinear_features=features.shape[1],new_native_predictions=NEW,arms=pm['arms'],
        isolation='Only verified freshly trained native predictions and known ProbeX labels. Original nested user folds/configuration preserved. Qualifying outcomes remain zero.',
        scope='Adaptive ProbeX source-model extension; no official or untouched Test claim.')
    u.write(REPORT/'manifest.json',result)
    return result


def watch():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            ready = inventory()
            parent_ready = (parent.REPORT/'independent-verification.json').exists()
            if all(ready.values()) and parent_ready:
                break
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_source_jobs',ready=ready,parent_ready=parent_ready))
            time.sleep(60)
        prepare()
        u.write(REPORT/'status.json',dict(state='prepared',nonlinear_training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',choices=['tree','nn'])
    args = parser.parse_args()
    if args.watch:
        watch()
    elif args.worker:
        configure(); n.worker(args.worker)
    elif args.prepare:
        prepare()
    else:
        print(inventory())
