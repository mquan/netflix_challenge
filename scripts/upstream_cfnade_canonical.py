#!/usr/bin/env python3
"""Run the original 320/96/95 CFNADE preset with verified exact resume support."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import upstream_repro as u
import upstream_cfnade_checkpoint as candidate
from upstream_full_reproduce import group_rss,stop

BASE,TREE=candidate.BASE,candidate.TREE
REPORT=u.ROOT/'experiments/upstream-cfnade-canonical'
JOB='cfnade-96'


def verify_candidate():
    u.validate_fresh();proof=u.read(candidate.REPORT/'verification.json');prepared=u.read(candidate.REPORT/'preparation.json')
    if not proof['pass_all'] or proof['preparation']!=prepared:raise ValueError('Current passing CF proof required')
    required={'parity'}|{f'{kind}-{epoch}' for epoch in [3,5] for kind in ['create','resume','identity-rejected','tamper-rejected']}
    if {r['name'] for r in proof['checks']}!=required or not all(r['pass_check'] for r in proof['checks']):
        raise ValueError('Incomplete CF checkpoint proof')
    if prepared['script_sha256']!=u.sha(Path(candidate.__file__)):raise ValueError('Checkpoint preparation script changed')
    for name,digest in prepared['files'].items():
        if u.sha(TREE/name)!=digest:raise ValueError('Candidate source changed: '+name)
    if u.sha(TREE/'Cargo.lock')!=prepared['cargo_lock_sha256']:raise ValueError('Candidate lock changed')
    for name,digest in proof['binaries'].items():
        if u.sha(BASE/'target/release'/name)!=digest:raise ValueError('Proof binary changed')
    for name in ['src/bin/cfnade-new.rs','pipeline-new.toml']:
        pinned=subprocess.check_output(['git','show',u.PIN+':'+name],cwd=u.SOURCE)
        if (TREE/name).read_bytes()!=pinned:raise ValueError('Canonical dispatcher or pipeline changed')
    sentinel='data/qual_ratings/qual_ratings.csv.gz'
    if u.sha(TREE/sentinel)!=u.sha(u.FRESH/sentinel):raise ValueError('Candidate qualifying outcomes not zero sentinel')
    return prepared


def layout():
    for name in ['raw','train','probe','trainx','probex','fulltrain','qual']:
        path=TREE/'data'/name;source=u.FRESH/'data'/name
        if not path.is_symlink():
            if path.exists():raise ValueError('Unowned CF native data path')
            path.symlink_to(source,target_is_directory=True)
        if path.resolve()!=source.resolve():raise ValueError('Wrong CF native data path')
    target=TREE/'target'
    if not target.is_symlink():
        if target.exists():raise ValueError('Unowned CF build path')
        target.symlink_to(BASE/'target',target_is_directory=True)
    if target.resolve()!=(BASE/'target').resolve():raise ValueError('Wrong CF build target')


def other_rss():
    groups=set()
    for relative,key in [('experiments/full-upstream-reproduction/status.json','child_pid'),
        ('experiments/full-upstream-reproduction/mapped-training/tsvdx5-150o.status.json','pid'),
        ('experiments/full-upstream-reproduction/qualifying-backfill/status.json','pid'),
        ('experiments/upstream-production/status.json','pid')]:
        path=u.ROOT/relative
        if path.exists():
            record=u.read(path)
            if record.get('state') in ['training','full_probe_refitting','replaying'] and record.get(key):groups.add(record[key])
    return sum(group_rss(pid) for pid in groups)


def memory_ready():
    path=u.ROOT/'experiments/upstream-voting-blend/status.json'
    if not path.exists() or u.read(path)['state']!='complete':return False
    disk=os.statvfs(BASE)
    return other_rss()<16*2**30 and disk.f_bavail*disk.f_frsize>28*2**30


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepared=verify_candidate();layout()
        exporter=TREE/'src/bin/cfnade-export-raw.rs';support=u.ROOT/'research/cfnade_fast/export_raw.rs'
        if exporter.exists() and u.sha(exporter)!=u.sha(support):raise ValueError('Raw exporter source changed')
        shutil.copyfile(support,exporter)
        env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_NET_OFFLINE='true',RAYON_NUM_THREADS='2')
        for key in ['CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:env.pop(key,None)
        with (REPORT/'build.log').open('w') as log:
            subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
                '--bin','cfnade-new','--bin','cfnade-export-raw'],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_candidate()
        paths=[u.FRESH/'data'/split/(name+'.npy') for split in ['trainx','probex','fulltrain','qual']
            for name in ['user_idxs','user_cnts','item_idxs','item_cnts','item_years','ratings','dates','is_test']]
        inputs={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        data_id=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        source_id=u.sha(candidate.REPORT/'preparation.json')
        job=u.read(u.BASE/'registry.json')['jobs'][JOB]
        recipe=dict(source_pin=u.PIN,source_id=source_id,data_id=data_id,input_sha256=inputs,
            canonical_command=job['command'],candidate_proof_sha256=u.sha(candidate.REPORT/'verification.json'),
            runner_sha256=u.sha(Path(__file__)),exporter_sha256=u.sha(support),
            binaries={name:u.sha(BASE/'target/release'/name) for name in ['cfnade-new','cfnade-export-raw']},
            epochs_per_phase=95,hidden=320,rank=96,qualification_labels_zero=True,
            raw_derivation='Same raw SGD trajectory as canonical no-SWA model; checkpoint arrays and derivation tested bit for bit.')
        plan=REPORT/'plan.json'
        if plan.exists():
            if u.read(plan)!=recipe:raise ValueError('Frozen CF run recipe differs')
        else:
            if any((TREE/path).exists() for path in job['outputs']):raise ValueError('Unreceipted CF outputs lack owned run plan')
            u.write(plan,recipe)
        receipts=BASE/'receipts';receipts.mkdir(exist_ok=True);receipt=receipts/(JOB+'.json')
        if receipt.exists():
            record=u.read(receipt)
            if (record['source_commit']!=u.PIN or record['job']!=JOB or record['source_id']!=source_id
                or record['plan_sha256']!=u.sha(plan)):raise ValueError('Wrong completed CF owner receipt')
            for name,digest in record['outputs'].items():
                if u.sha(TREE/name)!=digest:raise ValueError('Completed CF output changed')
            return
        env.update(CF_CHECKPOINT_DIR=str(BASE/'canonical-checkpoints'),CF_CHECKPOINT_SOURCE=source_id,CF_CHECKPOINT_DATA=data_id)
        child=None;attempt=0;started=dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            while True:
                while not memory_ready():
                    u.write(REPORT/'status.json',dict(state='waiting_for_voting_blend_and_memory',job=JOB));time.sleep(60)
                attempt+=1
                path=REPORT/f'attempt-{time.time_ns()}.log'
                with path.open('w') as log:
                    child=subprocess.Popen([str(u.BASE/'target/release/run'),'-n','-f',JOB],cwd=TREE,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0;paused=False
                    while child.poll() is None:
                        own=group_rss(child.pid);other=other_rss();disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
                        high=high+1 if own>11*2**30 or own+other>27*2**30 else 0
                        epochs={}
                        for split in ['trainx','fulltrain']:
                            latest=BASE/'canonical-checkpoints'/split/'latest.json'
                            epochs[split]=u.read(latest)['epoch'] if latest.exists() else 0
                        u.write(REPORT/'status.json',dict(state='training',job=JOB,pid=child.pid,attempt=attempt,
                            epochs=epochs,rss_bytes=own,other_rss_bytes=other,disk_free_bytes=free,
                            heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                        if high>=3 or free<16*2**30:stop(child);paused=True;break
                        time.sleep(30)
                if paused:child=None;continue
                if child.returncode:raise RuntimeError('CF native worker failed; checkpoint and partial outputs preserved')
                break
            for split in ['trainx','fulltrain']:
                if u.read(BASE/'canonical-checkpoints'/split/'latest.json')['epoch']!=95:raise ValueError('Incomplete canonical CF epochs')
            while True:
                while not memory_ready():
                    u.write(REPORT/'status.json',dict(state='waiting_to_export_raw',job=JOB));time.sleep(60)
                with (REPORT/'raw-export.log').open('a') as log:
                    child=subprocess.Popen([str(BASE/'target/release/cfnade-export-raw')],cwd=TREE,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0;paused=False
                    while child.poll() is None:
                        own=group_rss(child.pid);other=other_rss()
                        high=high+1 if own>11*2**30 or own+other>27*2**30 else 0
                        u.write(REPORT/'status.json',dict(state='exporting_raw',job=JOB,pid=child.pid,rss_bytes=own,other_rss_bytes=other))
                        if high>=3:stop(child);paused=True;break
                        time.sleep(30)
                if paused:continue
                if child.returncode:raise RuntimeError('Raw export failed; canonical checkpoints preserved')
                break
            verify_candidate()
            outputs=[p for p in (TREE/'preds_new').rglob('*') if p.is_file()]
            for p in outputs:
                if p.suffix=='.npy':
                    value=np.load(p,mmap_mode='r',allow_pickle=False)
                    rows=2817131 if '.qual.npy' in p.name else 1128164
                    if value.shape!=(rows,) or value.dtype!=np.dtype('float32') or not np.isfinite(value).all():raise ValueError('Invalid canonical prediction')
            u.write(receipt,dict(source_commit=u.PIN,job=JOB,command=job['command'],started=started,
                finished=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=u.sha(plan),source_id=source_id,
                outputs={str(p.relative_to(TREE)):u.sha(p) for p in outputs},
                provenance='Fresh canonical 320/96/95 CFNADE with exact loop/cache optimizations and optimizer/RNG checkpoints; raw no-SWA counterpart derived from the same verified SGD trajectory. No released prediction inputs or qualifying outcomes.'))
            shutil.copyfile(receipt,REPORT/'native-receipt.json');u.write(REPORT/'status.json',dict(state='complete'))
        except BaseException as error:
            if child is not None:stop(child)
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise


if __name__=='__main__':run()
