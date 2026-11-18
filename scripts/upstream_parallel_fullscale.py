#!/usr/bin/env python3
"""Queue the original 1000-factor parallel recipe if the first frozen attempt misses."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import tomli
import numpy as np
import upstream_repro as u
import upstream_parallel_mapped as candidate
import upstream_extension_run as gate
from upstream_full_reproduce import group_rss,stop

JOB='tsvdx4p-1000lm'
REPORT=u.ROOT/'experiments/upstream-parallel-fullscale'
BASE,TREE=candidate.BASE,candidate.TREE


def verify_candidate():
    u.validate_fresh()
    proof=candidate.REPORT/'verification.json';record=u.read(proof)
    prepared=u.read(BASE/'parallel-preparation.json')
    if not record['pass_all'] or record['preparation']!=prepared:raise ValueError('Parallel storage proof does not bind preparation')
    if prepared['source_pin']!=u.PIN or prepared['script_sha256']!=u.sha(Path(candidate.__file__)):
        raise ValueError('Parallel source preparation changed')
    for name,digest in prepared['generated'].items():
        if u.sha(TREE/name)!=digest:raise ValueError('Parallel candidate changed: '+name)
    if u.sha(BASE/'target/release/tsvdx4p-storage-parity')!=record['executable_sha256']:
        raise ValueError('Parallel proof binary changed')
    if u.sha(TREE/'Cargo.lock')!=record['cargo_lock_sha256']:raise ValueError('Parallel dependency lock changed')
    old=tomli.loads(candidate.serial.pinned('Cargo.lock'));new=tomli.loads((TREE/'Cargo.lock').read_text())
    next(p for p in new['package'] if p['name']=='netflix_prize')['dependencies'].remove('libc')
    if old!=new:raise ValueError('Parallel dependencies differ beyond existing direct libc')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()!=u.PIN:
        raise ValueError('Parallel source pin changed')
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=TREE,text=True).splitlines())
    if changed-{'src/tx.rs','src/tsvdx4p.rs','src/lib.rs','Cargo.toml','Cargo.lock','data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Unexpected parallel tracked source changes')
    for name in ['src/bin/tsvdx4p-new.rs','pipeline-new.toml']:
        if (TREE/name).read_text()!=candidate.serial.pinned(name):raise ValueError('Original parallel dispatcher/pipeline changed')
    return proof,prepared


def other_rss(exclude=None):
    rows=[line.split(None,3) for line in subprocess.check_output(['ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines()]
    groups={int(r[1]) for r in rows if len(r)==4 and '/target/release/' in r[3].split()[0]}
    for relative in ['experiments/upstream-extension-run/worker-status.json','experiments/upstream-production/status.json']:
        path=u.ROOT/relative
        if path.exists():
            state=u.read(path)
            if state.get('pid'):groups.add(state['pid'])
    groups.discard(exclude)
    return sum(int(r[2])*1024 for r in rows if len(r)==4 and int(r[1]) in groups)


def setup():
    for split in ['raw','train','probe','trainx','probex','fulltrain','qual']:
        path=TREE/'data'/split;target=u.FRESH/'data'/split
        if not path.is_symlink():
            if path.exists():raise ValueError('Unexpected parallel data path')
            path.symlink_to(target,target_is_directory=True)
        if path.resolve()!=target.resolve():raise ValueError('Incorrect parallel data alias')
    target=TREE/'target'
    if not target.is_symlink():
        if target.exists():raise ValueError('Unexpected parallel target directory')
        target.symlink_to(BASE/'target',target_is_directory=True)
    if target.resolve()!=(BASE/'target').resolve():raise ValueError('Incorrect parallel build target')


def run():
    proof,prepared=verify_candidate();setup()
    jobs=u.read(u.BASE/'registry.json')['jobs'];job=jobs[JOB]
    receipt=BASE/'receipts'/(JOB+'.json');receipt.parent.mkdir(exist_ok=True)
    if receipt.exists():
        record=u.read(receipt)
        if record['source_commit']!=u.PIN or record['job']!=JOB or record['command']!=job['command'] or record['proof_sha256']!=u.sha(proof):
            raise ValueError('Parallel completion receipt differs')
        for name,digest in record['outputs'].items():
            if u.sha(TREE/name)!=digest:raise ValueError('Completed parallel output changed')
        return
    if any((TREE/name).exists() for name in job['outputs']):raise ValueError('Partial parallel outputs preserved')
    scratch=BASE/'fullscale-scratch';scratch.mkdir(exist_ok=True)
    if list(scratch.glob('tx-*.f32')):raise ValueError('Old parallel mapping files preserved')
    free=os.statvfs(BASE);free=free.f_bavail*free.f_frsize
    if free<40*2**30 or other_rss()>12*2**30:raise ValueError('Insufficient full-scale admission headroom')
    env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_NET_OFFLINE='true',
        CARGO_BUILD_JOBS='2',TX_MMAP_DIR=str(scratch),TX_LAZY_DAY='1',RAYON_NUM_THREADS='4')
    out=TREE/'preds_new';out.mkdir(exist_ok=True)
    initial={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    state=dict(state='training',job=JOB,factors=1000,epochs_per_phase=20,logical_chunks=16,physical_workers=4,
        started=dt.datetime.now(dt.timezone.utc).isoformat());child=None;peak=0
    with (BASE/'fullscale.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with (REPORT/'training.log').open('w') as log:
                child=subprocess.Popen([str(u.BASE/'target/release/run'),'-n',JOB],cwd=TREE,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                high=0
                while child.poll() is None:
                    own=group_rss(child.pid);other=other_rss(child.pid);peak=max(peak,own)
                    disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
                    high=high+1 if own>22*2**30 or own+other>27*2**30 else 0
                    state.update(pid=child.pid,rss_bytes=own,other_rss_bytes=other,peak_rss_bytes=peak,
                        disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat())
                    u.write(REPORT/'status.json',state)
                    if high>=3 or free<12*2**30:raise RuntimeError('Parallel full-scale resource guard; partials preserved')
                    time.sleep(10)
            if child.returncode:raise RuntimeError('Parallel native command failed')
            verify_candidate()
            generated=[p for p in out.rglob('*') if p.is_file() and initial.get(str(p))!=(p.stat().st_size,p.stat().st_mtime_ns)]
            outputs=sorted(set(generated+[TREE/name for name in job['outputs']]))
            if not all(p.is_file() for p in outputs):raise ValueError('Missing declared parallel outputs')
            for split,rows in [('probex',1128164),('qual',2817131)]:
                a=np.load(out/(JOB+'.'+split+'.npy'),mmap_mode='r',allow_pickle=False)
                if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Invalid parallel paired vector')
            record=dict(source_commit=u.PIN,job=JOB,command=job['command'],proof_sha256=u.sha(proof),storage=prepared,
                script_sha256=u.sha(Path(__file__)),binary_sha256=u.sha(BASE/'target/release/tsvdx4p-new'),
                started=state['started'],finished=dt.datetime.now(dt.timezone.utc).isoformat(),peak_rss_bytes=peak,
                factors=1000,epochs_per_phase=20,logical_chunks=16,physical_workers=4,qualifying_outcomes_zero=True,
                outputs={str(p.relative_to(TREE)):u.sha(p) for p in outputs},
                provenance='Original published 1000-factor parallel preset, both20-epoch phases, original16 logical chunks; mapped user matrices disclosed and numerically verified. Public low_memory remains true; no daily buffers, rank reduction, pretrained prediction inputs or qualifying outcome fitting.')
            u.write(receipt,record);u.write(REPORT/'native-receipt.json',record)
            state.update(state='complete',receipt_sha256=u.sha(receipt));u.write(REPORT/'status.json',state)
        except BaseException as error:
            if child is not None:stop(child)
            state.update(state='incomplete_preserved',error=repr(error));u.write(REPORT/'status.json',state);raise


def watch():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            first=gate.first_gate();second=gate.first_gate(u.ROOT/'experiments/upstream-extension-production')
            if 'target_already_verified' in [first,second]:
                u.write(REPORT/'status.json',dict(state='not_needed_after_target_pass',goal_requires_independent_review=True));return
            other=other_rss();disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
            if first=='continue_fixed_extension' and other<=12*2**30 and free>40*2**30:break
            u.write(REPORT/'status.json',dict(state='waiting_for_first_audit_and_memory',first_gate=first,
                second_gate=second,other_rss_bytes=other,disk_free_bytes=free));time.sleep(60)
        run()


if __name__=='__main__':watch()
