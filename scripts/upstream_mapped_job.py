#!/usr/bin/env python3
"""Train a pinned serial temporal recipe with separately verified disk storage."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import tomli
import upstream_repro as u
import upstream_mapped_prepare as m
from upstream_full_reproduce import group_rss,stop

REPORT=u.ROOT/'experiments/full-upstream-reproduction/mapped-training'
ALLOWED={'tsvdx5-150o','tsvdx5-120o','tsvdx5-300','tsvdx5-400','tsvdx5-1000',
         'tsvdx5-120','tsvdx5-150','tsvdx4-160o','tsvdx4-4800lm'}


def verify_candidate():
    u.validate_fresh()
    proof=u.ROOT/'experiments/full-upstream-reproduction/mapped-parity/verification.json'
    record=u.read(proof);prepared=u.read(m.BASE/'preparation.json')
    if not record['pass_all'] or record['preparation']!=prepared:raise ValueError('Candidate lacks current passing parity')
    if record['executable_sha256']!=u.sha(m.BASE/'target/release/tx-storage-parity'):
        raise ValueError('Parity executable changed')
    if set(prepared['generated'])!={'src/tx.rs','src/tx_original.rs','src/tx_storage.rs','src/lib.rs',
          'src/bin/tx-storage-parity.rs','Cargo.toml','data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Missing candidate source bindings')
    for path,digest in prepared['generated'].items():
        if u.sha(m.TREE/path)!=digest:raise ValueError('Candidate changed: '+path)
    for name,digest in prepared['support'].items():
        if u.sha(m.SUPPORT/name)!=digest:raise ValueError('Support source changed: '+name)
    patch=subprocess.check_output(['git','diff','--','src/tx.rs','src/lib.rs','Cargo.toml'],cwd=m.TREE)
    import hashlib
    if hashlib.sha256(patch).hexdigest()!=prepared['patch_sha256']:raise ValueError('Candidate patch changed')
    if u.sha(m.TREE/'data/qual_ratings/qual_ratings.csv.gz')!=u.sha(u.FRESH/'data/qual_ratings/qual_ratings.csv.gz'):
        raise ValueError('Candidate qualifying labels are not zero sentinel')
    changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=m.TREE,text=True).splitlines())
    if changed-{'src/tx.rs','src/lib.rs','Cargo.toml','Cargo.lock','data/qual_ratings/qual_ratings.csv.gz'}:
        raise ValueError('Unexpected tracked source changes')
    old=tomli.loads(m.pinned('Cargo.lock'));new=tomli.loads((m.TREE/'Cargo.lock').read_text())
    root=next(p for p in new['package'] if p['name']=='netflix_prize')
    root['dependencies'].remove('libc')
    if old!=new:raise ValueError('Dependency lock differs beyond direct existing libc dependency')
    return proof,prepared


def layout():
    for split in ['trainx','probex','fulltrain','qual','train','probe','raw']:
        path=m.TREE/'data'/split;target=u.FRESH/'data'/split
        if not path.is_symlink():
            if path.exists():raise ValueError('Unexpected candidate data directory '+str(path))
            path.symlink_to(target,target_is_directory=True)
        if path.resolve()!=target.resolve():raise ValueError('Wrong candidate native data')
    path=m.TREE/'target';target=m.BASE/'target'
    if not path.is_symlink():
        if path.exists():raise ValueError('Unexpected candidate cargo target')
        path.symlink_to(target,target_is_directory=True)
    if path.resolve()!=target.resolve():raise ValueError('Wrong candidate build target')


def regular_rss():
    path=u.ROOT/'experiments/full-upstream-reproduction/status.json'
    if not path.exists():return 0
    state=u.read(path);pid=state.get('child_pid')
    if state['state']!='training' or not pid:return 0
    check=subprocess.run(['ps','-o','command=','-p',str(pid)],capture_output=True,text=True)
    return group_rss(pid) if 'scripts/upstream_repro.py --run-job' in check.stdout else 0


def run(name):
    if name not in ALLOWED:raise ValueError('Not a verified serial temporal recipe')
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    REPORT.mkdir(parents=True,exist_ok=True);receipts=m.BASE/'receipts';receipts.mkdir(exist_ok=True)
    with (m.BASE/'training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof,prepared=verify_candidate();layout()
        inventory=u.read(u.ROOT/'experiments/full-upstream-reproduction/inventory.json')
        if u.sha(u.BASE/'registry.json')!=inventory['registry_sha256']:raise ValueError('Source registry changed')
        jobs=u.read(u.BASE/'registry.json')['jobs'];job=jobs[name]
        source=next(p for p in job['inputs'] if p.startswith('src/bin/'))
        if (m.TREE/source).read_text()!=m.pinned(source):raise ValueError('Recipe dispatcher changed')
        if (m.TREE/'pipeline-new.toml').read_text()!=m.pinned('pipeline-new.toml'):raise ValueError('Pipeline changed')
        if any(n not in {'ingest','newsplit'} for n in job['inputs_from']):raise ValueError('Unexpected model parent')
        if (receipts/(name+'.json')).exists():raise ValueError('Receipt already exists; inspect instead of overwrite')
        if any((m.TREE/f).exists() for f in job['outputs']):raise ValueError('Unreceipted outputs preserved')
        if list((m.BASE/'training-scratch').glob('tx-*.f32')):raise ValueError('Prior mapping files preserved; inspect ownership')
        free=os.statvfs(m.BASE);free=free.f_bavail*free.f_frsize
        # 18.14M fulltrain user-days: 1000 factors need ~78 GB backing before outputs.
        disk_gib={'tsvdx5-1000':110,'tsvdx5-400':65,'tsvdx5-300':55,'tsvdx4-4800lm':60}.get(name,40)
        if free<disk_gib*2**30 or regular_rss()>14*2**30:raise ValueError('Insufficient resource headroom')
        env=u.environment();env.update(CARGO_TARGET_DIR=str(m.BASE/'target'),CARGO_NET_OFFLINE='true',
             TX_MMAP_DIR=str(m.BASE/'training-scratch'),TX_LAZY_DAY='1',RAYON_NUM_THREADS='2')
        started=dt.datetime.now(dt.timezone.utc).isoformat()
        initial={str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in (m.TREE/'preds_new').rglob('*') if p.is_file()}
        state=dict(state='training',job=name,started=started,source_pin=u.PIN,storage=prepared,
                   parity_sha256=u.sha(proof),regular_tree_outputs_modified=False)
        child=None
        try:
            with (REPORT/(name+'.log')).open('w') as log:
                child=subprocess.Popen([str(u.BASE/'target/release/run'),'-n',name],cwd=m.TREE,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                own_high=total_high=0;peak=0
                while child.poll() is None:
                    own=group_rss(child.pid);other=regular_rss();peak=max(peak,own)
                    disk=os.statvfs(m.BASE);disk_free=disk.f_bavail*disk.f_frsize
                    own_high=own_high+1 if own>20*2**30 else 0
                    total_high=total_high+1 if own+other>27*2**30 else 0
                    state.update(heartbeat=dt.datetime.now(dt.timezone.utc).isoformat(),pid=child.pid,
                        rss_bytes=own,other_native_rss_bytes=other,peak_rss_bytes=peak,disk_free_bytes=disk_free)
                    u.write(REPORT/(name+'.status.json'),state)
                    if own_high>=3 or total_high>=3 or disk_free<12*2**30:
                        raise RuntimeError('Mapped-job resource guard; preserve partial outputs')
                    time.sleep(30)
            if child.returncode:raise RuntimeError('Native mapped execution failed: '+str(child.returncode))
            verify_candidate()
            generated=[p for p in (m.TREE/'preds_new').rglob('*') if p.is_file()
                       and initial.get(str(p))!=(p.stat().st_size,p.stat().st_mtime_ns)]
            outputs=sorted(set(generated+[m.TREE/f for f in job['outputs']]))
            if not all(p.is_file() for p in outputs):raise ValueError('Missing declared mapped output')
            record=dict(source_commit=u.PIN,job=name,command=job['command'],started=started,
                finished=dt.datetime.now(dt.timezone.utc).isoformat(),outputs={str(p.relative_to(m.TREE)):u.sha(p) for p in outputs},
                storage=prepared,parity_sha256=u.sha(proof),peak_rss_bytes=peak,
                cargo_lock_sha256=u.sha(m.TREE/'Cargo.lock'),
                execution_environment={k:env[k] for k in ['TX_MMAP_DIR','TX_LAZY_DAY','RAYON_NUM_THREADS',
                    'CARGO_TARGET_DIR','CARGO_NET_OFFLINE','RUSTFLAGS'] if k in env},
                binary_sha256=u.sha(m.BASE/'target/release'/Path(source).stem),
                provenance='Fresh pinned recipe with disclosed disk-backed arrays and parity-verified lazy cache; separate prediction tree, no released prediction inputs, qualifying outcomes zero.')
            u.write(receipts/(name+'.json'),record);state.update(state='complete',receipt_sha256=u.sha(receipts/(name+'.json')))
            u.write(REPORT/(name+'.status.json'),state)
        except BaseException as error:
            if child is not None:stop(child)
            state.update(state='incomplete_preserved',error=repr(error))
            u.write(REPORT/(name+'.status.json'),state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('job',choices=sorted(ALLOWED));run(ap.parse_args().job)
