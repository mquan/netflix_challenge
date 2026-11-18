#!/usr/bin/env python3
"""Reproduce independent canonical CFNADE phases concurrently within RAM limits."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_cfnade_checkpoint as candidate
import upstream_cfnade_canonical as common
from upstream_full_reproduce import group_rss,stop

BASE,TREE=common.BASE,common.TREE
REPORT=u.ROOT/'experiments/upstream-cfnade-parallel'
PHASES={'x':'trainx','qual':'fulltrain'}


def admit(voting_state,other,active_rss,free):
    return voting_state=='complete' and other+sum(active_rss)+6*2**30<26*2**30 and free>28*2**30


def epoch(phase):
    path=BASE/'parallel-checkpoints'/PHASES[phase]/'latest.json'
    return u.read(path)['epoch'] if path.exists() else 0


def phase_complete(phase):
    query='probex' if phase=='x' else 'qual'
    marker=REPORT/(phase+'-complete.json');output=TREE/'preds_new'/f'cfnade-96.{query}.npy'
    if epoch(phase)!=95 or not marker.exists() or not output.is_file():return False
    record=u.read(marker)
    if record['plan_sha256']!=u.sha(REPORT/'plan.json') or record['terminal_sha256']!=u.sha(output):
        raise ValueError('Completed independent-phase receipt changed')
    return True


def finalize_phase(phase):
    query='probex' if phase=='x' else 'qual';rows=1128164 if phase=='x' else 2817131
    output=TREE/'preds_new'/f'cfnade-96.{query}.npy'
    if epoch(phase)!=95:raise ValueError('Native phase exited before canonical epoch count')
    a=np.load(output,mmap_mode='r',allow_pickle=False)
    if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Invalid phase terminal prediction')
    u.write(REPORT/(phase+'-complete.json'),dict(plan_sha256=u.sha(REPORT/'plan.json'),terminal_sha256=u.sha(output),epochs=95))


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);common.verify_candidate();common.layout()
        support=u.ROOT/'research/cfnade_fast'
        for source,name in [('stage.rs','cfnade-stage.rs'),('export_raw.rs','cfnade-export-raw.rs')]:
            target=TREE/'src/bin'/name
            if target.exists() and u.sha(target)!=u.sha(support/source):raise ValueError('Canonical stage glue changed')
            shutil.copyfile(support/source,target)
        env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_NET_OFFLINE='true',RAYON_NUM_THREADS='2')
        for key in ['CF_EXIT_AFTER_CHECKPOINT','CF_PROOF_OUTPUT','CF_PROOF_INTERRUPT']:env.pop(key,None)
        with (REPORT/'build.log').open('w') as log:
            subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
                '--bin','cfnade-stage','--bin','cfnade-export-raw'],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        common.verify_candidate()
        paths=[u.FRESH/'data'/split/(name+'.npy') for split in ['trainx','probex','fulltrain','qual']
            for name in ['user_idxs','user_cnts','item_idxs','item_cnts','item_years','ratings','dates','is_test']]
        inputs={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
        data_id=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
        source_id=u.sha(candidate.REPORT/'preparation.json');job=u.read(u.BASE/'registry.json')['jobs']['cfnade-96']
        recipe=dict(source_pin=u.PIN,source_id=source_id,data_id=data_id,input_sha256=inputs,
            canonical_reference_command=job['command'],
            execution_commands={phase:[str(BASE/'target/release/cfnade-stage'),phase] for phase in PHASES},
            semantics='Each independent phase calls the same pinned fit driver with original 320/96/95 preset; no shared learning state. Phase-1 fit2 loop and generic fit have the same model initialization, epoch updates and saved predictions.',
            proof_sha256=u.sha(candidate.REPORT/'verification.json'),runner_sha256=u.sha(Path(__file__)),
            common_script_sha256=u.sha(Path(common.__file__)),
            support_sha256={name:u.sha(support/name) for name in ['stage.rs','export_raw.rs']},
            binaries={name:u.sha(BASE/'target/release'/name) for name in ['cfnade-stage','cfnade-export-raw']},
            hidden=320,rank=96,epochs_per_phase=95,qualifying_outcomes_zero=True)
        plan=REPORT/'plan.json'
        if plan.exists():
            if u.read(plan)!=recipe:raise ValueError('Frozen independent-phase recipe changed')
        else:
            if any((TREE/path).exists() for path in job['outputs']):raise ValueError('Unowned canonical outputs exist')
            u.write(plan,recipe)
        env.update(CF_CHECKPOINT_DIR=str(BASE/'parallel-checkpoints'),CF_CHECKPOINT_SOURCE=source_id,CF_CHECKPOINT_DATA=data_id)
        receipts=BASE/'receipts';receipts.mkdir(exist_ok=True);receipt=receipts/'cfnade-96-parallel.json'
        if receipt.exists():
            record=u.read(receipt)
            if record['plan_sha256']!=u.sha(plan):raise ValueError('Wrong canonical completion receipt')
            for name,digest in record['outputs'].items():
                if u.sha(TREE/name)!=digest:raise ValueError('Completed canonical output changed')
            return
        active={};handles={};high=0;started=dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            while not all(phase_complete(phase) for phase in PHASES):
                for phase,child in list(active.items()):
                    if child.poll() is not None:
                        handles.pop(phase).close();active.pop(phase)
                        if child.returncode:raise RuntimeError('Canonical phase failed; checkpoint preserved: '+phase)
                        finalize_phase(phase)
                other=common.other_rss();rss={phase:group_rss(child.pid) for phase,child in active.items()}
                disk=os.statvfs(BASE);free=disk.f_bavail*disk.f_frsize
                voting=u.read(u.ROOT/'experiments/upstream-voting-blend/status.json')['state']
                for phase in PHASES:
                    if phase in active or phase_complete(phase):continue
                    if admit(voting,other,list(rss.values()),free):
                        log=(REPORT/f'{phase}-attempt-{time.time_ns()}.log').open('w');handles[phase]=log
                        active[phase]=subprocess.Popen(recipe['execution_commands'][phase],cwd=TREE,env=env,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        rss[phase]=6*2**30 # Reserve constructor headroom until the next actual sample.
                high=high+1 if other+sum(rss.values())>27*2**30 or free<16*2**30 else 0
                if high>=3 and active:
                    phase='qual' if 'qual' in active else next(iter(active))
                    stop(active.pop(phase));handles.pop(phase).close();high=0
                u.write(REPORT/'status.json',dict(state='training' if active else 'waiting_for_voting_and_memory',
                    epochs={phase:epoch(phase) for phase in PHASES},active={phase:child.pid for phase,child in active.items()},
                    rss_bytes=rss,other_rss_bytes=other,disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                time.sleep(30)
            for phase,child in active.items():
                child.wait()
                if child.returncode:raise RuntimeError('Canonical phase exit failed')
                handles[phase].close()
            active.clear();handles.clear()
            while common.other_rss()>16*2**30:
                u.write(REPORT/'status.json',dict(state='waiting_to_export_raw',epochs={p:95 for p in PHASES}));time.sleep(60)
            with (REPORT/'raw-export.log').open('a') as log:
                child=subprocess.Popen([str(BASE/'target/release/cfnade-export-raw')],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active['export']=child;child.wait()
                if child.returncode:raise RuntimeError('Canonical raw export failed')
                active.clear()
            common.verify_candidate()
            logfiles=[TREE/'preds_new'/f'cfnade-96.phase-{query}.out' for query in ['probex','qual']]
            (TREE/'preds_new/cfnade-96.out').write_text('Original canonical preset, independent source-driver phases; exact optimization/checkpoint patch disclosed.\n'+''.join(p.read_text() for p in logfiles))
            outputs=[p for p in (TREE/'preds_new').rglob('*') if p.is_file()]
            for p in outputs:
                if p.suffix=='.npy':
                    a=np.load(p,mmap_mode='r',allow_pickle=False);rows=2817131 if '.qual.npy' in p.name else 1128164
                    if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Bad canonical output')
            u.write(receipt,dict(source_commit=u.PIN,job='cfnade-96',canonical_reference_command=job['command'],
                execution_commands=recipe['execution_commands'],plan_sha256=u.sha(plan),started=started,
                finished=dt.datetime.now(dt.timezone.utc).isoformat(),outputs={str(p.relative_to(TREE)):u.sha(p) for p in outputs},
                provenance='Fresh canonical capacity/epochs, independent exact source-driver phases, optimizer/RNG resume, same-trajectory raw SGD export; no pretrained predictions or qualifying outcomes.'))
            shutil.copyfile(receipt,REPORT/'native-receipt.json');u.write(REPORT/'status.json',dict(state='complete'))
        except BaseException as error:
            for child in active.values():stop(child)
            for handle in handles.values():handle.close()
            u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise


if __name__=='__main__':run()
