#!/usr/bin/env python3
"""Guarded full23/23 original1200 producer after original1600 and paired391."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_original1200 as experiment
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identity

u=experiment.u
REPORT=u.ROOT/'experiments/upstream-original1200-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline=None,reserve=0,disk_floor=18.75):
    return (sum(w['bytes'] for w in workers.values())+reserve<=24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<=24*2**30 and
        (baseline is None or current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20) and
        shutil.disk_usage(u.ROOT).free>=disk_floor*2**30)


def watchdog(owner_pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=owner_pid:raise ValueError('Exact original1200 owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(owner_pid)):
            if (REPORT/'owner-release.json').exists():return
            path=REPORT/'child-ownership.json';captured=u.read(path) if path.exists() else None
            stopped=False
            if captured and identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(reason='Original1200 resource owner lost',
                child=captured,actual_child_RAM_preserved=stopped,observed=stamp()))
            return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing original1200 queue/state retained')
    REPORT.mkdir();child=None;captured=None;watch=None;lanes=[]
    files=[Path(__file__),Path(experiment.__file__),Path(experiment.preparation.__file__),
        u.ROOT/'scripts/verify_upstream_original1200.py',experiment.DECL,
        u.ROOT/'experiments/upstream-original1200-preflight/preflight.json',
        Path(resources.__file__),Path(memory.__file__),Path(identity.__file__)]
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),
        factors=1200,neighbors=100,epochs_each=23,expected_numeric_outputs=58,
        rule='Wait independently complete original1600 and paired391H128. CPU preparation/build/admission reserve3GiB,24GiB research,128MiB swapgrowth,comp<4.5GiB admission/5GiB continuous; before preparation/build require20.5GiB free, preserve20GiB through stages for original18.75 floor+1.25output reserve. After build reserve22GiB before nonblocking jointGPU/primary lanes, comp<4.5GiB for3samples;20.125GiB before native start. Native22GiB/24GiB research/128MiB swapgrowth/18.75GiB disk. Original4Rayon/1BLAS. Full23/23 training, all58numericoutputs and separate independent full source/data/build/output verification. Same actual child RAM preserved on guard/monitor loss; no automatic retries or smaller recipe.'))
    def check():
        for relative,digest in bindings.items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Frozen original1200 queue/source changed: '+relative)
    def interrupted(signum,frame):raise KeyboardInterrupt('Original1200 resource owner interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def capacity(reserve,disk_floor,phase,baseline=None):
        good=0
        while True:
            check();workers=resources.footprints(os.getpid());current=memory.memory()
            good=good+1 if safe(workers,current,baseline,reserve,disk_floor) and current['compressor_bytes']<4.5*2**30 else 0
            if good>=3:return current
            u.write(REPORT/'status.json',dict(state='waiting_original_resource_admission',next_phase=phase,
                good_samples=good,workers=workers,memory=current,observed=stamp()));time.sleep(10)
    def run_owned(command,cwd,env,phase,log_path,baseline,disk_floor,own_cap=None):
        nonlocal child,captured
        check()
        with log_path.open('x') as log:
            captured=None
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            captured=identity.process(child.pid)
            if not captured or captured['pgid']!=child.pid or str(command[0]) not in captured['command']:
                if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                raise ValueError('Dedicated actual original1200 stage required')
            u.write(REPORT/'child-ownership.json',captured)
            u.write(REPORT/(phase+'-ownership.json'),captured)
            while child.poll() is None:
                check();workers=resources.footprints(child.pid);current=memory.memory()
                groups={line.split()[0]:int(line.split()[1]) for line in
                    subprocess.check_output(['ps','-axo','pid=,pgid='],text=True).splitlines() if len(line.split())==2}
                own=dict(bytes=sum(w['bytes'] for pid,w in workers.items() if groups.get(pid)==child.pid),
                    rss_bytes=sum(w['rss_bytes'] for pid,w in workers.items() if groups.get(pid)==child.pid))
                if (not safe(workers,current,baseline,disk_floor=disk_floor) or
                    (phase!='native' and current['compressor_bytes']>=5*2**30) or
                    (own_cap is not None and max(own['bytes'],own['rss_bytes'])>own_cap) or watch.poll() is not None):
                    raise RuntimeError('Original1200 resource/watchdog guard; actual group RAM preserved')
                u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()))
                time.sleep(2)
            result=child.returncode
            child=None;captured=None
            if result:raise RuntimeError('Original1200 stage failed without retry: '+phase)
            return result
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            prerequisites=[(u.ROOT/'experiments/upstream-high-capacity-successor/completion.json',
                'complete_original1600_and_independent_full_output_verification'),
                (u.ROOT/'experiments/upstream-own391-paired-h128-queue/completion.json',
                'complete_matched_B391_neural_arm_actual_fits_replay_and_canonical_scores')]
            while not all(path.exists() for path,_ in prerequisites):
                check();u.write(REPORT/'status.json',dict(state='waiting_original1600_and_paired391H128_complete',observed=stamp()));time.sleep(10)
            for path,state in prerequisites:
                if u.read(path)['state']!=state:raise ValueError('Full independent prerequisite completion required')
                bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
            baseline=capacity(3*2**30,20.5,'prepare')
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],
                    cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for phase,script,args in [('prepare','upstream_original1200.py',['prepare']),
                ('build','upstream_original1200.py',['build']),('admission','verify_upstream_original1200.py',[])]:
                if phase=='build':capacity(3*2**30,20.5,phase,baseline)
                elif phase=='admission':capacity(3*2**30,20,phase,baseline)
                run_owned([sys.executable,'-u',str(u.ROOT/'scripts'/script),*args],u.ROOT,None,phase,
                    REPORT/(phase+'.log'),baseline,20,own_cap=3*2**30)
            for name in ['execution-plan.json','original-build-receipt.json','independent-admission.json']:
                path=experiment.REPORT/name;bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
            gpu=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a')
            primary=(u.ROOT/'experiments/full-upstream-reproduction/runner.lock').open('a');lanes=[gpu,primary]
            while True:
                capacity(22*2**30,20.125,'native',baseline)
                acquired=[]
                try:
                    for lane in lanes:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired.append(lane)
                    check();workers=resources.footprints(os.getpid());current=memory.memory()
                    if safe(workers,current,baseline,22*2**30,20.125) and current['compressor_bytes']<4.5*2**30:break
                except BlockingIOError:pass
                for lane in acquired:fcntl.flock(lane,fcntl.LOCK_UN)
                time.sleep(10)
            admission=u.read(experiment.REPORT/'independent-admission.json')
            env=u.environment();threads={'RAYON_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'};env.update(threads)
            started=stamp();native_baseline=memory.memory()
            run_owned(admission['native_command'],experiment.TREE,env,'native',experiment.REPORT/'native.log',
                native_baseline,18.75,own_cap=22*2**30)
            u.write(experiment.REPORT/'native-run.json',dict(command=admission['native_command'],cwd=str(experiment.TREE),
                returncode=0,started=started,finished=stamp(),environment=threads,qualifying_labels_read=False))
            # Numerical native state is now terminal; release compute lanes
            # before independently rehashing the complete input/output bank.
            for lane in lanes:fcntl.flock(lane,fcntl.LOCK_UN)
            capacity(3*2**30,18.75,'independent_complete',native_baseline)
            run_owned([sys.executable,'-u',str(u.ROOT/'scripts/verify_upstream_original1200.py'),'--complete'],
                u.ROOT,None,'independent_complete',REPORT/'independent-complete.log',native_baseline,18.75,own_cap=3*2**30)
            proof=experiment.REPORT/'independent-verification.json'
            if u.read(proof)['state']!='complete_independent_original1200_both23_phases_and_all58_numeric_outputs':
                raise ValueError('Full independent original1200 completion required')
            u.write(REPORT/'completion.json',dict(state='complete_original1200_full23_23_and_independent58_output_verification',
                proof_sha256=u.sha(proof),observed=stamp(),qualifying_labels_read=False,no_score_promotion=True))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()))
            u.write(REPORT/'owner-release.json',dict(state='all_original1200_children_terminal_after_full_independent_proof'))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identity.same_owner(captured,identity.process(child.pid))):
            os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_child_RAM_preserved=True,
            no_automatic_retry=True,observed=stamp()));raise
    finally:
        for lane in lanes:lane.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args()
    watchdog(args.watch) if args.watch else main()
