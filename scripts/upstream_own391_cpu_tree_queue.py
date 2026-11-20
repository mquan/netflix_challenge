#!/usr/bin/env python3
"""Bound the complete CPU tree group alongside native work; no GPU locks."""
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
import upstream_repro as u
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory
import upstream_cf96_primary_lane as identity

REPORT=u.ROOT/'experiments/upstream-own391-cpu-tree-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve<24*2**30 and
        sum(w['rss_bytes'] for w in workers.values())+reserve<24*2**30 and current['compressor_bytes']<5*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def watchdog(pid):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=pid:raise ValueError('Exact original tree owner required')
    while not (REPORT/'owner-release.json').exists():
        if not identity.same_owner(owner,identity.process(pid)):
            if (REPORT/'owner-release.json').exists():return
            p=REPORT/'child-ownership.json';captured=u.read(p) if p.exists() else None;stopped=False
            if captured and identity.same_owner(captured,identity.process(captured['pid'])):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(child=captured,actual_group_RAM_preserved=stopped,observed=stamp()));return
        time.sleep(2)


def main():
    if REPORT.exists():raise ValueError('Existing tree queue retained')
    REPORT.mkdir();child=None;captured=None;watch=None
    files=[Path(__file__),u.ROOT/'OWN391_CPU_TREE_PLAN.md',Path(resources.__file__),Path(memory.__file__),Path(identity.__file__),
        u.ROOT/'experiments/upstream-own391-paired-data/independent-verification.json']
    files += [u.ROOT/'scripts'/s for s in ['upstream_tree_pipe.py','upstream_own391_tree_source.py','upstream_own391_tree_worker.py','verify_upstream_own391_cpu_tree.py']]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(REPORT/'monitor.json',identity.process(os.getpid()))
    u.write(REPORT/'plan.json',dict(source_sha256=frozen,created=stamp(),cpu_only=True,
        rule='All617features/618treeinputs,original31leaf/min1000/max2000pat75/3seeds/5nesteduserfolds/2threads. SeparateTorch source andLightGBM processes via bounded binarypipe; oneRAMmatrix,nofeaturediskcache. Reserve8GiB,capcompleteworkerPGID8GiB/research24GiB/compressor5GiB/swapgrowth128MiB/disk18.75GiB. Stablecomp<3GiB admission; noGPU/primarylane/nopeer signals. PreserveactualgroupRAM,noautomaticretry. Allactualsavedtrees/independentoperatorstream/heldprediction/canonicalscores required. Noqualifyingpredictions oroutcomes.'))
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen original CPU tree execution changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Original CPU tree monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    baseline=memory.memory()
    try:
        with (REPORT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            good=0
            while True:
                check();workers=resources.footprints(os.getpid());current=memory.memory()
                admitted=(safe(workers,current,baseline,8*2**30) and current['compressor_bytes']<3*2**30 and
                    shutil.disk_usage(u.ROOT).free>18.75*2**30+128*2**20)
                good=good+1 if admitted else 0
                if good>=3:baseline=current;break
                u.write(REPORT/'status.json',dict(state='waiting_eight_GiB_stable_CPU_headroom',good_samples=good,workers=workers,memory=current,observed=stamp()));time.sleep(10)
            with (REPORT/'watchdog.log').open('x') as log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phases=[('declare','upstream_own391_tree_source.py',['declare']),('full_nested_fit','upstream_own391_tree_worker.py',[]),
                ('independent_saved_model_replay','verify_upstream_own391_cpu_tree.py',[])]
            for phase,script,args in phases:
                check();path=u.ROOT/'scripts'/script
                with (REPORT/(phase+'.log')).open('x') as log:
                    captured=None
                    child=subprocess.Popen([sys.executable,'-u',str(path),*args],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    captured=identity.process(child.pid)
                    if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                        raise ValueError('Dedicated full CPU tree group required')
                    u.write(REPORT/'child-ownership.json',captured);u.write(REPORT/(phase+'-ownership.json'),captured)
                    while child.poll() is None:
                        check();workers=resources.footprints(child.pid);current=memory.memory()
                        groups={s.split()[0]:int(s.split()[1]) for s in subprocess.check_output(['ps','-axo','pid=,pgid='],text=True).splitlines() if len(s.split())==2}
                        own=dict(bytes=sum(w['bytes'] for pid,w in workers.items() if groups.get(pid)==child.pid),rss_bytes=sum(w['rss_bytes'] for pid,w in workers.items() if groups.get(pid)==child.pid))
                        if not safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>8*2**30 or watch.poll() is not None:
                            raise RuntimeError('CPU tree resource/watchdog guard; complete group RAM preserved')
                        u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,group_memory=own,memory=current,observed=stamp()));time.sleep(3)
                    if child.returncode:raise RuntimeError('Original CPU tree phase failed without retry: '+phase)
                    child=None;captured=None
            proof=u.ROOT/'experiments/upstream-own391-cpu-tree/independent-verification.json'
            if u.read(proof)['state']!='complete_independent_matched391_CPU_tree_models_inputs_ridge_predictions_and_canonical_scores':
                raise ValueError('Full original CPU tree actual-model/canonical proof required')
            u.write(REPORT/'completion.json',dict(state='complete_full_budget_matched391_CPU_tree_and_independent_replay',verification_sha256=u.sha(proof),qualifying_labels_read=False,observed=stamp()))
            u.write(REPORT/'status.json',dict(state='complete',observed=stamp()));u.write(REPORT/'owner-release.json',dict(state='all_original_tree_children_terminal_after_full_independent_replay'))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identity.same_owner(captured,identity.process(child.pid))):os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,actual_group_RAM_preserved=True,no_automatic_retry=True,observed=stamp()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=int);args=parser.parse_args();watchdog(args.watch) if args.watch else main()
