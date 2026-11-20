#!/usr/bin/env python3
"""Guarded whole-bank admission/fit/replay queue; no retries or scope changes."""
import argparse
import datetime as dt
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
import upstream_cf96_primary_lane as identities

REPORT=u.ROOT/'experiments/upstream-full-fwls-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,reserve=0):
    return (sum(v['bytes'] for v in workers.values())+reserve<24*2**30 and
        sum(v['rss_bytes'] for v in workers.values())+reserve<24*2**30 and
        current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
        current['compressor_bytes']<5*2**30 and shutil.disk_usage(u.ROOT).free>=18.75*2**30)


def watch(supervisor):
    owner=u.read(REPORT/'monitor.json')
    if owner['pid']!=supervisor:raise ValueError('Original queue monitor identity required')
    while not (REPORT/'owner-release.json').exists():
        if not identities.same_owner(owner,identities.process(supervisor)):
            if (REPORT/'owner-release.json').exists():return
            path=REPORT/'child-ownership.json';child=u.read(path) if path.exists() else None;stopped=False
            if child and identities.same_owner(child,identities.process(child['pid'])):
                os.killpg(child['pid'],signal.SIGSTOP);stopped=True
            u.write(REPORT/'watchdog-recovery.json',dict(state='owner_lost_actual_child_RAM_preserved',child=child,stopped=stopped,observed=stamp()))
            return
        time.sleep(2)


def main(prerequisite):
    if REPORT.exists():raise ValueError('Existing guarded full-bank queue retained')
    REPORT.mkdir();child=None;captured=None;watchdog=None
    u.write(REPORT/'monitor.json',identities.process(os.getpid()))
    phases=[('admit_all_inputs','upstream_full_fwls_inputs.py',[]),
        ('verify_all_inputs','verify_upstream_full_fwls_inputs.py',[]),
        ('fit_full_fwls','upstream_full_fwls.py',[]),
        ('verify_full_fwls','verify_upstream_full_fwls.py',[]),
        ('audit_full_fwls','verify_upstream_full_fwls.py',['--audit'])]
    files=[Path(__file__),Path(resources.__file__),Path(memory.__file__),Path(identities.__file__),
        u.ROOT/'OWN_FULL_FEATURE_WEIGHTED_STACKING_PLAN.md']+[u.ROOT/'scripts'/script for _,script,_ in phases]
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,created=stamp(),existing_preaudit_verifier=prerequisite,
        rule='Wait exact existing complete13 preaudit. All413 source entries, complete paired aliases, all distinct signals×14 original gates,lambda30,five userfolds. Separate full data/fit/replay/audit. FourGiB worker reserve/cap,24GiB research physical/RSS,5GiB compressor,128MiB swap growth,18.75GiB disk. CPU-only; no lane lock. Source/guard/watchdog failure suspends actual child RAM; no automatic retry or shrinking.'))
    def check():
        for relative,digest in bindings.items():
            if u.sha(u.ROOT/relative)!=digest:raise ValueError('Original declared full-bank queue source changed: '+relative)
    def interrupted(signum,frame):raise KeyboardInterrupt('Original full-bank monitor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        required=u.ROOT/'experiments/upstream-complete13-calibration/independent-preaudit-verification.json'
        while not required.exists():
            process=identities.process(prerequisite)
            if not process or 'verify_upstream_complete13_calibration.py' not in process['command']:
                raise ValueError('Original prerequisite verifier is terminal without its required proof; no restart')
            check();u.write(REPORT/'status.json',dict(state='waiting_exact_original_complete13_preaudit',pid=prerequisite,observed=stamp()));time.sleep(5)
        if u.read(required)['state']!='complete_independent8191_support_and_ridge10_all_frozen_arithmetic_before_audit':raise ValueError('Original complete13 preaudit required')
        with (REPORT/'watchdog.log').open('x') as stream:
            watchdog=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],
                cwd=u.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        baseline=memory.memory()
        for phase,script,args in phases:
            while True:
                check();workers=resources.footprints();current=memory.memory()
                if safe(workers,current,baseline,reserve=4*2**30):break
                if watchdog.poll() is not None:raise RuntimeError('Original watchdog exited before admission')
                u.write(REPORT/'status.json',dict(state='waiting_four_GiB_before_'+phase,workers=workers,memory=current,observed=stamp()));time.sleep(10)
            with (REPORT/(phase+'.log')).open('x') as stream:
                path=u.ROOT/'scripts'/script
                child=subprocess.Popen([sys.executable,'-u',str(path),*args],cwd=u.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                captured=identities.process(child.pid)
                if not captured or captured['pgid']!=child.pid or str(path) not in captured['command']:
                    if child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
                    raise ValueError('Actual dedicated full-bank child identity required')
                u.write(REPORT/'child-ownership.json',captured)
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory()
                    own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    if not safe(workers,current,baseline) or max(own['bytes'],own['rss_bytes'])>4*2**30 or watchdog.poll() is not None:
                        raise RuntimeError('Full-bank resource/watchdog guard; preserve actual child RAM')
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()));time.sleep(3)
                if child.returncode:raise RuntimeError('Declared original phase failed; no retry: '+phase)
                child=None;captured=None
        audit=u.ROOT/'experiments/upstream-full-fwls/independent-recovered-audit.json'
        if u.read(audit)['state']!='complete_independent_frozen_full_own_bank_fwls_canonical_recovered_audit':raise ValueError('Complete full-model independent recovered audit required')
        u.write(REPORT/'completion.json',dict(state='complete_full_own_bank_inputs_five_actual_models_independent_replay_and_canonical_audit',audit_sha256=u.sha(audit),observed=stamp()))
        u.write(REPORT/'owner-release.json',dict(state='all_owned_full_bank_phases_terminal_after_independent_verification'))
    except BaseException as error:
        if child is not None and child.poll() is None and (captured is None or identities.same_owner(captured,identities.process(child.pid))):
            os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child=captured,original_child_RAM_preserved=True,no_automatic_retry=True,observed=stamp()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prerequisite',type=int);parser.add_argument('--watch',type=int);args=parser.parse_args()
    watch(args.watch) if args.watch else main(args.prerequisite)
