#!/usr/bin/env python3
"""Run all original own330 fits/replays after own176, reserving full557-input RAM."""
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
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-own330-queue'
PARKED=None


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def ready():
    path=u.ROOT/'experiments/upstream-own176-recent10-queue/completion.json'
    return path.exists() and u.read(path).get('state')=='complete_all_own176_original_fits_and_independent_full_pipeline_replays'


def inputs_ready():
    path=u.ROOT/'experiments/upstream-own330-neural-inputs/independent-verification.json'
    return path.exists() and u.read(path).get('state')=='complete_all_original330_design_and_nonlinear_input_bits_replayed'


def release_parking():
    if PARKED:
        current=owners.process(PARKED['pid'])
        if owners.same_owner(PARKED,current) and 'T' in current['state']:
            os.kill(PARKED['pid'],signal.SIGCONT)


def park_only_idle_native_queue():
    global PARKED
    if ready():return
    matches=[]
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        pid,cmd=row.strip().split(None,1);tokens=cmd.split()
        if Path(tokens[0]).name.startswith('python') and any(Path(t).name=='upstream_high_capacity_original_queue.py' for t in tokens[1:]):
            matches.append(int(pid))
    if len(matches)>1:raise ValueError('Ambiguous native-queue owners; do not signal either')
    if not matches:return
    expected=owners.process(matches[0]);status=u.ROOT/'experiments/upstream-high-capacity-original/queue-status.json'
    if not expected or 'T' in expected['state'] or owners.children(expected['pid']) or not status.exists():return
    if u.read(status).get('state')!='waiting_for_complete_own176_and_preceding_original_studies' or ready():return
    PARKED=expected
    u.write(REPORT/'priority-parking.json',dict(original_native_queue=expected,own176_still_incomplete=True,
            policy='Only its idle dependency-wait supervisor is parked. No native model exists or is interrupted. Restore same queue after own330 release.'))
    os.kill(expected['pid'],signal.SIGSTOP)
    actual=owners.process(expected['pid'])
    if not owners.same_owner(expected,actual) or 'T' not in actual['state'] or owners.children(expected['pid']) or ready():
        release_parking();PARKED=None


def safe(workers,current,baseline,free,reserve=0):
    return (sum(v['bytes'] for v in workers.values())+reserve<24*2**30 and
            sum(v['rss_bytes'] for v in workers.values())+reserve<24*2**30 and
            current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and free>=18.75*2**30)


def main():
    REPORT.mkdir(exist_ok=False);child=None;watch=None;monitor=owners.process(os.getpid())
    files=[Path(__file__),u.ROOT/'scripts/upstream_own330_mps.py',u.ROOT/'scripts/verify_upstream_own330_mps.py',
           u.ROOT/'scripts/upstream_own330_watchdog.py',u.ROOT/'scripts/upstream_own330_neural_inputs.py',
           u.ROOT/'scripts/verify_upstream_own330_neural_inputs.py',Path(resources.__file__),u.ROOT/'OWN330_NEURAL_PLAN.md']
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen complete own330 study changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Own330 full-study supervisor signal')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while not inputs_ready():
            check();u.write(REPORT/'status.json',dict(state='waiting_for_full_own330_input_operator_replay',observed=stamp()));time.sleep(10)
        park_only_idle_native_queue()
        u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,monitor=monitor,parked_highcap=PARKED,
                rule='Complete330raw/all90contexts,406ridge/556features/557NNinputs. OriginalH128/rank32,5folds/3seeds,40pat4,batch8192,AdamW and exact full eligible statistics. Wait completed own176 before GPU acquisition; reserve16GiB matching child cap,24GiB total RSS/physical,stable swap,18.75GiB disk floor. Independent actual model/statistic/scaler/input replays mandatory. Idle native queue may be parked only before own176 completion, never an active model. All release/error paths restore that same owner; monitor-loss watchdog preserves exact worker RAM. No automatic training restart, feature cut, mixture grid or qualifying refit.'))
        with (REPORT/'watchdog.log').open('x') as log:
            watch=subprocess.Popen([sys.executable,str(u.ROOT/'scripts/upstream_own330_watchdog.py')],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while not ready():
            check()
            if watch.poll() is not None:raise RuntimeError('Exact-owner watchdog exited before full study')
            u.write(REPORT/'status.json',dict(state='waiting_for_completed_original_own176_study',parked_native_queue=PARKED,observed=stamp()));time.sleep(10)
        lane=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lane,fcntl.LOCK_EX)
        previous=memory.memory();good=0
        while good<2:
            check();workers=resources.footprints();current=memory.memory()
            admitted=(safe(workers,current,previous,shutil.disk_usage(u.ROOT).free,16*2**30) and current['compressor_bytes']<5*2**30 and
                      current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20)
            good=good+1 if admitted else 0;previous=current
            if watch.poll() is not None:raise RuntimeError('Exact-owner watchdog exited during admission')
            u.write(REPORT/'status.json',dict(state='waiting_for_complete557_input_memory_reservation',workers=workers,memory=current,good_samples=good,observed=stamp()))
            if good<2:time.sleep(10)
        baseline=current
        for phase,args in [('prepare-full-original-study',['upstream_own330_mps.py']),
                           ('all-five-three-seed-fits',['upstream_own330_mps.py','--worker']),
                           ('independent-all-actual-pipelines',['verify_upstream_own330_mps.py'])]:
            check()
            with (REPORT/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(u.ROOT/'scripts'/args[0]),*args[1:]],
                        cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                captured=owners.process(child.pid)
                if not captured or captured['pgid']!=child.pid:raise RuntimeError('Actual dedicated worker identity missing')
                u.write(REPORT/'active-child.json',dict(phase=phase,process=captured))
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    if not safe(workers,current,baseline,shutil.disk_usage(u.ROOT).free) or max(own['bytes'],own['rss_bytes'])>16*2**30:
                        raise RuntimeError('Complete557-input resource guard exceeded; retain same worker RAM')
                    if watch.poll() is not None:raise RuntimeError('Worker monitor-loss watchdog exited unexpectedly')
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()))
                    time.sleep(2)
            if child.returncode:raise RuntimeError('Original full study phase failed without automatic restart: '+phase)
            child=None
        path=u.ROOT/'experiments/upstream-own330-mps/independent-verification.json';proof=u.read(path)
        if proof['state']!='complete_actual_own330_all_fits_original_statistics_scalers_and_saved_model_replays':raise ValueError('All complete actual neural pipelines must pass')
        u.write(REPORT/'completion.json',dict(state='complete_all_original330_fits_and_full_actual_saved_pipeline_replays',
                verification_sha256=u.sha(path),nn_rmse=proof['nn_rmse'],fixed_half_tree154_rmse=proof['fixed_half_tree154_rmse'],
                qualifying_labels_read=False,production_started=False,observed=stamp()))
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),original_worker_RAM_preserved_if_live=True,observed=stamp()));raise
    finally:
        release_parking()
        u.write(REPORT/'owner-release.json',dict(monitor=monitor,parked_native_queue_restored_if_same_identity=True,
                original_worker_preserved_on_error=True,observed=stamp()))


if __name__=='__main__':main()
