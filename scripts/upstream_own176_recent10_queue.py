#!/usr/bin/env python3
"""Run the complete own176 study after original temporal and matched recovery finish."""
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

REPORT=u.ROOT/'experiments/upstream-own176-recent10-queue'
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def ready():
    serial=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery/completion.json'
    matched=u.ROOT/'experiments/upstream-split3-matched-resume/independent-verification.json'
    return (serial.exists() and matched.exists() and
        u.read(serial).get('state')=='complete_original_TX22_22_then_resumed_same_CF_native_and_owner' and
        u.read(matched).get('state')=='complete_both_matched_arms_original_statistics_scalers_and_actual_saved_model_replays')

def main():
    REPORT.mkdir(exist_ok=False);child=None
    files=[Path(__file__),u.ROOT/'scripts/upstream_own176_recent10_mps.py',u.ROOT/'scripts/verify_upstream_own176_recent10_mps.py',
        u.ROOT/'scripts/upstream_own176_recent10_data.py',Path(resources.__file__)]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    def check():
        for key,h in frozen.items():
            if u.sha(u.ROOT/key)!=h:raise ValueError('Frozen complete own176 execution changed')
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,
        rule='Wait full independently verified originalTX22/22 AND complete interrupted matched-study recovery. Exclusive GPU lock. Reserve14GiB for full403-input original fitting/verifier,matching owned14GiB limit,total24GiB RSS/physical footprint;compressor<5GiB,stable swap,admissiondisk>19GiB,continuousfloor16GiB,swapgrowth128MiB. All176raw/all90contexts,5folds/3seeds/H128/r32/full40patience4 unchanged. No automatic restart, subset or qualifying fit.'))
    def interrupted(signum,frame):raise KeyboardInterrupt('Full own176 supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        # Do not hold the GPU lock while waiting for another GPU study. It must
        # be able to finish regardless of which supervisor starts first.
        while not ready():
            check();u.write(REPORT/'status.json',dict(state='waiting_for_completed_original_temporal_and_matched_dependencies',observed=stamp()))
            time.sleep(10)
        lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        previous=memory.memory();good=0
        while good<2:
            check();workers=resources.footprints();current=memory.memory()
            admission=(ready() and sum(v['bytes'] for v in workers.values())+14*2**30<24*2**30 and
                sum(v['rss_bytes'] for v in workers.values())+14*2**30<24*2**30 and current['compressor_bytes']<5*2**30 and
                current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20 and shutil.disk_usage(u.ROOT).free>19*2**30)
            good=good+1 if admission else 0;previous=current
            u.write(REPORT/'status.json',dict(state='waiting_for_complete_original_dependencies_and_full403_input_worker_headroom',workers=workers,memory=current,good_samples=good,observed=stamp()))
            if good<2:time.sleep(10)
        baseline=current
        for phase,args in [('prepare-complete-sources',['upstream_own176_recent10_mps.py']),('all-five-full-fits',['upstream_own176_recent10_mps.py','--worker']),('independent-full-statistics-and-model-replay',['verify_upstream_own176_recent10_mps.py'])]:
            check();high=0
            with (REPORT/(phase+'.log')).open('x') as log:
                command=[sys.executable,str(u.ROOT/'scripts'/args[0]),*args[1:]]
                child=subprocess.Popen(['nice','-n','10',*command],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                u.write(REPORT/'launch.json',dict(phase=phase,pid=child.pid,command=command,observed=stamp()))
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    high=high+1 if (sum(v['bytes'] for v in workers.values())>24*2**30 or sum(v['rss_bytes'] for v in workers.values())>24*2**30 or
                        own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,consecutive_high_samples=high,observed=stamp()))
                    if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Full own176 resource guard exceeded; actual artifacts retained')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Full own176 phase failed; actual evidence retained: '+phase)
                child=None
        path=u.ROOT/'experiments/upstream-own176-recent10-mps/independent-verification.json';proof=u.read(path)
        if proof['state']!='complete_actual_own176_all_fits_original_statistics_scalers_and_saved_model_replays':raise ValueError('Actual complete neural verification required')
        u.write(REPORT/'completion.json',dict(state='complete_all_own176_original_fits_and_independent_full_pipeline_replays',observed=stamp(),verification_sha256=u.sha(path),
            nn_rmse=proof['nn_rmse'],fixed_half_tree154_rmse=proof['fixed_half_tree154_rmse'],qualifying_labels_read=False,production_started=False))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=stamp()));raise

if __name__=='__main__':main()
