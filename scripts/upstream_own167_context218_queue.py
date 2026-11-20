#!/usr/bin/env python3
"""Serial full study and independent replay under the original numerical recipe."""
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
import upstream_own167_context218_normalization_queue as resource

REPORT=u.ROOT/'experiments/upstream-own167-context218-queue'
WORKER=u.ROOT/'scripts/upstream_own167_context218_mps.py'
VERIFIER=u.ROOT/'scripts/verify_upstream_own167_context218_mps.py'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    REPORT.mkdir(exist_ok=False)
    scripts=[Path(__file__),WORKER,VERIFIER,Path(resource.__file__)]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in scripts}
    def check_frozen():
        for key,digest in frozen.items():
            if u.sha(u.ROOT/key)!=digest:raise ValueError('Frozen study execution changed')
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,
        rule='Exclusive original GPU lock. Reserve11GiB full numerical worker footprint before admission. Own14GiB,total24GiB sustained RSS and compressed-inclusive footprint;16GiB disk floor,swap growth128MiB. All data,H128/rank32,5folds,3seeds,max40/patience4 unchanged. No automatic restart or qualifying access.',
        phases=['prepare_frozen_sources','complete_nested_fitting','independent_original_statistics_and_saved_pipeline_replay']))
    child=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Study supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        previous=resource.memory.memory();good=0
        while good<2:
            check_frozen();workers=resource.footprints();current=resource.memory.memory()
            total=sum(v['bytes'] for v in workers.values())
            admitted=(total+11*2**30<24*2**30 and current['compressor_bytes']<10*2**30 and
                current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20 and shutil.disk_usage(u.ROOT).free>16*2**30)
            good=good+1 if admitted else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_full_study_headroom',workers=workers,memory=current,good_samples=good,observed=stamp()))
            previous=current
            if good<2:time.sleep(10)
        baseline=current
        for phase,command in [('prepare',[sys.executable,str(WORKER)]),
            ('full-nested-fit',[sys.executable,str(WORKER),'--worker']),('independent-replay',[sys.executable,str(VERIFIER)])]:
            check_frozen();high=0
            with (REPORT/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(['nice','-n','10',*command],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                u.write(REPORT/'launch.json',dict(phase=phase,pid=child.pid,command=command,observed=stamp()))
                while child.poll() is None:
                    check_frozen();workers=resource.footprints(child.pid);current=resource.memory.memory()
                    own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0));total=sum(v['bytes'] for v in workers.values());rss=sum(v['rss_bytes'] for v in workers.values())
                    high=high+1 if (total>24*2**30 or rss>24*2**30 or own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or
                        current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                    free=shutil.disk_usage(u.ROOT).free
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,
                        disk_free_bytes=free,consecutive_high_samples=high,observed=stamp()))
                    if high>=3 or free<16*2**30:raise RuntimeError('Full study resource guard exceeded; original partial artifacts preserved')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Full study phase failed; original artifacts preserved: '+phase)
                child=None
        check_frozen()
        result=u.read(u.ROOT/'experiments/upstream-own167-context218-mps/independent-verification.json')
        if result['state']!='complete_actual_own167_context218_folds_replays_statistics_and_scores':raise ValueError('Independent actual full study proof required')
        u.write(REPORT/'completion.json',dict(state='complete_full_own167_context218_study_and_independent_saved_pipeline_replay',
            observed=stamp(),records=result['records'],nn_improves_nn154=result['nn_improves_nn154'],nn_improves_nn166=result['nn_improves_nn166'],
            fixed_half_improves_global=result['fixed_half_improves_global'],qualifying_labels_read=False,production_started=False))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=stamp()))
        raise

if __name__=='__main__':main()
