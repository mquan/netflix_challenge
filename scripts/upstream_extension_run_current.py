#!/usr/bin/env python3
"""After the first frozen attempt, evaluate and deploy the fixed fresh extension."""
import argparse
import datetime as dt
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import upstream_source_extension as experiment
import verify_upstream_nonlinear as verifier
from upstream_full_reproduce import group_rss,stop

u,n=experiment.u,experiment.n
REPORT=u.ROOT/'experiments/upstream-extension-run'
FIRST=u.ROOT/'experiments/upstream-phase-production'


def development_reference(reports=None, prepared=None):
    """Use independently verified known-label results, never qualifying scores."""
    reports=reports or [u.ROOT/'experiments/upstream-cf12-context',u.ROOT/'experiments/upstream-cf12-tree']
    prepared=prepared or experiment.RUN
    references=[]
    for root in reports:
        proof_path=root/'independent-verification.json'
        if not proof_path.exists():continue
        evaluation=root/'evaluation.json';manifest=root/'manifest.json';proof=u.read(proof_path)
        if proof.get('state')!='complete' or not proof.get('fixed_average_bitwise_equal') or proof['evaluation_sha256']!=u.sha(evaluation):
            raise ValueError('Development reference verification changed')
        data=u.read(manifest);arm='fixed-half-tree-nn'
        if data.get('qualifying_labels_read') is not False:
            raise ValueError('Development reference must exclude qualifying labels')
        for name in ['labels.npy','users.npy','groups.npy']:
            if data['prepared_sha256'][name]!=u.sha(prepared/name):
                raise ValueError('Development reference row identity differs: '+name)
        row=next(r for r in u.read(evaluation)['records'] if r['arm']==arm)
        checked=next(r for r in proof['records'] if r['arm']==arm)
        vector=root.parent.parent/'artifacts'/root.name/(arm+'.f32')
        if (not math.isfinite(row['rmse']) or not math.isfinite(checked['independent_rmse']) or
                row['prediction_sha256']!=checked['prediction_sha256'] or u.sha(vector)!=checked['prediction_sha256'] or
                abs(row['rmse']-checked['independent_rmse'])>1e-12):
            raise ValueError('Development reference score binding changed')
        references.append(dict(rmse=checked['independent_rmse'],report=str(root),
            evaluation_sha256=u.sha(evaluation),verification_sha256=u.sha(proof_path),manifest_sha256=u.sha(manifest)))
    if not references:raise ValueError('Verified current development reference required')
    return min(references,key=lambda r:r['rmse'])


def first_gate(root=FIRST):
    path=root/'independent-qualifying-verification.json'
    if not path.exists():return 'waiting'
    record=u.read(path)
    if record['audit_report_sha256']!=u.sha(root/'recovered-audit/report.json'):
        raise ValueError('First qualifying audit binding changed')
    frozen=u.read(root/'report.json')
    if (frozen.get('state')!='complete' or frozen.get('qualifying_labels_read') is not False or
        frozen.get('qualifying_rows')!=2817131 or (root/'qualifying-final.f32').stat().st_size!=2817131*4):
        raise ValueError('First attempt is not a complete frozen qualifying vector')
    if record['prediction_sha256']!=frozen['frozen_prediction_sha256'] or record['prediction_sha256']!=u.sha(root/'qualifying-final.f32'):
        raise ValueError('First frozen qualifying vector changed')
    groups=record['groups']
    if set(groups)!={'quiz','test'} or sorted(g['rows'] for g in groups.values())!=[1408342,1408789]:
        raise ValueError('First recovered groups have invalid coverage')
    passed=all(g['rmse']<.85725 for g in groups.values())
    if passed!=record['both_groups_below_target']:raise ValueError('First target boolean disagrees with scores')
    return 'target_already_verified' if passed else 'continue_fixed_extension'


def native_rss():
    rows=[line.split(None,3) for line in subprocess.check_output(['ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines()]
    groups={int(row[1]) for row in rows if len(row)==4 and '/target/release/' in row[3].split()[0]}
    return sum(int(row[2])*1024 for row in rows if len(row)==4 and int(row[1]) in groups)


def guarded(command,log_path):
    with log_path.open('a') as log:
        child=subprocess.Popen(command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                own=group_rss(child.pid);other=native_rss();disk=os.statvfs(u.ROOT);free=disk.f_bavail*disk.f_frsize
                high=high+1 if own+other>27*2**30 else 0
                u.write(REPORT/'worker-status.json',dict(pid=child.pid,rss_bytes=own,native_rss_bytes=other,
                    disk_free_bytes=free,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<16*2**30:raise RuntimeError('Extension guard; partial fold artifacts preserved')
                time.sleep(10)
        except BaseException:stop(child);raise
        if child.returncode:raise RuntimeError('Extension child failed; inspect preserved logs/artifacts')


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            gate=first_gate()
            if gate=='target_already_verified':
                u.write(REPORT/'status.json',dict(state='not_needed_after_first_target_pass',goal_requires_independent_review=True));return
            ready=(experiment.REPORT/'manifest.json').exists()
            if gate=='continue_fixed_extension' and ready:break
            u.write(REPORT/'status.json',dict(state='waiting_for_first_audit_and_fresh_models',first_gate=gate,development_prepared=ready));time.sleep(60)
        while native_rss()>16*2**30:
            u.write(REPORT/'status.json',dict(state='waiting_for_memory'));time.sleep(60)
        with (experiment.REPORT/'run.lock').open('a') as data_lock:
            u.write(REPORT/'status.json',dict(state='waiting_for_development_lock'))
            fcntl.flock(data_lock,fcntl.LOCK_EX)
            experiment.configure();n.verify_prepared()
            reference=development_reference()
            u.write(REPORT/'plan.json',dict(script_sha256=u.sha(Path(__file__)),new_predictions=experiment.NEW,
                development_reference=reference,
                manifest_sha256=u.sha(experiment.REPORT/'manifest.json'),
                source_rule='Fixed all-six fresh native extension. Known-ProbeX user folds select budgets and continuation. Qualifying residuals/scores do not choose predictors, hyperparameters or weights; the prior frozen audit only checks whether the goal already passed.',
                production_gate='Fixed equal tree/NN development average must improve the strongest completed independently verified CF12 context/matching-tree average on identical known rows.'))
            for family in ['tree','nn']:
                u.write(REPORT/'status.json',dict(state='nested_fitting',family=family))
                guarded([sys.executable,str(Path(experiment.__file__)),'--worker',family],REPORT/(family+'.log'))
                guarded([sys.executable,str(Path(__file__)),'--verify-family',family],REPORT/(family+'-replay.log'))
            result=n.summarize()
            # A concurrently completed matching-tree study may strengthen the
            # reference during fitting. Recheck before allowing production.
            reference=development_reference();parent=reference['rmse']
            result.update(parent_best_rmse=parent,development_reference=reference,runner_sha256=u.sha(Path(__file__)))
            for row in result['records']:row['delta_vs_previous_best']=row['rmse']-parent
            u.write(experiment.REPORT/'evaluation.json',result);verifier.score()
            candidate=next(r['rmse'] for r in result['records'] if r['arm']=='fixed-half-tree-nn')
            if candidate>=parent:
                u.write(REPORT/'status.json',dict(state='development_not_improved',candidate=candidate,parent=parent,goal_complete=False));return
        # The paired bank is assembled only after the fixed development gate.
        guarded([sys.executable,str(u.ROOT/'scripts/upstream_extension_bank.py')],REPORT/'bank.log')
        u.write(REPORT/'status.json',dict(state='final_production_running'))
        # Its core fitter owns independent child groups and mirrors actual fitter
        # PIDs to the existing resource-accounting status path.
        with (REPORT/'production.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(u.ROOT/'scripts/upstream_extension_production_current.py')],cwd=u.ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:child.wait()
            except BaseException:stop(child);raise
        if child.returncode:raise RuntimeError('Extension final production failed; outputs preserved')
        u.write(REPORT/'status.json',dict(state='production_audit_complete',goal_requires_independent_review=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-family',choices=['tree','nn']);args=parser.parse_args()
    if args.verify_family:experiment.configure();verifier.replay(args.verify_family)
    else:
        try:run()
        except BaseException as error:
            if not isinstance(error,BlockingIOError):
                u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
            raise
