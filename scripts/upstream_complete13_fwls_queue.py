#!/usr/bin/env python3
"""Finish the existing13-family fit, then checked audits and declared FWLS work."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'experiments/upstream-complete13-fwls-queue'
PYTHON=ROOT/'.venv/bin/python'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):
    path=REPORT/name;temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(observed=dt.datetime.now(dt.timezone.utc).isoformat(),**value),indent=2)+'\n');temporary.replace(path)
def read(path):return json.loads(path.read_text())


def main():
    pid=int(sys.argv[1])
    if REPORT.exists():raise ValueError('Existing full or partial queue evidence retained')
    REPORT.mkdir()
    files=['upstream_own352_fold_bag_assembly.py','verify_upstream_own352_fold_bag_assembly.py',
           'upstream_complete13_calibration.py','verify_upstream_complete13_calibration.py',
           'upstream_own13_fwls.py','verify_upstream_own13_fwls.py',Path(__file__).name]
    bindings={str(Path('scripts')/p):sha(ROOT/'scripts'/p) for p in files}
    for name in ['ALL_COMPLETE13_CALIBRATION_PLAN.md','OWN13_FEATURE_WEIGHTED_STACKING_PLAN.md']:
        bindings[name]=sha(ROOT/name)
    write('plan.json',dict(existing_complete13_worker_pid=pid,source_sha256=bindings,
        rule='Wait exact existing fit process; no restarts. Require frozen13 and verified7; preverify13 before either audit. Then original declared FWLS fit, independent full replay, frozen recovered audit. Preserve all failures.'))
    def check():
        for relative,digest in bindings.items():
            if sha(ROOT/relative)!=digest:raise ValueError('Frozen queue source changed: '+relative)
    while True:
        result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
        if result.returncode!=0 or not result.stdout.strip():break
        if 'scripts/upstream_complete13_calibration.py' not in result.stdout:
            raise ValueError('Existing fit process identity changed; no restart')
        check();write('status.json',dict(state='waiting_exact_existing_complete13_fit',pid=pid));time.sleep(5)
    if read(ROOT/'experiments/upstream-complete13-calibration/frozen.json')['state']!='complete_both_complete13_candidates_frozen_before_any_new_audit':
        raise ValueError('Original complete13 worker did not finish its complete freeze')
    if read(ROOT/'experiments/upstream-own352-fold-bag-assembly/independent-preaudit-verification.json')['state']!='complete_independent_own352_foldbag_127_support_and_all_frozen_arithmetic_before_audit':
        raise ValueError('Original seven-family independent proof required')
    phases=[('verify_complete13','verify_upstream_complete13_calibration.py',[]),
            ('audit_own352_seven','verify_upstream_own352_fold_bag_assembly.py',['--audit']),
            ('audit_complete13','verify_upstream_complete13_calibration.py',['--audit']),
            ('fit_context_fwls','upstream_own13_fwls.py',[]),
            ('verify_context_fwls','verify_upstream_own13_fwls.py',[]),
            ('audit_context_fwls','verify_upstream_own13_fwls.py',['--audit'])]
    for phase,script,args in phases:
        check();write('status.json',dict(state=phase))
        with (REPORT/(phase+'.log')).open('x') as stream:
            child=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write('status.json',dict(state=phase,pid=child.pid))
            while child.poll() is None:
                check();time.sleep(5)
            if child.returncode:raise ValueError('Declared phase failed and preserved: '+phase)
        check()
    write('completion.json',dict(state='complete_all_four_frozen_comparison_audits_and_declared_context_fwls_fit_replay_audit',source_sha256=bindings))
    write('status.json',dict(state='complete'))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if REPORT.exists():write('failure.json',dict(state='failed_preserved_without_restart',error=repr(exc)))
        raise
