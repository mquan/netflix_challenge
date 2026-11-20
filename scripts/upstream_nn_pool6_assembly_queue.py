#!/usr/bin/env python3
"""Run the frozen assembly/check/audit chain only after both admission gates."""
import datetime as dt
from pathlib import Path
import subprocess
import sys
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-nn-pool6-assembly-queue'

def main():
    REPORT.mkdir(exist_ok=False)
    scripts=[u.ROOT/'scripts/upstream_nn_pool6_assembly.py',u.ROOT/'scripts/verify_upstream_nn_pool6_assembly.py']
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),*scripts]}
    gates=[(u.ROOT/'experiments/upstream-nn-pool6-admission-queue/completion.json','complete_all_five_independent_full_pipeline_admissions'),
        (u.ROOT/'experiments/upstream-nn-pool6-original-bank-admission/verification.json','complete_original_bank_expectations_for_all_five_full_refits')]
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=bindings,
        stages=['freeze_three_candidates','independent127_support_and_arithmetic_check','frozen_recovered_audit'],
        qualifying_outcomes_used_for_fitting_or_selection=False))
    while not all(p.exists() for p,_ in gates):
        for key,digest in bindings.items():
            if u.sha(u.ROOT/key)!=digest:raise ValueError('Frozen assembly implementation changed')
        u.write(REPORT/'status.json',dict(state='waiting_for_both_full_admission_gates',
            pending=[str(p.relative_to(u.ROOT)) for p,_ in gates if not p.exists()],observed=dt.datetime.now(dt.timezone.utc).isoformat()))
        time.sleep(10)
    for path,state in gates:
        if u.read(path)['state']!=state:raise ValueError('Required independent admission state differs')
    for stage,command in [('freeze',[sys.executable,str(scripts[0])]),
        ('independent-preaudit',[sys.executable,str(scripts[1])]),
        ('frozen-recovered-audit',[sys.executable,str(scripts[1]),'--audit'])]:
        for key,digest in bindings.items():
            if u.sha(u.ROOT/key)!=digest:raise ValueError('Frozen assembly implementation changed')
        u.write(REPORT/'status.json',dict(state=stage,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
        with (REPORT/(stage+'.log')).open('x') as log:
            subprocess.run(command,cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    u.write(REPORT/'completion.json',dict(state='complete_frozen_candidates_independent_simplex_and_recovered_audit',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),goal_completion_requires_result_review=True))

if __name__=='__main__':main()
