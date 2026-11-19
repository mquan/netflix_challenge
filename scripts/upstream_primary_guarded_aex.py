#!/usr/bin/env python3
"""Continue the original DAG while delegating its already-running AEX128 job."""
from pathlib import Path
import upstream_primary_guarded as original

u,core=original.u,original.core
REPORT=u.ROOT/'experiments/upstream-primary-handoff-aex'
DELEGATED=original.DELEGATED|{'aex-128'}


def configure():
    path=REPORT/'plan.json';plan=u.read(path)
    for name,digest in plan['code_sha256'].items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('AEX delegation binding changed: '+name)
    if set(plan['delegated_source_jobs'])!=DELEGATED:raise ValueError('Delegated source jobs changed')
    original.DELEGATED=set(DELEGATED)
    original.configure()
    inherited=core.plan
    def plan_with_delegation():
        result,jobs=inherited()
        result['aex_delegation_plan_sha256']=u.sha(path)
        result['continuation_script_sha256']=u.sha(Path(__file__))
        u.write(core.REPORT/'inventory.json',result);return result,jobs
    core.plan=plan_with_delegation


if __name__=='__main__':
    configure();core.run()
