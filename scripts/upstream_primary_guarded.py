#!/usr/bin/env python3
"""Resume the original native DAG with explicit external-worker/storage holds."""
from pathlib import Path
import upstream_full_reproduce as core

u = core.u
REPORT = u.ROOT/'experiments/upstream-primary-handoff'
CORE_SHA = '436f181b5b73c228c67abf5a945140eb37ef95472d42ecc451f1f60cbe7a7a80'
DELEGATED = {'cfnade-48', 'cfnade-96', 'cfnade-96r'}


def holds(jobs):
    # Even the 1000-factor parallel heap model is delegated to the separately
    # queued mapped backend. Never silently change its factor count or epochs.
    storage = {name for name in jobs if name.startswith('tsvdx4p-')}
    return storage, DELEGATED & jobs.keys()


def configure():
    if u.sha(Path(core.__file__)) != CORE_SHA:
        raise ValueError('Original native scheduler changed')
    plan = u.read(REPORT/'plan.json')
    for name, digest in plan['code_sha256'].items():
        if u.sha(u.ROOT/name) != digest:
            raise ValueError('Handoff code binding changed: '+name)
    jobs = u.read(u.BASE/'registry.json')['jobs']
    storage, delegated = holds(jobs)
    core.LARGE = core.LARGE | storage | delegated
    original_plan = core.plan

    def guarded_plan():
        result, registry = original_plan()
        result['memory_backend_required'] = sorted(core.LARGE-delegated)
        result['externally_running_source_jobs'] = sorted(delegated)
        result['continuation_script_sha256'] = u.sha(Path(__file__))
        result['handoff_plan_sha256'] = u.sha(REPORT/'plan.json')
        u.write(core.REPORT/'inventory.json', result)
        return result, registry

    core.plan = guarded_plan


if __name__ == '__main__':
    configure()
    core.run()
