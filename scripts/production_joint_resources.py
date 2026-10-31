"""Gate one production joint thread until the dependency phase is audited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def ready(plan, processes, pressure, receipts):
    gate = plan['resource_gate']
    if pressure != '1' or any(p in processes for p in gate['blocked_process_patterns']):
        return False
    required = gate['required_dependency_audits']
    if len(required) != 4 or len(receipts) != len(required):
        return False
    return all(actual.get('state') == 'verified_complete' and actual.get('metric', 'missing') is None and
               actual.get('stage') == expected['stage'] and actual.get('kind') == expected['kind'] and
               actual.get('plan_sha256') == expected['plan_sha256'] for expected, actual in zip(required, receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        gate = plan['resource_gate']
        if hashlib.sha256(Path(gate['dependency_plan']).read_bytes()).hexdigest() != gate['dependency_plan_sha256']:
            raise ValueError('Dependency plan changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(s['path']).read_text()) for s in gate['required_dependency_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_JOINT')
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_JOINT_DEPENDENCIES_S0_AND_NORMAL_MEMORY')
    raise SystemExit(1)
