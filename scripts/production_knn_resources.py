"""Gate item-kNN until every production dependency and joint stage is audited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def ready(plan, processes, pressure, receipts):
    gate = plan['resource_gate']
    if pressure != '1' or any(p in processes for p in gate['blocked_process_patterns']):
        return False
    required = gate['required_audits']
    if len(required) != 16 or len(receipts) != 16:
        return False
    for expected, actual in zip(required, receipts):
        if actual.get('state') != 'verified_complete' or actual.get('metric', 'missing') is not None:
            return False
        if any(actual.get(k) != v for k, v in expected.items() if k != 'path'):
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        for phase in plan['source_phases'].values():
            if hashlib.sha256(Path(phase['plan']).read_bytes()).hexdigest() != phase['plan_sha256']:
                raise ValueError('Source phase changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(s['path']).read_text()) for s in plan['resource_gate']['required_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_KNN')
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_KNN_JOINT_DEPENDENCIES_S0_AND_NORMAL_MEMORY')
    raise SystemExit(1)
