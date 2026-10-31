"""Wait for all four production kernel stages before six-thread K1 training."""
import argparse
import json
from pathlib import Path
import subprocess

from production_rbm_audit import digest


def ready(plan, processes, pressure, receipts):
    gate = plan['resource_gate']
    if pressure != '1' or any(pattern in processes for pattern in gate['blocked_process_patterns']):
        return False
    required = gate['required_audits']
    if len(required) != 4 or len(receipts) != 4:
        return False
    for expected, actual in zip(required, receipts):
        if actual.get('state') != 'verified_complete' or actual.get('metric', 'missing') is not None:
            return False
        if any(actual.get(key) != value for key, value in expected.items() if key != 'path'):
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        gate = plan['resource_gate']
        if digest(gate['kernel_plan']) != gate['kernel_plan_sha256']:
            raise ValueError('Kernel prerequisite phase changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(item['path']).read_text()) for item in gate['required_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_USER_KNN')
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_USER_KNN_KERNEL_AND_NORMAL_MEMORY')
    raise SystemExit(1)
