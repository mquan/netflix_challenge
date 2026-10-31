"""Read-only resource gate for the serialized production RBM queue."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def ready(plan, processes, pressure, receipts):
    if pressure != '1':
        return False
    if any(pattern in processes for pattern in plan['resource_gate']['blocked_process_patterns']):
        return False
    required = plan['resource_gate']['required_cpu_neural_audits']
    if len(receipts) != len(required) or not required:
        return False
    for expected, actual in zip(required, receipts):
        if (actual.get('state') != 'verified_complete' or actual.get('metric', 'missing') is not None or
                actual.get('stage') != expected['stage'] or actual.get('column') != expected['column'] or
                actual.get('plan_sha256') != expected['plan_sha256']):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        gate = plan['resource_gate']
        cpu_plan = Path(gate['cpu_neural_plan'])
        if hashlib.sha256(cpu_plan.read_bytes()).hexdigest() != gate['cpu_neural_plan_sha256']:
            raise ValueError('CPU neural plan changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(
            ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(item['path']).read_text())
                    for item in gate['required_cpu_neural_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_RBM')
            return
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_RBM_CPU_NEURAL_S0_AND_NORMAL_MEMORY')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
