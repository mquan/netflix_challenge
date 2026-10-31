"""Wait for all production RBMs and CPU/S0 trainers before the Bayes queue."""
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
    if len(required) != 12 or len(receipts) != 12:
        return False
    for expected, actual in zip(required, receipts):
        if actual.get('state') != 'verified_complete' or actual.get('metric', 'missing') is not None:
            return False
        if any(actual.get(key) != value for key, value in expected.items() if key != 'path'):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        gate = plan['resource_gate']
        if digest(gate['rbm_plan']) != gate['rbm_plan_sha256']:
            raise ValueError('RBM phase changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(item['path']).read_text()) for item in gate['required_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_BAYES')
            return
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_BAYES_RBM_CPU_S0_AND_NORMAL_MEMORY')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
