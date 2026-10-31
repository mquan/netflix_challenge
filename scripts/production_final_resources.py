"""Gate the final fixed blender until all production stages and queues finish."""
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
    if len(required) != 82 or len(receipts) != 82:
        return False
    return all(all(key in actual and actual[key] == value for key, value in expected.items() if key != 'path')
               for expected, actual in zip(required, receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        for spec in plan['source_phases'].values():
            if digest(spec['plan']) != spec['plan_sha256']:
                raise ValueError('Final prerequisite phase changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(item['path']).read_text()) for item in plan['resource_gate']['required_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_FINAL_BLENDER')
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_FINAL_ALL_82_AUDITS_QUEUE_EXIT_AND_NORMAL_MEMORY')
    raise SystemExit(1)
