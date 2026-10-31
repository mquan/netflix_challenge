"""Allow AutoRec to replace the primary GPU queue only after all seven audits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def ready(plan, processes, pressure, receipts):
    gate = plan['resource_gate']
    if pressure != '1' or any(p in processes for p in gate['blocked_process_patterns']):
        return False
    expected = gate['required_primary_gpu_audits']
    if len(expected) != 7 or len(receipts) != 7:
        return False
    return all(actual.get('state') == 'verified_complete' and actual.get('metric', 'missing') is None and
               all(actual.get(k) == v for k, v in required.items() if k != 'path')
               for required, actual in zip(expected, receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        gate = plan['resource_gate']
        if hashlib.sha256(Path(gate['primary_gpu_plan']).read_bytes()).hexdigest() != gate['primary_gpu_plan_sha256']:
            raise ValueError('Primary GPU plan changed')
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        receipts = [json.loads(Path(s['path']).read_text()) for s in gate['required_primary_gpu_audits']]
        if ready(plan, processes, pressure, receipts):
            print('READY_P0_AUTOREC_GPU_HANDOFF')
            raise SystemExit(0)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_AUTOREC_PRIMARY_GPU_AUDITS_AND_NORMAL_MEMORY')
    raise SystemExit(1)
