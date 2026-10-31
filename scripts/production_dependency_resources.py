"""Gate the single-thread dependency phase until both S0 lanes finish."""
import argparse
import json
from pathlib import Path
import subprocess


def ready(plan, processes, pressure):
    return pressure == '1' and not any(
        pattern in processes for pattern in plan['resource_gate']['blocked_process_patterns'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    try:
        processes = subprocess.check_output(['ps', '-ww', '-axo', 'command='], text=True)
        pressure = subprocess.check_output(
            ['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        if ready(plan, processes, pressure):
            print('READY_P0_DEPENDENCIES')
            raise SystemExit(0)
    except (OSError, subprocess.CalledProcessError):
        pass
    print('WAIT_P0_DEPENDENCIES_S0_AND_NORMAL_MEMORY')
    raise SystemExit(1)
