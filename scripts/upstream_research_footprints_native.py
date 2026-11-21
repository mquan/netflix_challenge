"""Fail-closed Darwin physical/RSS accounting without a vmmap subprocess per PID.

Separate observer candidate; no existing controller imports this module. The
Darwin SDK rusage_info_v0 layout exposes the same physical footprint counter as
vmmap, in bytes. Both process-list and kernel start identities must survive the
sample. A changing peer set fails closed, allowing a supervisor to resample.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class RUsageV0(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups',
            'pageins', 'wired_size', 'resident_size', 'phys_footprint',
            'proc_start_abstime', 'proc_exit_abstime')]


def usage(pid):
    if sys.platform != 'darwin':
        raise RuntimeError('Native Darwin observer requires Darwin')
    if ctypes.sizeof(RUsageV0) != 96 or RUsageV0.phys_footprint.offset != 72:
        raise RuntimeError('Unexpected Darwin rusage_info_v0 ABI')
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    fn = lib.proc_pid_rusage
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    value = RUsageV0()
    if fn(int(pid), 0, ctypes.byref(value)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(pid))
    if not value.proc_start_abstime or value.proc_exit_abstime:
        raise RuntimeError('Live kernel process identity required: ' + str(pid))
    return dict(bytes=int(value.phys_footprint),
                rss_bytes=int(value.resident_size),
                started=int(value.proc_start_abstime))


def processes():
    result = {}
    with subprocess.Popen(
            ['ps', '-axo', 'pid=,ppid=,stat=,rss=,lstart=,command='],
            text=True, stdout=subprocess.PIPE) as process:
        output, _ = process.communicate()
        if process.returncode:
            raise RuntimeError('Process-list observation failed')
        observer_pid = process.pid
    for line in output.splitlines():
        parts = line.strip().split(None, 9)
        if len(parts) != 10:
            raise RuntimeError('Cannot parse complete process identity')
        pid, ppid, state, rss, *rest = parts
        if int(pid) == observer_pid:
            continue  # This exact, already terminal accounting subprocess only.
        if state.startswith('Z'):
            continue
        result[int(pid)] = dict(ppid=int(ppid), state=state,
                                rss_bytes=int(rss) * 1024,
                                started=' '.join(rest[:5]), command=rest[5])
    return result


def selected(rows, extra):
    result = set()
    for pid, row in rows.items():
        tokens = row['command'].split()
        if not tokens:
            raise RuntimeError('Missing process command')
        native = '/target/release/' in tokens[0]
        script = any(Path(t).name.startswith(('upstream_', 'verify_upstream_'))
                     for t in tokens[1:])
        if pid == extra or native or str(ROOT) in row['command'] or script:
            result.add(pid)
    # Count descendants even if exec changes their command outside repo scope.
    while True:
        descendants = {pid for pid, row in rows.items() if row['ppid'] in result}
        larger = result | descendants
        if larger == result:
            return result
        result = larger


def identity(row):
    return row['ppid'], row['started'], row['command']


def footprints(extra=None):
    before = processes()
    peers = selected(before, extra)
    if extra is not None and extra not in peers:
        raise RuntimeError('Required worker absent from process list')
    first = {pid: usage(pid) for pid in sorted(peers)}
    last_samples = {pid: usage(pid) for pid in sorted(peers)}
    after = processes()
    if selected(after, extra) != peers:
        raise RuntimeError('Research peer set changed during accounting')
    result = {}
    for pid in sorted(peers):
        if identity(before[pid]) != identity(after[pid]):
            raise RuntimeError('Process identity changed during accounting')
        last = last_samples[pid]
        if first[pid]['started'] != last['started']:
            raise RuntimeError('Kernel process identity changed during accounting')
        result[str(pid)] = dict(
            bytes=max(first[pid]['bytes'], last['bytes']),
            rss_bytes=max(before[pid]['rss_bytes'], after[pid]['rss_bytes'],
                          first[pid]['rss_bytes'], last['rss_bytes']),
            command=after[pid]['command'])
    return result
