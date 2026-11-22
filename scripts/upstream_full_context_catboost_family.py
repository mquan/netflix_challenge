"""Whole scientific-family accounting for the planned owned CPU tree pipeline.

No observer scope is broadened and no process is launched or admitted here.
The original research-peer observer remains authoritative. These helpers
sum its selected root/descendant processes, reserve the full family cap in
each domain, and allow a future supervisor to preserve the whole group.
"""
import os
import signal
import upstream_research_footprints_native_original_scope as resources
import upstream_cf96_primary_lane as identity

GiB = 2**30
FAMILY_LIMIT = 16 * GiB
TOTAL_LIMIT = 24 * GiB
FLOOR = int(18.75 * GiB)
RESERVE = 64 * 2**20


def matches(captured, actual, group):
    return bool(actual and not actual['state'].startswith('Z')
        and actual['pid'] == captured['pid'] and actual['started'] == captured['started']
        and actual['pgid'] == group
        and actual['command'] in (captured['command'], captured.get('worker_command')))


def members(rows, selected, captured):
    """All original-scope scientific descendants; ordinary observers stay out."""
    root = captured['pid']
    if root not in rows or root not in selected:
        raise RuntimeError('Original dedicated family root absent from native scope')
    row = rows[root]
    if (row['started'] != captured['started']
            or row['command'] not in (captured['command'], captured.get('worker_command'))):
        raise RuntimeError('Original scientific family root identity changed')
    descendants = {root}
    while True:
        expanded = descendants | {pid for pid, value in rows.items() if value['ppid'] in descendants}
        if expanded == descendants: break
        descendants = expanded
    return descendants & set(selected)


def measured(captured, role_captures=None):
    """Wrap the unchanged native observer with stable family ancestry checks."""
    before = resources.native.processes()
    selected = resources.selected(before, captured['pid'])
    workers = resources.footprints(captured['pid'])
    after = resources.native.processes()
    if resources.selected(after, captured['pid']) != selected or set(map(int, workers)) != selected:
        raise RuntimeError('Original research peer set changed during family accounting')
    for pid in selected:
        if resources.native.identity(before[pid]) != resources.native.identity(after[pid]):
            raise RuntimeError('Original research peer identity changed during family accounting')
    group = members(before, selected, captured)
    if group != members(after, selected, captured):
        raise RuntimeError('Scientific family ancestry changed during accounting')
    if not matches(captured, identity.process(captured['pid']), captured['pid']):
        raise RuntimeError('Original dedicated root group identity required')
    for role in (role_captures or {}).values():
        actual = identity.process(role['pid'])
        if actual and actual['started'] == role['started'] and not actual['state'].startswith('Z'):
            if not matches(role, actual, captured['pid']) or role['pid'] not in group:
                raise RuntimeError('Original role escaped its dedicated scientific family')
    return workers, group


def usage(workers, group):
    if not group or any(str(pid) not in workers for pid in group):
        raise RuntimeError('Every actual scientific family counter required')
    for value in workers.values():
        if any(type(value[name]) is not int or value[name] < 0 for name in ('bytes', 'rss_bytes')):
            raise RuntimeError('Nonnegative native physical/RSS counters required')
    return dict(bytes=sum(workers[str(pid)]['bytes'] for pid in group),
                rss_bytes=sum(workers[str(pid)]['rss_bytes'] for pid in group))


def padded(workers, group, root_pid, limit=FAMILY_LIMIT):
    """Reserve exactly the full family cap separately in both domains."""
    if root_pid not in group or type(limit) is not int or limit <= 0:
        raise RuntimeError('Original family root and complete cap required')
    own = usage(workers, group)
    if max(own['bytes'], own['rss_bytes']) > limit:
        raise RuntimeError('Complete scientific family exceeds its original cap')
    values = {pid: dict(row) for pid, row in workers.items()}
    for pid in group:
        values[str(pid)]['bytes'] = 0; values[str(pid)]['rss_bytes'] = 0
    values[str(root_pid)]['bytes'] = limit; values[str(root_pid)]['rss_bytes'] = limit
    return values


def safe(workers, group, current, baseline, free_bytes, *, disk_reserve=RESERVE, limit=FAMILY_LIMIT):
    own = usage(workers, group)
    return bool(max(own['bytes'], own['rss_bytes']) <= limit
        and sum(row['bytes'] for row in workers.values()) < TOTAL_LIMIT
        and sum(row['rss_bytes'] for row in workers.values()) < TOTAL_LIMIT
        and current['compressor_bytes'] < 5 * GiB
        and current['swap_used_bytes'] - baseline['swap_used_bytes'] <= 128 * 2**20
        and free_bytes >= FLOOR + max(RESERVE, disk_reserve))


def readmission(workers, group, root_pid, current, baseline, free_bytes, *, disk_reserve=RESERVE, limit=FAMILY_LIMIT):
    reservation = padded(workers, group, root_pid, limit)
    return safe(reservation, group, current, baseline, free_bytes, disk_reserve=disk_reserve, limit=limit) and current['compressor_bytes'] < 4.5 * GiB


def live_roles(captured, role_captures):
    group = captured['pid']; result = {}
    for name, row in dict(root=captured, **role_captures).items():
        actual = identity.process(row['pid'])
        if matches(row, actual, group): result[name] = actual
    return result


def stop_family(captured, role_captures):
    """Stop the original whole group even after root loss if a role survives.

    The future owner/watchdog must retain both lane descriptors until every
    captured numerical role is terminal. This function never releases them.
    """
    alive = live_roles(captured, role_captures)
    if not alive: return False
    os.killpg(captured['pid'], signal.SIGSTOP)
    return True
