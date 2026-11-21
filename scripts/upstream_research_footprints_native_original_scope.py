"""Native counters with the original research-peer admission scope.

Candidate only: no live controller imports this module. The first native
observer broadened scope to all descendants; transient protected OS ps
children then caused false resource failures. This version restores the
original exact research predicate, retaining every declared numerical worker.
Observation races are retried at most three times; no numerical work retries.
"""
from pathlib import Path
import time
import upstream_research_footprints_native as native

ROOT=native.ROOT


def selected(rows,extra):
    result=set()
    for pid,row in rows.items():
        tokens=row['command'].split()
        if not tokens:raise RuntimeError('Missing process command')
        binary='/target/release/' in tokens[0]
        script=any(Path(token).name.startswith(('upstream_','verify_upstream_')) for token in tokens[1:])
        if pid==extra or binary or str(ROOT) in row['command'] or script:result.add(pid)
    return result


def sample(extra):
    before=native.processes();peers=selected(before,extra)
    if extra is not None and extra not in peers:raise RuntimeError('Required worker absent')
    first={pid:native.usage(pid) for pid in sorted(peers)}
    last={pid:native.usage(pid) for pid in sorted(peers)}
    after=native.processes()
    if selected(after,extra)!=peers:raise RuntimeError('Research peer set changed')
    result={}
    for pid in sorted(peers):
        if native.identity(before[pid])!=native.identity(after[pid]) or first[pid]['started']!=last[pid]['started']:
            raise RuntimeError('Research process identity changed')
        result[str(pid)]=dict(bytes=max(first[pid]['bytes'],last[pid]['bytes']),
            rss_bytes=max(before[pid]['rss_bytes'],after[pid]['rss_bytes'],first[pid]['rss_bytes'],last[pid]['rss_bytes']),
            command=after[pid]['command'])
    return result


def footprints(extra=None):
    for attempt in range(3):
        try:return sample(extra)
        except (RuntimeError,OSError):
            if attempt==2:raise
            time.sleep(.03)
