"""Retire only stopped, read-only OS ps children; never resume a model worker.

The unchanged worker guard treats a failed process-list subprocess as an
accounting failure and preserves its frame. Its existing owner alone controls
three-sample readmission. No parent/group signal or output draining is used.
"""
import argparse
import ctypes
import os
from pathlib import Path
import signal
import sys
import time
import upstream_factored_rbm_execution_native as execution

ROOT=execution.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-observer-child-reaper'
PS_COMMAND='ps -axo pid=,ppid=,stat=,rss=,lstart=,command='
MAX_EVENTS=256
MAX_PER_HOUR=32


def executable(pid):
    library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    fn=library.proc_pidpath
    fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32];fn.restype=ctypes.c_int
    buffer=ctypes.create_string_buffer(4096)
    if fn(pid,buffer,len(buffer))<=0:
        raise OSError(ctypes.get_errno(),'Cannot identify observer executable')
    return buffer.value.decode()


def exact_child(captured):
    actual=execution.identity.process(captured['pid'])
    if (not execution.identity.same_owner(captured,actual)
            or actual['ppid']!=captured['ppid'] or actual['pgid']!=captured['pgid']
            or actual['command']!=PS_COMMAND or 'T' not in actual['state']
            or executable(actual['pid'])!='/bin/ps'):
        raise ValueError('Exact stopped immediate platform ps child required')
    return actual


def terminate_child(captured):
    exact_child(captured)
    # Only this metadata PID. Never its parent, process group or SIGCONT.
    os.kill(captured['pid'],signal.SIGKILL)
    for _ in range(40):
        current=execution.identity.process(captured['pid'])
        if not current or not execution.identity.same_owner(captured,current):return
        time.sleep(.05)
    raise RuntimeError('Stopped metadata observer did not become terminal')


def prepare():
    if REPORT.exists():raise ValueError('Existing companion declaration retained')
    execution.check_sources(execution.read(execution.REPORT/'plan.json'))
    owner=execution.read(execution.REPORT/'monitor.json')
    watchdog=execution.read(execution.REPORT/'data-gate.json')['watchdog']
    files=[Path(__file__),ROOT/'scripts/test_upstream_rbm400_observer_child_reaper.py',
           ROOT/'OWN_RBM400_OBSERVER_CHILD_RETIREMENT.md',
           execution.REPORT/'plan.json',execution.REPORT/'monitor.json',
           execution.REPORT/'first-scientific-baseline.json',Path('/bin/ps')]
    execution.durable_directory(REPORT,ROOT,execution.metadata_guard)
    execution.exclusive_json(REPORT/'plan.json',dict(
        state='declared_child_only_readonly_ps_retirement_no_parent_continuation',
        observed=execution.stamp(),owner=owner,watchdog=watchdog,
        max_events=MAX_EVENTS,max_per_hour=MAX_PER_HOUR,
        source_sha256={str(p) if p==Path('/bin/ps') else str(p.relative_to(ROOT)):execution.inputs.sha(p) for p in files}))


def control(plan):
    if plan['max_events']!=MAX_EVENTS or plan['max_per_hour']!=MAX_PER_HOUR:
        raise ValueError('Declared companion rate bounds required')
    for name,digest in plan['source_sha256'].items():
        if execution.inputs.sha(Path(name) if Path(name).is_absolute() else ROOT/name)!=digest:
            raise ValueError('Companion source/control/executable changed')
    execution.check_sources(execution.read(execution.REPORT/'plan.json'))
    for captured in (plan['owner'],plan['watchdog']):
        if not execution.identity.same_owner(captured,execution.identity.process(captured['pid'])):
            raise ValueError('Original live producer owner and watchdog required')


def candidate(plan):
    control(plan)
    path=execution.REPORT/'child-ownership.json'
    if not path.exists():return None
    parent=execution.read(path);actual=execution.identity.process(parent['pid'])
    if not execution.same_child(parent,actual) or 'T' not in actual['state']:return None
    if actual['ppid']!=plan['owner']['pid'] or actual['pgid']!=actual['pid']:
        raise ValueError('Exact paused dedicated scientific worker required')
    phase=actual['command'].split()[-1]
    if phase not in execution.PHASES or actual['command']!=' '.join(execution.command(phase)):
        return None  # Never touch bootstrap or any other child kind.
    gate=execution.read(execution.REPORT/(phase+'-gate.json'))
    if (gate['pid']!=actual['pid'] or gate['command']!=execution.command(phase)
            or gate['execution_plan_sha256']!=execution.inputs.sha(execution.REPORT/'plan.json')
            or gate['baseline_sha256']!=execution.inputs.sha(execution.REPORT/'first-scientific-baseline.json')
            or gate['owner_sha256']!=execution.inputs.sha(execution.REPORT/'monitor.json')
            or not execution.identity.same_owner(gate['watchdog'],execution.identity.process(gate['watchdog']['pid']))):
        raise ValueError('Original source-bound worker dispatch gate required')
    children=[]
    for pid,row in execution.resources.processes().items():
        if row['ppid']==actual['pid'] and row['command']==PS_COMMAND and 'T' in row['state']:
            child=execution.identity.process(pid)
            if not child:continue
            if child['pgid']!=actual['pgid']:raise ValueError('Observer must belong to the exact paused parent group')
            exact_child(child);children.append(child)
    if len(children)>1:raise ValueError('Ambiguous stopped observer children')
    return dict(parent=parent,actual_parent=actual,child=children[0],phase=phase) if children else None


def main():
    plan=execution.read(REPORT/'plan.json');review=execution.read(REPORT/'review.json')
    if (review['state']!='complete_independent_child_only_observer_retirement_review'
            or review['findings'] or review['plan_sha256']!=execution.inputs.sha(REPORT/'plan.json')
            or review['verification_sha256']!=execution.inputs.sha(REPORT/'verification.json')):
        raise ValueError('Clean source-bound companion review required')
    control(plan)
    execution.exclusive_json(REPORT/'monitor.json',execution.identity.process(os.getpid()))
    events=[]
    try:
        while not any((execution.REPORT/name).exists() for name in ('completion.json','failure.json')):
            selected=candidate(plan)
            if selected:
                now=time.monotonic()
                if len(events)>=MAX_EVENTS or sum(now-t<3600 for t in events)>=MAX_PER_HOUR:
                    raise RuntimeError('Companion event bound reached; parent state retained')
                index=len(events)+1
                execution.exclusive_json(REPORT/('event-%04d-begin.json'%index),dict(
                    observed=execution.stamp(),**selected,
                    baseline_sha256=execution.inputs.sha(execution.REPORT/'first-scientific-baseline.json'),
                    discarded_state='readonly OS process listing only',parent_signal_sent=False))
                # Repeat every producer/parent/child identity immediately before
                # the sole permitted signal; fail closed if anything changed.
                again=candidate(plan)
                if again!=selected:raise ValueError('Observer candidate changed before retirement')
                terminate_child(selected['child']);events.append(now)
                execution.exclusive_json(REPORT/('event-%04d-terminal.json'%index),dict(
                    observed=execution.stamp(),child=selected['child'],signal='SIGKILL_child_PID_only',
                    listing_succeeded=False,parent_signal_sent=False,parent_restart=False,
                    same_parent_actual=execution.identity.process(selected['parent']['pid']),
                    continuation='Existing owner must obtain original three samples; unchanged guard handles failed observer'))
                time.sleep(5)
            else:time.sleep(2)
        execution.exclusive_json(REPORT/'completion.json',dict(state='companion_terminal_after_producer_terminal',events=len(events),observed=execution.stamp()))
    except BaseException as error:
        execution.exclusive_json(REPORT/'failure.json',dict(error=repr(error),events=len(events),parent_signalled=False,observed=execution.stamp()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args()
    if args.prepare and not args.start:prepare()
    elif args.start and not args.prepare:main()
    else:parser.error('Explicit prepare or start required')
