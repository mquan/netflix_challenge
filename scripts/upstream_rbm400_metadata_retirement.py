"""Explicit, journaled retirement of the exact metadata-only original attempt.

--prepare records identities without signals. --execute requires a reviewed
source-bound protocol and repeats all no-scientific-state checks while the
original owner and data group are frozen. Never touches any trained model.
"""
import argparse
import fcntl
import os
import signal
import sys
import time
from pathlib import Path
import upstream_rbm400_observer_recovery as recovery
import upstream_local_rbm_blend_queue as consumer
import upstream_research_footprints_native as observer

old = recovery.original
REPORT = recovery.REPORT


def exact(captured, worker=False):
    actual = old.identity.process(captured['pid'])
    same = old.same_child(captured, actual) if worker else old.identity.same_owner(captured, actual)
    if not same or actual['pgid'] != captured['pgid'] or actual['ppid'] != captured['ppid']:
        raise ValueError('Exact original live process identity required')
    return actual


def metadata_only():
    recovery.original_plan()
    consumer.check_sources(old.read(consumer.REPORT/'plan.json'))
    if {p.name for p in old.STUDY.iterdir()} != {'data-started.json'}:
        raise ValueError('Original producer has advanced beyond metadata-only start')
    start = old.read(old.STUDY/'data-started.json')
    if start['phase'] != 'data' or start['execution_plan_sha256'] != recovery.ORIGINAL_PLAN_SHA:
        raise ValueError('Exact data-only producer start required')
    if old.ART.exists() and any(old.ART.iterdir()):
        raise ValueError('Original artifact namespace must be empty')
    if consumer.study.REPORT.exists() or consumer.study.ART.exists():
        raise ValueError('Original consumer scientific namespace must be absent')
    for name in ('first-scientific-baseline.json', 'child-ownership.json', 'prerequisites.json', 'completion.json'):
        if (consumer.REPORT/name).exists():
            raise ValueError('Original consumer must still be idle before admission')
    if old.read(consumer.REPORT/'status.json')['state'] != 'waiting_full_actual_RBM_and_original_own15_verification':
        raise ValueError('Original consumer must be waiting on the original producer')
    ownership = {p.name for p in old.REPORT.glob('*-ownership.json')}
    if ownership != {'data-ownership.json', 'child-ownership.json'}:
        raise ValueError('Only the original data worker may have been dispatched')
    return dict(data_start_sha256=old.inputs.sha(old.STUDY/'data-started.json'),
                original_plan_sha256=recovery.ORIGINAL_PLAN_SHA,
                original_baseline_sha256=old.inputs.sha(old.REPORT/'first-scientific-baseline.json'),
                learned_state_existed=False)


def capture():
    metadata = metadata_only()
    owner = old.read(old.REPORT/'monitor.json')
    child = old.read(old.REPORT/'data-ownership.json')
    watch = old.read(old.REPORT/'data-gate.json')['watchdog']
    idle = old.read(consumer.REPORT/'monitor.json')
    command = ' '.join([sys.executable, '-u', str(Path(consumer.__file__).resolve()),
                        '--watch', str(idle['pid']), '--lane-fd', '3'])
    candidates = [pid for pid, row in observer.processes().items()
                  if row['ppid'] == idle['pid'] and row['command'] == command]
    if len(candidates) != 1:
        raise ValueError('Exact original idle consumer watchdog required')
    idle_watch = old.identity.process(candidates[0])
    processes = dict(owner=owner, worker=child, watchdog=watch,
                     consumer_owner=idle, consumer_watchdog=idle_watch)
    for key, value in processes.items():
        exact(value, worker=key == 'worker')
    if child['ppid'] != owner['pid'] or watch['ppid'] != owner['pid']:
        raise ValueError('Original data worker/watch must belong to the exact owner')
    return dict(state='declared_exact_metadata_only_retirement', observed=old.stamp(),
                processes=processes, **metadata)


def wait_state(captured, stopped=False):
    for _ in range(150):
        actual = old.identity.process(captured['pid'])
        if stopped:
            if actual and old.identity.same_owner(captured, actual) and 'T' in actual['state']:
                return
        elif not actual or not old.identity.same_owner(captured, actual):
            return
        time.sleep(.1)
    raise RuntimeError('Exact retirement transition not observed')


def journal(name, value):
    record=dict(value);record.setdefault('observed',old.stamp())
    old.exclusive_json(REPORT/name, record)


def execute():
    if (REPORT/'retirement.json').exists():
        raise ValueError('Completed retirement retained; no replay')
    plan = old.read(REPORT/'retirement-plan.json')
    review = old.read(REPORT/'review.json')
    protocol = old.read(REPORT/'protocol-verification.json')
    if review['state'] != 'complete_independent_native_observer_successor_review' or review['findings']:
        raise ValueError('Concrete clean independent review required')
    for name, digest in protocol['source_sha256'].items():
        if old.inputs.sha(old.ROOT/name) != digest:
            raise ValueError('Reviewed recovery source changed: ' + name)
    if review['protocol_sha256'] != old.inputs.sha(REPORT/'protocol-verification.json'):
        raise ValueError('Exact independently reviewed protocol required')
    if capture()['processes'] != plan['processes']:
        # State is deliberately not part of recorded identity equality below.
        for key, captured in plan['processes'].items():
            exact(captured, worker=key == 'worker')
    metadata = metadata_only()
    for key in ('data_start_sha256', 'original_plan_sha256', 'original_baseline_sha256'):
        if metadata[key] != plan[key]:
            raise ValueError('Original metadata or baseline changed')
    p = plan['processes']
    journal('retirement-begin.json', dict(plan_sha256=old.inputs.sha(REPORT/'retirement-plan.json'), processes=p))
    exact(p['owner']);os.kill(p['owner']['pid'], signal.SIGSTOP)
    wait_state(p['owner'], stopped=True)
    exact(p['worker'], worker=True);os.killpg(p['worker']['pid'], signal.SIGSTOP)
    wait_state(dict(p['worker'], command=p['worker']['worker_command']), stopped=True)
    # Both original execution frames are now frozen; failure leaves them
    # preserved for explicit recovery rather than continuing blindly.
    frozen_metadata = metadata_only()
    if frozen_metadata != metadata:
        raise ValueError('Metadata-only status changed before exact retirement')
    journal('retirement-frozen.json', dict(processes=p, metadata=frozen_metadata,
                learned_state_existed=False, hash_accumulator_only=True))
    exact(p['worker'], worker=True);os.killpg(p['worker']['pid'], signal.SIGKILL)
    wait_state(dict(p['worker'], command=p['worker']['worker_command']))
    journal('data-worker-terminal.json', dict(worker=p['worker'], partial_input_hash_restarted=True,
                learned_state_discarded=False))
    exact(p['owner']);os.kill(p['owner']['pid'], signal.SIGTERM)
    os.kill(p['owner']['pid'], signal.SIGCONT)
    wait_state(p['owner']);wait_state(p['watchdog'])
    # Consumer has never admitted or fitted a scientific child.
    metadata_only();exact(p['consumer_owner'])
    os.kill(p['consumer_owner']['pid'], signal.SIGTERM)
    wait_state(p['consumer_owner']);wait_state(p['consumer_watchdog'])
    locks=[]
    try:
        for name in ('experiments/full-upstream-reproduction/runner.lock',
                     'experiments/upstream-source-extension/run.lock'):
            stream=(old.ROOT/name).open('a');locks.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key, captured in p.items():
            actual=old.identity.process(captured['pid'])
            if actual and (old.same_child(captured,actual) if key=='worker' else old.identity.same_owner(captured,actual)):
                raise ValueError('Original process remains live')
        journal('retirement.json', dict(state='complete_exact_metadata_only_producer_and_idle_consumer_retirement',
                processes=p, **metadata, all_original_processes_terminal=True,
                both_lanes_verified_free=True, original_attempt_retained=True))
    finally:
        for stream in locks:stream.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.prepare and not args.execute:
        REPORT.mkdir(exist_ok=True)
        journal('retirement-plan.json', capture())
    elif args.execute and not args.prepare:execute()
    else:parser.error('Choose exactly one explicit mode')
