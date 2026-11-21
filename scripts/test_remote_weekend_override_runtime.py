"""Private admission/process fixtures; never signal or allocate scientific jobs."""
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import upstream_remote_weekend_override_runtime as r
import upstream_remote_weekend_override_return_queue as queue
import upstream_remote_weekend_override_handoff as handoff
import upstream_remote_weekend_override_worker as worker
import upstream_remote_weekend_override_context as context_module


SCRIPTS=Path(__file__).resolve().parent
JOB=r.JOBS[0]


def wait_for(condition,seconds=8):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if condition():return
        time.sleep(.03)
    raise AssertionError('Private process fixture did not reach expected state')


def lock_blocked(path):
    with path.open('a') as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(probe,fcntl.LOCK_UN);return False


def terminate_fixture(child):
    if child and child.poll() is None:
        # Only subprocesses created inside this test, never discovered production PIDs.
        child.kill();child.wait(timeout=5)


class RuntimeAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(r,'ROOT',self.root),
            patch.object(r,'PROTOCOL',self.root/'protocol.json')]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def workers(self,physical=1,rss=1):
        return {'1':dict(bytes=physical*2**30,rss_bytes=rss*2**30)}

    def test_both_memory_domains_and_original_swap_baseline(self):
        baseline=dict(swap_used_bytes=1000)
        current=dict(compressor_bytes=0,swap_used_bytes=1000)
        self.assertTrue(r.ready(self.workers(),current,baseline,r.FLOOR+r.META,r.LIMIT,r.META))
        for workers in (self.workers(23,1),self.workers(1,23)):
            self.assertFalse(r.ready(workers,current,baseline,r.FLOOR+r.META,r.LIMIT,r.META))
        current['swap_used_bytes']+=128*2**20+1
        self.assertFalse(r.safe(self.workers(),current,baseline,r.FLOOR+r.META))
        self.assertEqual(baseline,dict(swap_used_bytes=1000))
        for compressor in (4.5*2**30,5*2**30):
            c=dict(compressor_bytes=compressor,swap_used_bytes=1000)
            self.assertFalse(r.ready(self.workers(),c,baseline,r.FLOOR+r.META,0,r.META))
        self.assertFalse(r.safe(self.workers(),dict(compressor_bytes=0,swap_used_bytes=1000),baseline,r.FLOOR))

    def test_continuation_reserves_full_worker_in_each_domain(self):
        current=dict(compressor_bytes=0,swap_used_bytes=0)
        for physical,rss in ((22.5,1),(1,22.5)):
            workers=self.workers(.1,.1);workers['2']=dict(bytes=physical*2**30,rss_bytes=rss*2**30)
            self.assertFalse(r.continuation_ready(workers,1,current,current,r.FLOOR+r.META,r.META))
        self.assertFalse(r.continuation_ready(self.workers(2.01,1),1,current,current,r.FLOOR+r.META,r.META))
        self.assertFalse(r.continuation_ready({},1,current,current,r.FLOOR+r.META,r.META))

    def protocol(self):
        bindings={}
        for name in r.MUST_BIND:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+name)
            bindings[name]=r.ops.sha(p)
        value=dict(state='declared_full_weekend_owner_override_return_adapter',jobs=list(r.JOBS),
            phases=list(r.PHASES),environment=r.ENV,worker_cap_bytes=r.LIMIT,total_cap_bytes=r.TOTAL,
            disk_floor_bytes=r.FLOOR,numpy_version=importlib.metadata.version('numpy'),source_sha256=bindings)
        r.write(r.PROTOCOL,value);return value

    def test_declared_source_closure_and_exact_caps(self):
        original=self.protocol();r.check_protocol()
        for key,value in [('worker_cap_bytes',r.LIMIT+1),('jobs',[JOB]),('phases',['transport']),
                          ('environment',{}),('numpy_version','wrong')]:
            bad=copy.deepcopy(original);bad[key]=value;r.write(r.PROTOCOL,bad)
            with self.subTest(key=key),self.assertRaises(ValueError):r.check_protocol()
        bad=copy.deepcopy(original);del bad['source_sha256'][r.MUST_BIND[0]];r.write(r.PROTOCOL,bad)
        with self.assertRaises(ValueError):r.check_protocol()
        for name in ('scripts/upstream_split3_pilot.py','scripts/upstream_remote15h_native_stream.py'):
            r.write(r.PROTOCOL,original);p=self.root/name;before=p.read_bytes();p.write_text('changed row/download helper')
            with self.subTest(helper=name),self.assertRaises(ValueError):r.check_protocol()
            p.write_bytes(before)
        r.write(r.PROTOCOL,original);(self.root/r.MUST_BIND[0]).write_text('changed')
        with self.assertRaises(ValueError):r.check_protocol()
        with self.assertRaises(ValueError):r.check_bindings({'../outside':'a'*64})

    def test_exact_child_identity_prevents_recycled_or_other_process_signal(self):
        captured=dict(pid=123,pgid=123,started='start',command='bootstrap',worker_command='worker')
        for key,value in [('pid',124),('pgid',122),('started','other'),('command','other')]:
            actual=dict(captured);actual[key]=value
            with self.subTest(key=key),patch.object(r.identity,'process',return_value=actual),patch.object(r.os,'killpg') as kill:
                self.assertFalse(r.stop(captured));kill.assert_not_called()
        actual=dict(captured,command='worker')
        with patch.object(r.identity,'process',return_value=actual),patch.object(r.os,'killpg') as kill:
            self.assertTrue(r.stop(captured));kill.assert_called_once_with(123,signal.SIGSTOP)

    def lockfile(self):
        p=self.root/'experiments/full-upstream-reproduction/runner.lock';p.parent.mkdir(parents=True)
        return p

    def test_inherited_locked_description_required(self):
        p=self.lockfile()
        with p.open('a') as owner,p.open('a') as unrelated:
            os.set_inheritable(owner.fileno(),True)
            with self.assertRaises(ValueError):r.validate_lane(owner.fileno())
            fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);r.validate_lane(owner.fileno())
            os.set_inheritable(unrelated.fileno(),True)
            with self.assertRaises(BlockingIOError):r.validate_lane(unrelated.fileno())
            os.set_inheritable(owner.fileno(),False)
            with self.assertRaises(ValueError):r.validate_lane(owner.fileno())

    def test_stopped_worker_keeps_inherited_lock_after_parent_closes(self):
        p=self.lockfile();lane=p.open('a');child=None
        try:
            fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=subprocess.Popen([sys.executable,'-c',
                'import os,signal,time;os.kill(os.getpid(),signal.SIGSTOP);time.sleep(60)'],
                pass_fds=(lane.fileno(),),start_new_session=True)
            wait_for(lambda: 'T' in (r.identity.process(child.pid) or {}).get('state',''))
            lane.close();self.assertTrue(lock_blocked(p))
            terminate_fixture(child);self.assertFalse(lock_blocked(p))
        finally:
            lane.close();terminate_fixture(child)

    def test_watchdog_owner_loss_retains_lane_even_when_journal_write_fails(self):
        p=self.lockfile();lane=p.open('a');owner=child=watch=None
        q,_,_=r.paths(JOB);q.mkdir(parents=True)
        try:
            fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            r.write(q/'monitor.json',r.identity.process(owner.pid))
            r.write(q/'child-ownership.json',r.identity.process(child.pid))
            code=('import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(SCRIPTS))+');'
                'import upstream_remote_weekend_override_runtime as r;'
                'import upstream_remote_weekend_override_return_queue as q;'
                'r.ROOT=Path('+repr(str(self.root))+');original=r.write\n'
                'def broken(path,value,exclusive=False):\n'
                ' if path.name=="watchdog-recovery.json":raise OSError("fixture full disk")\n'
                ' return original(path,value,exclusive)\n'
                'r.write=broken\nq.watchdog('+repr(JOB)+','+str(owner.pid)+','+str(lane.fileno())+')')
            watch=subprocess.Popen([sys.executable,'-c',code],pass_fds=(lane.fileno(),),start_new_session=True,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            wait_for(lambda:(q/'watch-ready.json').exists())
            lane.close();terminate_fixture(owner)
            wait_for(lambda:'T' in (r.identity.process(child.pid) or {}).get('state',''))
            self.assertIsNone(watch.poll());self.assertTrue(lock_blocked(p))
            self.assertFalse((q/'watchdog-recovery.json').exists())
            terminate_fixture(child);watch.wait(timeout=6)
            self.assertEqual(watch.returncode,0);self.assertFalse(lock_blocked(p))
        finally:
            lane.close()
            for proc in (owner,child,watch):
                terminate_fixture(proc)
                if proc and proc.stderr:proc.stderr.close()

    def test_idle_handoff_rejects_all_scientific_and_partial_namespaces(self):
        old=handoff.old_report(JOB);old.mkdir(parents=True)
        owner=dict(pid=123,command=str(self.root/'scripts/upstream_remote_weekend_return_queue.py')+' --job '+JOB)
        r.write(old/'plan.json',dict(job=JOB,source_sha256={}))
        r.write(old/'status.json',dict(state='waiting_full_producer_and_independent_proof'))
        with patch.object(handoff,'alive',return_value=True):
            handoff.assert_idle(JOB,owner)
            for name in ('first-scientific-baseline.json','child-ownership.json','transport-ownership.json',
                         'transport.log','pinned-remote.json','completion.json'):
                p=old/name;p.write_text('{}')
                with self.subTest(name=name),self.assertRaises(ValueError):handoff.assert_idle(JOB,owner)
                p.unlink()
            for p in r.paths(JOB)[1:]:
                p.mkdir(parents=True)
                with self.assertRaises(ValueError):handoff.assert_idle(JOB,owner)
                p.rmdir()
            r.write(old/'status.json',dict(state='transport'))
            with self.assertRaises(ValueError):handoff.assert_idle(JOB,owner)

    def test_no_qualifying_score_in_producer_prerequisite(self):
        receipt,state,proof,pstate=queue.PREREQUISITE
        a,b=self.root/'experiments'/receipt,self.root/'experiments'/proof
        a.parent.mkdir(parents=True);b.parent.mkdir(parents=True)
        r.write(b,dict(state=pstate,groups={'source0':{'rmse':999}}))
        r.write(a,dict(state=state,verification_sha256=r.ops.sha(b)))
        self.assertEqual(len(queue.prerequisite()),2)
        r.write(b,dict(state=pstate,groups={'source0':{'rmse':.1}}))
        with self.assertRaises(ValueError):queue.prerequisite()
        r.write(a,dict(state=state,verification_sha256=r.ops.sha(b)))
        self.assertEqual(len(queue.prerequisite()),2)

    def test_worker_gate_live_owner_watchdog_and_immutable_baseline(self):
        p=self.lockfile();q,_,_=r.paths(JOB);q.mkdir(parents=True)
        with p.open('a') as lane,ExitStack() as stack:
            fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lane.fileno(),True)
            command=r.worker_command(JOB,'transport',lane.fileno());pid=os.getpid()
            captured=dict(pid=pid,pgid=pid,started='self',command='bootstrap',worker_command=' '.join(command))
            actual=dict(captured,command=' '.join(command));owner=dict(pid=1011,started='owner',command='owner')
            watch=dict(pid=1012,started='watch',command='watch')
            current=dict(swap_used_bytes=1,compressor_bytes=0)
            r.write(r.PROTOCOL,{})
            r.write(q/'plan.json',dict(job=JOB,protocol_sha256=r.ops.sha(r.PROTOCOL),source_sha256={}))
            r.write(q/'monitor.json',owner);r.write(q/'transport-ownership.json',captured)
            r.write(q/'first-scientific-baseline.json',current);r.write(q/'prerequisites.json',dict(source_sha256={}))
            gate=dict(pid=pid,phase='transport',command=command,protocol_sha256=r.ops.sha(r.PROTOCOL),
                queue_plan_sha256=r.ops.sha(q/'plan.json'),owner_sha256=r.ops.sha(q/'monitor.json'),
                baseline_sha256=r.ops.sha(q/'first-scientific-baseline.json'),watchdog=watch,predecessor_sha256={})
            r.write(q/'transport-gate.json',gate)
            processes={pid:actual,1011:owner,1012:watch}
            stack.enter_context(patch.object(r,'check_protocol',return_value={}))
            stack.enter_context(patch.object(r.identity,'process',side_effect=lambda pid:processes.get(pid)))
            stack.enter_context(patch.object(r.resources,'footprints',return_value={str(pid):dict(bytes=1024,rss_bytes=1024)}))
            stack.enter_context(patch.object(r.memory,'memory',return_value=current))
            stack.enter_context(patch.object(r.shutil,'disk_usage',return_value=type('Disk',(),{'free':r.FLOOR+r.META})()))
            stack.enter_context(patch.object(r,'output_remaining',return_value=r.META))
            stack.enter_context(patch.dict(os.environ,r.ENV))
            stop=stack.enter_context(patch.object(r,'stop',return_value=False))
            guard=r.Guard(JOB,'transport',lane.fileno());stop.assert_not_called()
            with patch.object(r,'output_remaining',side_effect=OSError('fixture stat unavailable')):
                with self.assertRaises(RuntimeError):guard.check()
                stop.assert_called_with(captured)
            with patch.object(r,'output_remaining',side_effect=[OSError('temporary stat unavailable'),r.META]),\
                 patch.object(r,'stop',return_value=True) as preserved:
                baseline_object=guard.baseline;guard.check()
                preserved.assert_called_once_with(captured)
                self.assertIs(guard.baseline,baseline_object)
            for missing in (1011,1012):
                saved=processes.pop(missing);guard.last=0
                with self.subTest(missing=missing),self.assertRaises(RuntimeError):guard.check()
                processes[missing]=saved
            bad=dict(gate,pid=pid+1);r.write(q/'transport-gate.json',bad)
            with self.assertRaises(ValueError):r.Guard(JOB,'transport',lane.fileno())
            r.write(q/'transport-gate.json',gate)
            r.write(q/'first-scientific-baseline.json',dict(swap_used_bytes=2,compressor_bytes=0));guard.last=0
            with self.assertRaises(ValueError):guard.check()

    def test_idle_retirement_failed_recheck_restores_only_its_own_control_pause(self):
        old=handoff.old_report(JOB);old.mkdir(parents=True)
        owner=dict(pid=123,started='owner',command='idle-original')
        r.write(old/'monitor.json',owner);r.write(old/'plan.json',{})
        watches=[dict(pid=124,started='watch',command='watch')]
        with ExitStack() as stack:
            stack.enter_context(patch.object(r,'check_protocol',return_value={}))
            stack.enter_context(patch.object(handoff,'assert_idle',side_effect=[None,ValueError('fixture racing state')]))
            stack.enter_context(patch.object(handoff,'watchdogs',return_value=watches))
            stack.enter_context(patch.object(handoff,'only_watchdog_child'))
            stack.enter_context(patch.object(handoff,'alive',return_value=True))
            stack.enter_context(patch.object(r.identity,'process',side_effect=[dict(owner,state='Ss'),dict(owner,state='Ts'),dict(owner,state='Ts')]))
            r.write(r.PROTOCOL,{})
            kill=stack.enter_context(patch.object(handoff.os,'kill'))
            with self.assertRaisesRegex(ValueError,'racing state'):handoff.retire(JOB)
            self.assertEqual([call.args for call in kill.call_args_list],[(123,signal.SIGSTOP),(123,signal.SIGCONT)])
            self.assertFalse((handoff.destination(JOB)/'verification.json').exists())

    def worker_fixture(self,stack):
        """Exercise wrapper orchestration only; this fixture is not scientific admission."""
        q,report,art=r.paths(JOB);q.mkdir(parents=True)
        r.write(r.PROTOCOL,{})
        for name in ('plan.json','monitor.json','prerequisites.json','first-scientific-baseline.json'):
            r.write(q/name,{})
        r.write(q/'pinned-remote.json',dict(commit='a'*40))
        for module in (r,worker):
            p=self.root/'scripts'/Path(module.__file__).name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(Path(module.__file__).read_bytes());stack.enter_context(patch.object(module,'__file__',str(p)))
        guard=SimpleNamespace(check=lambda **kwargs:None)
        stack.enter_context(patch.object(r,'Guard',return_value=guard))
        spec=dict(name=JOB,published_columns=['fixture_pair'])
        def transport(job,revision,dest,arrays,check):
            r.durable_directory(dest);r.durable_directory(arrays)
            r.write(dest/'archive-index.json',dict(fixture=True))
            r.write(dest/'plan.json',dict(metadata_sha256={'archive-index.json':'fixture'}))
            r.write(dest/'transport-verification.json',dict(fixture=True))
        base=dict(state='complete_independent_weekend_full_source_queries_projection_disjointness_and_all_compact_bits',
            all_pairs=['fixture_pair'],qualifying_labels_read=False,numerical_public_features_read=False,
            score_claimed=False,source_sha256={})
        def verify(job,dest,arrays,check):r.write(dest/'independent-verification.json',base,exclusive=True)
        ctx=SimpleNamespace(bindings={},require_revision=lambda revision:None,
            transport=SimpleNamespace(job_spec=lambda job:spec,transport=transport),
            verifier=SimpleNamespace(verify=verify))
        stack.enter_context(patch.object(context_module,'Context',return_value=ctx))
        return q,report,base

    def test_worker_preserves_pristine_validator_result_and_binds_new_sidecar(self):
        with ExitStack() as stack:
            q,report,base=self.worker_fixture(stack)
            worker.run(JOB,'transport',123);worker.run(JOB,'independent_admission',123)
            pristine=report/'original-scientific-validator/independent-verification.json'
            self.assertEqual(r.read(pristine),base)
            proof=r.read(report/'independent-verification.json')
            self.assertEqual(proof['all_pairs'],base['all_pairs'])
            self.assertFalse(proof['original_remote_resource_guarantees'])
            self.assertTrue(proof['coordinator_local_guards_unchanged'])
            self.assertEqual(proof['original_scientific_validator_result_sha256'],r.ops.sha(pristine))
            for p in (pristine,report/'variant-transport-context.json',r.PROTOCOL,q/'pinned-remote.json'):
                self.assertEqual(proof['source_sha256'][str(p.relative_to(self.root))],r.ops.sha(p))
            r.check_bindings(proof['source_sha256'])

    def test_worker_rejects_changed_preceding_context_before_scientific_stage(self):
        with ExitStack() as stack:
            _,report,_=self.worker_fixture(stack)
            worker.run(JOB,'transport',123)
            path=report/'variant-transport-context.json';prior=r.read(path);prior['remote_commit']='b'*40;r.write(path,prior)
            with self.assertRaises(ValueError):worker.run(JOB,'independent_admission',123)
            self.assertFalse((report/'original-scientific-validator').exists())

    def test_worker_rejects_partial_validator_result_without_final_proof(self):
        with ExitStack() as stack:
            _,report,base=self.worker_fixture(stack)
            worker.run(JOB,'transport',123);base['all_pairs']=[]
            with self.assertRaises(ValueError):worker.run(JOB,'independent_admission',123)
            self.assertFalse((report/'independent-verification.json').exists())


if __name__=='__main__':unittest.main()
