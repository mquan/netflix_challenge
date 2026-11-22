"""Original policy mocks plus actual group stop/continue and lane-FD lifetime."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch,Mock
import upstream_full_context_catboost_supervision as module


class SupervisionTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary).resolve();baseline=repo/'baseline.json';baseline.write_bytes(module.study.BASELINE.read_bytes())
            owner=module.family.identity.process(os.getpid())
            root=dict(pid=10,ppid=owner['pid'],pgid=10,state='T',started='original-root',command='python upstream_root.py')
            role=dict(pid=11,ppid=10,pgid=10,state='T',started='original-role',command='python upstream_source.py')
            c=module.Control(root,lambda:{'producer':role},repo=repo,baseline_path=baseline,owner=owner,watchdog=owner,
                check_sources=lambda:None,remaining_output=lambda:module.family.RESERVE,
                emergency_reserve=2*module.family.GiB,request_path=repo/'request.json')
            yield c,role

    def test_three_consecutive_full_reservations_reset_after_any_failed_sample(self):
        with self.fixture() as (c,role):
            steps=[True,False,True,True,True];slept=[]
            original=module.family.identity.process
            with patch.object(c,'sample',side_effect=[(v,{'sample':i}) for i,v in enumerate(steps)]), \
                    patch.object(module.family.identity,'process',side_effect=lambda pid:c.root if pid==10 else original(pid)), \
                    patch.object(module.time,'sleep',side_effect=slept.append),patch.object(module.os,'killpg') as resume:
                value=c.wait_then_resume()
            self.assertEqual(value,{'sample':4});self.assertEqual(slept,[10]*4)
            resume.assert_called_once_with(10,signal.SIGCONT)

    def test_original_baseline_and_remaining_output_fail_closed(self):
        with self.fixture() as (c,role):
            c.remaining_output=lambda:module.family.RESERVE-1
            with self.assertRaisesRegex(ValueError,'remaining-output'):c.reserve()
            c.baseline_path.write_text('{"swap_used_bytes":0,"compressor_bytes":0}')
            with self.assertRaises(ValueError):c.check()

    def test_owner_loss_or_root_loss_never_resumes_or_releases_a_surviving_role(self):
        with self.fixture() as (c,role):
            original=module.family.identity.process
            with patch.object(module.family.identity,'process',side_effect=lambda pid:None if pid==10 else original(pid)),patch.object(module.os,'killpg') as resume:
                with self.assertRaisesRegex(RuntimeError,'root loss'):c.wait_then_resume()
                resume.assert_not_called()
            with patch.object(module.family.identity,'process',return_value=None),patch.object(module.os,'killpg') as resume:
                with self.assertRaisesRegex(RuntimeError,'owner and watchdog'):c.wait_then_resume()
                resume.assert_not_called()

    def test_watchdog_uncertain_or_absent_pointer_retains_previous_live_family(self):
        with self.fixture() as (c,role):
            for second in [None,ValueError('metadata uncertain')]:
                with self.subTest(second=str(second)):
                    provider=Mock(side_effect=[(c.root,{'producer':role}),second])
                    release=Mock(side_effect=[False,True])
                    logs=[]
                    with patch.object(module.family,'live_roles',return_value={'root':c.root,'producer':role}), \
                            patch.object(module.family.identity,'process',return_value=c.owner), \
                            patch.object(module.family,'stop_family',return_value=True) as stop, \
                            patch.object(module.time,'sleep',side_effect=[None,KeyboardInterrupt()]), \
                            patch.object(module.os,'close') as close:
                        def record(value):
                            close.assert_not_called();logs.append(value)
                        with self.assertRaises(KeyboardInterrupt):module.watch(c.owner,provider,release,[123],record=record)
                        stop.assert_called_once_with(c.root,{'producer':role});close.assert_called_once_with(123)
                    self.assertTrue(logs)

    def wait_for(self,predicate,timeout=15):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            value=predicate()
            if value:return value
            time.sleep(.05)
        self.fail('Actual private helper did not reach its expected state')

    def test_actual_same_PID_stack_RNG_and_objects_survive_full_group_stop_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary).resolve();baseline=repo/'baseline.json';baseline.write_bytes(module.study.BASELINE.read_bytes())
            owner=module.family.identity.process(os.getpid());(repo/'owner.json').write_text(json.dumps(owner))
            code=r'''import sys,json,os,random
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import upstream_full_context_catboost_supervision as m
repo=Path(sys.argv[2]);owner=json.loads((repo/'owner.json').read_text());root=m.family.identity.process(os.getpid())
(repo/'root.json').write_text(json.dumps(root))
c=m.Control(root,lambda:{},repo=repo,baseline_path=repo/'baseline.json',owner=owner,watchdog=owner,
    check_sources=lambda:None,remaining_output=lambda:m.family.RESERVE,emergency_reserve=m.family.RESERVE,request_path=repo/'request.json')
# Resource samples are synthetic; the SIGSTOP/CONT and object/RNG frame are real.
m.family.measured=lambda root,roles:({str(root['pid']):dict(bytes=1048576,rss_bytes=1048576)},{root['pid']})
count=[0]
def measured_memory():
    count[0]+=1
    return dict(compressor_bytes=5*m.family.GiB if count[0]==1 else 0,swap_used_bytes=c.baseline['swap_used_bytes'])
m.memory.memory=measured_memory
rng=random.Random(381);objects={'batch':17,'optimizer':[.1,.2]};before=id(objects);first=rng.random()
g=m.Guard(c);g();second=rng.random()
reference=random.Random(381)
assert [first,second]==[reference.random(),reference.random()] and id(objects)==before and objects['batch']==17
assert 'torch' not in sys.modules and g.stops==1
(repo/'done.json').write_text(json.dumps(dict(pid=os.getpid(),stops=g.stops,same_objects=True,same_rng=True)))
'''
            helper=subprocess.Popen([sys.executable,'-c',code,str(Path(module.__file__).parent),str(repo)],start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                root=self.wait_for(lambda:json.loads((repo/'root.json').read_text()) if (repo/'root.json').exists() else None)
                self.wait_for(lambda:module.family.identity.process(helper.pid) and 'T' in module.family.identity.process(helper.pid)['state'])
                c=module.Control(root,lambda:{},repo=repo,baseline_path=baseline,owner=owner,watchdog=owner,
                    check_sources=lambda:None,remaining_output=lambda:module.family.RESERVE,emergency_reserve=module.family.RESERVE,request_path=repo/'owner-request.json')
                slept=[];workers={str(root['pid']):dict(bytes=1048576,rss_bytes=1048576)}
                with patch.object(module.family,'measured',return_value=(workers,{root['pid']})), \
                        patch.object(module.memory,'memory',return_value=dict(compressor_bytes=0,swap_used_bytes=c.baseline['swap_used_bytes'])), \
                        patch.object(module.time,'sleep',side_effect=slept.append):
                    c.wait_then_resume()
                output,errors=helper.communicate(timeout=15)
                self.assertEqual(helper.returncode,0,(output+errors).decode());self.assertEqual(slept,[10,10])
                done=json.loads((repo/'done.json').read_text());self.assertEqual(done['pid'],helper.pid)
                self.assertTrue(done['same_objects'] and done['same_rng']);self.assertEqual(done['stops'],1)
            finally:
                if helper.poll() is None:
                    os.killpg(helper.pid,signal.SIGCONT);helper.terminate();helper.wait(timeout=10)
                helper.stdout.close();helper.stderr.close()

    def test_actual_watchdog_keeps_inherited_lane_lock_while_preserved_root_lives(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo=Path(temporary).resolve();lane=repo/'lane.lock'
            root=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
            watcher=None
            with lane.open('a') as descriptor:
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                capture=module.family.identity.process(root.pid);assert capture
                (repo/'root.json').write_text(json.dumps(capture));owner=module.family.identity.process(os.getpid())
                (repo/'owner.json').write_text(json.dumps(owner))
                code=r'''import sys,json,os
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import upstream_full_context_catboost_supervision as m
repo=Path(sys.argv[2]);root=json.loads((repo/'root.json').read_text());owner=json.loads((repo/'owner.json').read_text())
actual=m.family.identity.process
m.family.identity.process=lambda pid:None if pid==owner['pid'] else actual(pid)
m.watch(owner,lambda:(root,{}),lambda:False,[int(sys.argv[3])],record=lambda value:(repo/'watch.json').write_text(json.dumps(value)))
'''
                watcher=subprocess.Popen([sys.executable,'-c',code,str(Path(module.__file__).parent),str(repo),str(descriptor.fileno())],pass_fds=(descriptor.fileno(),),start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                self.wait_for(lambda:module.family.identity.process(root.pid) and 'T' in module.family.identity.process(root.pid)['state'])
                with lane.open('a') as probe:
                    with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertIsNone(watcher.poll())
                # Only the private sleeping helper is terminated for cleanup.
                root.terminate();os.killpg(root.pid,signal.SIGCONT);root.wait(timeout=10)
                output,errors=watcher.communicate(timeout=15)
                self.assertEqual(watcher.returncode,0,(output+errors).decode())
                with lane.open('a') as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                if root.poll() is None:
                    os.killpg(root.pid,signal.SIGCONT);root.terminate();root.wait(timeout=10)
                if watcher is not None:
                    if watcher.poll() is None:watcher.terminate();watcher.wait(timeout=10)
                    watcher.stdout.close();watcher.stderr.close()


if __name__=='__main__':unittest.main()
