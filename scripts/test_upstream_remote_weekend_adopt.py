"""Synthetic adoption-control tests; never touch remote or production PIDs."""
from contextlib import ExitStack,contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import upstream_remote_weekend_adopt as adoption
import upstream_cf96_primary_lane as processes
import upstream_remote_weekend as original

ROOT=Path(__file__).resolve().parents[1]
COMMIT='ee38c3ec91e8160250d159c423e1581640c96700'


def blob(name):return subprocess.check_output(['git','show',COMMIT+':experiments/upstream-remote-weekend/execution/'+name],cwd=ROOT)


@contextmanager
def metadata_session():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder).resolve();base=root/'native';base.mkdir();tree=base/'source';tree.mkdir()
        name='s3-rbm400pf'
        for key in adoption.ORIGINAL_CONTROL_SHA[name]:(base/(name+'-'+key+'.json')).write_bytes(blob(name+'/'+name+'-'+key+'.json'))
        (base/'preparation.json').write_bytes(blob('bootstrap/preparation.json'))
        (base/(name+'.log')).write_text(''.join('Epoch %02d | RMSE .9\n'%i for i in range(1,6)))
        session=adoption.Session.__new__(adoption.Session);session.root=root;session.base=base;session.tree=tree;session.name=name
        session.plan=original.read(ROOT/'experiments/upstream-remote-weekend/plan.json');session.job=next(row for row in session.plan['jobs'] if row['name']==name)
        captured=adoption.read(base/(name+'-ownership.json'));actual=dict(captured,command=captured['native_command'],state='Ts')
        session.original=SimpleNamespace(PLAN=ROOT/'experiments/upstream-remote-weekend/plan.json',
            identity=SimpleNamespace(process=lambda pid:actual if pid==captured['pid'] else None,same_owner=processes.same_owner))
        yield session


class AdoptionProtocolTests(unittest.TestCase):
    def test_import_is_metadata_only_and_never_imports_numpy_or_torch(self):
        code="import sys; import upstream_remote_weekend_adopt as a; assert 'numpy' not in sys.modules and 'torch' not in sys.modules; print(len(a.TREES))"
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts')),capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'3')

    def test_non_remote_host_rejected_before_loading_any_production_session_or_signal(self):
        with patch.object(adoption.sys,'platform','darwin'),patch.object(adoption.subprocess,'check_output',return_value=str(32*adoption.GIB)),patch.object(adoption,'Session') as session,patch.object(adoption.os,'killpg') as signals:
            with self.assertRaisesRegex(ValueError,'original96GiB remote host'):adoption.run('/not-accessed','s3-rbm400pf')
            session.assert_not_called();signals.assert_not_called()

    def test_all24_original_control_expectations_match_immutable_guard_commit(self):
        for name,bindings in adoption.ORIGINAL_CONTROL_SHA.items():
            for key,expected in bindings.items():self.assertEqual(hashlib.sha256(blob(name+'/'+name+'-'+key+'.json')).hexdigest(),expected)

    def test_metadata_accepts_original_stopped_five_epoch_prefix_and_rejects_baseline_rebase(self):
        with metadata_session() as session,patch.object(adoption,'attest_runtime',return_value={'synthetic':True}):
            result=session.original_evidence()
            self.assertEqual(result['original_epoch_prefix'],list(range(1,6)))
            baseline=session.base/(session.name+'-baseline.json');value=adoption.read(baseline);value['swap_used_bytes']+=1
            baseline.write_text(json.dumps(value))
            launch=session.base/(session.name+'-launch.json');value=adoption.read(launch);value['original_swap_baseline']['swap_used_bytes']+=1;launch.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'pre-recovery execution expectation'):session.original_evidence()

    def test_wrong_native_start_and_prior_live_owner_prevent_adoption(self):
        for change in ('start','owner'):
            with self.subTest(change=change),metadata_session() as session,patch.object(adoption,'attest_runtime',return_value={'synthetic':True}):
                captured=adoption.read(session.base/(session.name+'-ownership.json'));owner=adoption.read(session.base/(session.name+'-owner.json'))
                actual=dict(captured,command=captured['native_command'],state='Ts')
                if change=='start':actual['started']='recycled PID'
                session.original.identity.process=lambda pid:actual if pid==captured['pid'] else (owner if change=='owner' and pid==owner['pid'] else None)
                with self.assertRaises(ValueError):session.original_evidence()

    def test_original_resource_boundaries_credit_physical_and_rss_independently(self):
        g=adoption.GIB;workers={'1':dict(bytes=4*g,rss_bytes=60*g),'2':dict(bytes=g/10,rss_bytes=g/10),'3':dict(bytes=15*g,rss_bytes=g)}
        baseline={'swap_used_bytes':0};current={'compressor_bytes':9*g,'swap_used_bytes':128*2**20}
        self.assertFalse(adoption.capacity(workers,current,baseline,1,2,True))
        workers['3']=dict(bytes=g,rss_bytes=15*g)
        self.assertFalse(adoption.capacity(workers,current,baseline,1,2,True))
        workers['3']=dict(bytes=g,rss_bytes=g)
        self.assertTrue(adoption.capacity(workers,current,baseline,1,2,True))
        self.assertFalse(adoption.capacity(workers,dict(current,swap_used_bytes=128*2**20+1),baseline,1,2,True))
        self.assertTrue(adoption.capacity(workers,dict(current,compressor_bytes=11*g),baseline,1,2))
        self.assertFalse(adoption.capacity(workers,dict(current,compressor_bytes=11*g),baseline,1,2,True))
        self.assertFalse(adoption.capacity(workers,dict(current,compressor_bytes=12*g),baseline,1,2))

    def test_declared_full_native_output_populations_are_unchanged(self):
        plan=original.read(ROOT/'experiments/upstream-remote-weekend/plan.json')
        for job in plan['jobs']:
            expected=adoption.shapes(job)
            self.assertEqual(len(expected),2*len(job['published_columns'])+2+(2 if job['kind']=='rbm' else 0))
            for column in job['published_columns']:
                self.assertEqual(expected['preds_3/'+column+'.probe_3.npy'],(1760494,))
                self.assertEqual(expected['preds_3/'+column+'.qual.npy'],(2817131,))

    def test400_first_requires_confirmed_running_not_a_pre_signal_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();first=root/'experiments/upstream-remote-weekend/recovery-v1/s3-rbm400pf';first.mkdir(parents=True)
            native={'pid':7,'pgid':7,'started':'synthetic','command':'native','native_command':'native','state':'Ts'}
            (first/'plan.json').write_text(json.dumps({'original_native':native}));(first/'initial-continuation.json').write_text('{}')
            with self.assertRaises(ValueError):adoption.require400_first(root,'s3-rbm800p')
            observation={'actual_running_state_observed':True,'observed_native':native,
                'execution_plan_sha256':adoption.sha(first/'plan.json'),'continuation_intent_sha256':adoption.sha(first/'initial-continuation.json')}
            (first/'initial-running-observation.json').write_text(json.dumps(observation))
            with self.assertRaises(ValueError):adoption.require400_first(root,'s3-rbm800p')
            observation['observed_native']=dict(native,state='R');(first/'initial-running-observation.json').write_text(json.dumps(observation))
            adoption.require400_first(root,'s3-rbm800p')

    def test_initial_resume_signals_only_owned_dummy_then_resource_stop_never_autocontinues(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'plan.json').write_text('{}');(root/'dummy-baseline.json').write_text('{}')
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            try:
                captured=processes.process(child.pid);captured['native_command']=captured['command'];os.killpg(child.pid,signal.SIGSTOP)
                for _ in range(100):
                    if 'T' in processes.process(child.pid)['state']:break
                    time.sleep(.01)
                s=adoption.Session.__new__(adoption.Session);s.name='dummy';s.base=root;s.report=root;s.tree=root;s.captured=captured
                s.job={}
                s.owner=processes.process(os.getpid());s.watch_identity=s.owner;s.original=SimpleNamespace(identity=processes)
                s.inputs_verified=True;s.pause_number=0;s.admission=lambda **kwargs:None;s.check=lambda:None
                with patch.object(adoption,'attest_runtime',return_value={}):s.continue_once(0)
                for _ in range(100):
                    if 'T' not in processes.process(child.pid)['state']:break
                    time.sleep(.01)
                self.assertNotIn('T',processes.process(child.pid)['state'])
                self.assertTrue((root/'initial-running-observation.json').exists())
                checks=[0]
                class StopObservation(Exception):pass
                def check():
                    checks[0]+=1
                    if checks[0]>1:raise StopObservation('test ends with actual RAM still paused')
                s.check=check;s.measured=lambda **kwargs:(False,{'synthetic_pressure':True})
                real_killpg=os.killpg
                with patch.object(adoption.os,'killpg',wraps=real_killpg) as signals:
                    with self.assertRaises(StopObservation):s.monitor()
                self.assertEqual([call.args for call in signals.call_args_list],[(child.pid,signal.SIGSTOP)])
                self.assertIn('T',processes.process(child.pid)['state']);self.assertEqual(s.pause_number,1)
                self.assertTrue((root/'pause-001.json').exists());self.assertFalse((root/'continuation-001.json').exists())
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

    def test_failed_signal_keeps_intent_but_never_claims_native_running(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'plan.json').write_text('{}');(root/'dummy-baseline.json').write_text('{}')
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            try:
                captured=processes.process(child.pid);captured['native_command']=captured['command'];os.killpg(child.pid,signal.SIGSTOP)
                for _ in range(100):
                    if 'T' in processes.process(child.pid)['state']:break
                    time.sleep(.01)
                s=adoption.Session.__new__(adoption.Session);s.name='dummy';s.base=root;s.report=root;s.tree=root;s.captured=captured;s.job={}
                s.owner=processes.process(os.getpid());s.watch_identity=s.owner;s.original=SimpleNamespace(identity=processes)
                s.inputs_verified=True;s.admission=lambda **kwargs:None;s.check=lambda:None
                with patch.object(adoption,'attest_runtime',return_value={}),patch.object(adoption.os,'killpg',side_effect=OSError('synthetic signal failure')):
                    with self.assertRaises(OSError):s.continue_once(0)
                self.assertTrue((root/'initial-continuation.json').exists());self.assertFalse((root/'initial-running-observation.json').exists())
                self.assertIn('T',processes.process(child.pid)['state'])
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

    def test_watchdog_keeps_real_operation_lock_after_owner_loss_and_journal_failure(self):
        import fcntl
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();lock=(root/'operation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True);watch=None
            try:
                captured=processes.process(child.pid);captured['native_command']=captured['command']
                (root/'plan.json').write_text(json.dumps({'original_native':captured}))
                (root/'owner.json').write_text(json.dumps({'pid':2147483647,'started':'absent','command':'absent'}))
                code="import sys; from pathlib import Path; from types import SimpleNamespace; import upstream_remote_weekend_adopt as a; import upstream_cf96_primary_lane as p; a.require_host=lambda:None; a.Session=lambda *args:SimpleNamespace(report=Path(sys.argv[1]),base=Path(sys.argv[1]),original=SimpleNamespace(identity=p)); saved=a.durable; a.durable=lambda path,value: (_ for _ in ()).throw(OSError('synthetic full disk')) if path.name=='watchdog-recovery.json' else saved(path,value); a.watchdog(None,None,int(sys.argv[2]))"
                watch=subprocess.Popen([sys.executable,'-c',code,str(root),str(lock.fileno())],env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts')),pass_fds=(lock.fileno(),))
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if 'T' in processes.process(child.pid)['state']:break
                    time.sleep(.05)
                self.assertIn('T',processes.process(child.pid)['state']);self.assertIsNone(watch.poll());lock.close()
                with (root/'operation.lock').open('a') as contender:
                    with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);watch.wait(timeout=5);self.assertEqual(watch.returncode,0)
                with (root/'operation.lock').open('a') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                if watch is not None and watch.poll() is None:watch.terminate();watch.wait(timeout=5)
                lock.close()


class SyntheticFinalizationTests(unittest.TestCase):
    @contextmanager
    def session(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();tree=root/'source';out=tree/'preds_3';report=root/'recovery';out.mkdir(parents=True);report.mkdir()
            job=next(row for row in original.read(ROOT/'experiments/upstream-remote-weekend/plan.json')['jobs'] if row['name']=='s3-rbm400pf')
            expected={name:((2,3) if '.ifeat.' in name else (3,)) for name in adoption.shapes(job)}
            for name,shape in expected.items():
                with (tree/name).open('wb') as stream:np.save(stream,np.full(shape,3.,dtype='<f4'),allow_pickle=False)
                os.utime(tree/name,(adoption.DEADLINE.timestamp()-3600,)*2)
            log=root/(job['name']+'.log');log.write_text(''.join('Epoch %02d | RMSE .9\n'%i for i in list(range(1,16))*2));os.utime(log,(adoption.DEADLINE.timestamp()-3600,)*2)
            (root/(job['name']+'-launch.json')).write_text('{"synthetic":true}')
            (report/'plan.json').write_text('{"synthetic":true}')
            session=adoption.Session.__new__(adoption.Session);session.job=job;session.name=job['name'];session.tree=tree;session.base=root;session.report=report
            session.owner={'synthetic_owner':True};session.watch_identity={'synthetic_watchdog':True};session.admission=lambda **kwargs:None
            session.check=lambda:None;session.verify_inputs=lambda:None;session.hash_guard=lambda:None
            session.original=SimpleNamespace(np=np,trace_check=original.trace_check)
            with patch.object(adoption,'shapes',return_value=expected):yield session,expected

    def test_full15_plus15_trace_all_native_files_and_truthful_unknown_exit_status(self):
        with self.session() as (s,expected):
            s.finalize();receipt=adoption.read(s.base/(s.name+'-receipt.json'))
            self.assertEqual(set(receipt['outputs']),set(expected)|{'preds_3/'+s.name+'.out'})
            self.assertIsNone(receipt['same_process_adoption']['native_exit_code'])
            self.assertFalse(receipt['same_process_adoption']['orphan_exit_status_observable'])
            code=receipt['same_process_adoption']['controller_utf8'].encode()
            self.assertEqual(hashlib.sha256(code).hexdigest(),receipt['same_process_adoption']['controller_sha256'])
            self.assertTrue((s.report/'completion.json').exists())

    def test_missing_epoch_nonfinite_values_or_changed_terminal_bits_reject_completion(self):
        for change in ('missing','nan','terminal'):
            with self.subTest(change=change),self.session() as (s,expected):
                path=s.tree/('preds_3/'+s.name+'_ep07.qual.npy')
                if change=='missing':path.unlink()
                elif change=='nan':
                    with path.open('wb') as stream:np.save(stream,np.array([3.,np.nan,3.],dtype='<f4'),allow_pickle=False)
                else:
                    path=s.tree/('preds_3/'+s.name+'.qual.npy')
                    with path.open('wb') as stream:np.save(stream,np.array([3.,3.,3.1],dtype='<f4'),allow_pickle=False)
                os.utime(path if path.exists() else s.base,(adoption.DEADLINE.timestamp()-3600,)*2)
                with self.assertRaises(ValueError):s.finalize()
                self.assertFalse((s.base/(s.name+'-receipt.json')).exists())


if __name__=='__main__':unittest.main()
