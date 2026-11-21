"""Owned execution boundary tests; no actual production admission or MPS work."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_factored_rbm_execution_metal as execution
import upstream_factored_rbm_worker_metal as worker
import test_upstream_factored_rbm_training as fixtures


class ExecutionTests(unittest.TestCase):
    def test_fresh_worker_import_has_no_torch_or_production_side_effect(self):
        code="import sys; import upstream_factored_rbm_worker_metal as w; assert 'torch' not in sys.modules; print(w.execution.PHASES)"
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent))
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True)
        self.assertIn('mps_preflight',result.stdout)

    def test_original_resource_boundaries_and_stricter_readmission(self):
        g=2**30;workers={'1':dict(bytes=2*g,rss_bytes=3*g)};baseline=dict(swap_used_bytes=g)
        current=dict(compressor_bytes=4*g,swap_used_bytes=g+128*2**20)
        free=execution.FLOOR+execution.OUTPUT_RESERVE
        self.assertTrue(execution.ready(workers,current,baseline,free,8*g,execution.OUTPUT_RESERVE))
        for changed in (dict(current,compressor_bytes=5*g),dict(current,swap_used_bytes=g+128*2**20+1)):
            self.assertFalse(execution.safe(workers,changed,baseline,free))
        self.assertFalse(execution.safe(workers,current,baseline,free-1))
        self.assertFalse(execution.safe(workers,current,baseline,free,reserve=21*g))
        self.assertTrue(execution.safe(workers,dict(current,compressor_bytes=4.6*g),baseline,free))
        self.assertFalse(execution.ready(workers,dict(current,compressor_bytes=4.6*g),baseline,free,0,execution.OUTPUT_RESERVE))

    def test_process_identity_rejects_recycled_pid_or_foreign_group(self):
        captured=dict(pid=12,started='original',pgid=12,command='bootstrap',worker_command='worker')
        self.assertTrue(execution.same_child(captured,dict(captured,command='worker')))
        for changed in (dict(captured,started='recycled'),dict(captured,pgid=11),dict(captured,command='foreign')):
            self.assertFalse(execution.same_child(captured,changed))
        self.assertFalse(execution.same_child(captured,None))

    def test_continuation_reserves_physical_and_rss_independently(self):
        g=2**30;baseline={'swap_used_bytes':0};current={'compressor_bytes':0,'swap_used_bytes':0}
        workers={'12':dict(bytes=2*g,rss_bytes=6*g),'13':dict(bytes=17*g,rss_bytes=1*g)}
        self.assertFalse(execution.continuation_ready(workers,12,'fit_known',current,baseline,100*g,64*2**20))
        workers['13']=dict(bytes=g,rss_bytes=17*g)
        self.assertFalse(execution.continuation_ready(workers,12,'fit_known',current,baseline,100*g,64*2**20))
        workers['13']=dict(bytes=g,rss_bytes=g)
        self.assertTrue(execution.continuation_ready(workers,12,'fit_known',current,baseline,100*g,64*2**20))

    def test_owner_loss_keeps_lane_descriptors_until_stopped_child_terminal_when_journaling_fails(self):
        # Real independent processes and flock descriptors. Only this test's
        # dedicated sleeper is signaled; no production process is touched.
        import fcntl
        import signal
        import time
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder).resolve();owner=dict(pid=2147483647,started='missing',command='missing')
            (report/'monitor.json').write_text(json.dumps(owner))
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            watch=None
            primary=(report/'primary.lock').open('a');gpu=(report/'gpu.lock').open('a')
            try:
                fcntl.flock(primary,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
                captured=execution.identity.process(child.pid);(report/'child-ownership.json').write_text(json.dumps(captured))
                code="import sys; from pathlib import Path; import upstream_factored_rbm_execution_metal as e; e.REPORT=Path(sys.argv[1]); e.exclusive_json=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('disk floor')); e.watchdog(2147483647,[int(x) for x in sys.argv[2:]])"
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent))
                watch=subprocess.Popen([sys.executable,'-c',code,str(report),str(primary.fileno()),str(gpu.fileno())],env=env,pass_fds=(primary.fileno(),gpu.fileno()))
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    actual=execution.identity.process(child.pid)
                    if actual and 'T' in actual['state']:break
                    time.sleep(.05)
                self.assertIn('T',execution.identity.process(child.pid)['state']);self.assertIsNone(watch.poll())
                primary.close();gpu.close()
                for name in ('primary.lock','gpu.lock'):
                    with (report/name).open('a') as contender:
                        with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);watch.wait(timeout=5)
                self.assertEqual(watch.returncode,0)
                for name in ('primary.lock','gpu.lock'):
                    with (report/name).open('a') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                if watch is not None and watch.poll() is None:watch.terminate();watch.wait(timeout=5)
                primary.close();gpu.close()

    def test_cpu_or_partial_preflight_cannot_authorize_fitting(self):
        from upstream_factored_rbm_preflight import REQUIRED_STRESS
        value=dict(state=execution.RESULTS['mps_preflight'][1],execution_plan_sha256='a'*64,qualifying_labels_read=False,
            device='mps',actual_MPS_tested=True,passed_for_production_device=True,full_required_stress_tested=True,
            fresh_mps_runtime_import=True,cases={'stress':{'shape':REQUIRED_STRESS}})
        execution.validate_result('mps_preflight',value,'a'*64)
        for key,change in [('device','cpu'),('fresh_mps_runtime_import',False),('full_required_stress_tested',False),('qualifying_labels_read',True)]:
            with self.assertRaises(ValueError):execution.validate_result('mps_preflight',dict(value,**{key:change}),'a'*64)

    def test_self_pause_preserves_guard_and_original_baseline_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder);baseline_path=report/'first-scientific-baseline.json';baseline_path.write_text('{}')
            guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
            guard.phase='data';guard.limit=3*2**30;guard.last=0.;guard.counter=0;guard.plan={};guard.baseline={}
            guard.baseline_sha=execution.inputs.sha(baseline_path);guard.predecessors={};guard.captured={'pid':777}
            workers={'777':dict(bytes=1,rss_bytes=1)};current={'compressor_bytes':0,'swap_used_bytes':0}
            with patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'),patch.object(execution.os,'getpid',return_value=777),patch.object(execution.resources,'footprints',return_value=workers),patch.object(execution.memory,'memory',return_value=current),patch.object(execution,'safe',side_effect=[False,True]),patch.object(execution,'stop_child',return_value=True) as stop:
                guard(extra_disk_bytes=123)
            stop.assert_called_once_with(guard.captured);self.assertEqual(guard.counter,1)
            self.assertEqual(guard.baseline_sha,execution.inputs.sha(baseline_path))
            self.assertEqual(json.loads((report/'data-worker-resource-request.json').read_text())['extra_disk_reserve'],execution.OUTPUT_RESERVE)

    def test_measurement_failures_preserve_the_same_worker_stack(self):
        failures=[subprocess.CalledProcessError(1,['vmmap']),ValueError('malformed memory number'),AttributeError('missing memory field'),OSError('disk query')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as folder:
                report=Path(folder);path=report/'first-scientific-baseline.json';path.write_text('{}')
                guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
                guard.phase='data';guard.limit=3*2**30;guard.last=0.;guard.counter=0;guard.plan={}
                guard.baseline={'swap_used_bytes':0};guard.baseline_sha=execution.inputs.sha(path)
                guard.predecessors={};guard.captured={'pid':777}
                workers={'777':dict(bytes=1,rss_bytes=1)};current={'compressor_bytes':0,'swap_used_bytes':0}
                real_disk_usage=execution.shutil.disk_usage;calls=[0]
                def measured_disk(root):
                    calls[0]+=1
                    if isinstance(failure,OSError) and calls[0]==1:raise failure
                    return real_disk_usage(root)
                samples=[workers,workers] if isinstance(failure,OSError) else [failure,workers]
                with patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'),patch.object(execution.os,'getpid',return_value=777),patch.object(execution.resources,'footprints',side_effect=samples),patch.object(execution.memory,'memory',return_value=current),patch.object(execution.shutil,'disk_usage',side_effect=measured_disk),patch.object(execution,'stop_child',return_value=True) as stop:
                    guard()
                self.assertEqual(guard.counter,1);stop.assert_called_once_with(guard.captured)


class WorkerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def test_full_synthetic15_epoch_saved_model_replay_and_tamper_rejection(self):
        helper=fixtures.TrainingTests();helper.torch=self.torch;reader,initial=helper.setup_state()
        configuration=asdict(initial.config);configuration['epochs']=15
        plan={'configuration':configuration};sha='a'*64;data_sha='b'*64
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();study=root/'study';art=root/'art';report=root/'report'
            for directory in (study,art,report):directory.mkdir()
            (report/'first-scientific-baseline.json').write_text('{}')
            (report/'fit_known-ownership.json').write_text(json.dumps({'pid':os.getpid()}))
            (report/'fit_qualifying-ownership.json').write_text(json.dumps({'pid':os.getpid()}))
            (report/'plan.json').write_text(json.dumps(plan));(study/'mps-preflight.json').write_text('{}')
            data={};declaration={'known_rows':len(reader.qr['user_idxs']),'phases':{'qualifying':{'query_rows':len(reader.qr['user_idxs'])}}}
            @contextmanager
            def synthetic_reader(phase,guard):yield reader,data,declaration,data_sha
            real_state=worker.State;real_replay=worker.terminal.replay
            def cpu_state(r,digest,config,device):return real_state(r,digest,config,'cpu')
            def cpu_replay(record,r,phase,digest,config,guard,device):return real_replay(record,r,phase,digest,config,guard,'cpu')
            def json_output(path,value,guard):
                with Path(path).open('x') as stream:json.dump(value,stream)
            def payload(path,writer,root,guard):return execution.durable_payload(path,writer,root,guard,floor_bytes=0,reserve_bytes=0)
            with patch.object(execution,'ROOT',root),patch.object(execution,'REPORT',report),patch.object(execution,'STUDY',study),patch.object(execution,'ART',art),patch.object(execution,'exclusive_json',side_effect=json_output),patch.object(worker,'reader_for',synthetic_reader),patch.object(worker,'torch_runtime',return_value=self.torch),patch.object(worker,'State',side_effect=cpu_state),patch.object(worker,'project_known',side_effect=lambda v,*args:v.copy()),patch.object(worker.terminal,'durable_payload',side_effect=payload),patch.object(worker.terminal,'replay',side_effect=cpu_replay),patch.object(worker,'Config',return_value=fixtures.Config(**configuration)):
                # Config() in replay is only the fixed production epoch-count
                # assertion; verify_history requires Config(**configuration).
                with patch.object(worker,'Config',side_effect=lambda **kwargs:fixtures.Config(**(kwargs or configuration))):
                    receipt=worker.fit('known',plan,lambda **kwargs:None,sha)
                    self.assertEqual(receipt['epochs'],15);self.assertEqual(len(receipt['epoch_sha256']),15)
                    (study/'known-fit.json').write_text(json.dumps(receipt))
                    proof=worker.replay('known',plan,lambda **kwargs:None,sha)
                    self.assertTrue(proof['all_native_prediction_bits_match']);self.assertTrue(proof['known_projection_bits_match'])
                    (study/'known-replay.json').write_text(json.dumps(proof))
                    qualifying=worker.fit('qualifying',plan,lambda **kwargs:None,sha)
                    (study/'qualifying-fit.json').write_text(json.dumps(qualifying))
                    qproof=worker.replay('qualifying',plan,lambda **kwargs:None,sha)
                    (study/'qualifying-replay.json').write_text(json.dumps(qproof))
                    with patch.object(worker,'data_receipt',return_value=(data,declaration,data_sha)):
                        pair=worker.assemble(plan,lambda **kwargs:None,sha)
                    self.assertEqual(pair['models_fitted'],2);self.assertEqual(pair['epochs_each'],15)
                    self.assertFalse(pair['score_computed']);self.assertFalse(pair['qualifying_labels_read'])
                    values=np.load(art/'known-native.npy',allow_pickle=False);values[0]+=np.float32(.01)
                    with (art/'known-native.npy').open('wb') as stream:np.save(stream,values,allow_pickle=False)
                    with self.assertRaises(ValueError):worker.replay('known',plan,lambda **kwargs:None,sha)
                    # Even a newly authenticated modified vector must fail the
                    # numerical replay, rather than merely the payload hash.
                    receipt['native_predictions']['sha256']=execution.inputs.sha(art/'known-native.npy')
                    (study/'known-fit.json').write_text(json.dumps(receipt))
                    with self.assertRaisesRegex(ValueError,'every original f32 bit'):
                        worker.replay('known',plan,lambda **kwargs:None,sha)
                    with (art/'known-native.npy').open('wb') as stream:np.save(stream,np.array([1],dtype=np.float32),allow_pickle=False)
                    with self.assertRaises(ValueError):worker.replay('known',plan,lambda **kwargs:None,sha)


if __name__=='__main__':unittest.main()
