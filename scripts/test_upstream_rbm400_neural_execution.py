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
import upstream_rbm400_neural_execution as execution
import upstream_rbm400_neural_worker as worker
import test_upstream_factored_rbm_training as fixtures


class ExecutionTests(unittest.TestCase):
    def test_output_stat_failure_preserves_same_frame_and_original_baseline(self):
        import torch
        g=2**30
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder);path=report/'first-scientific-baseline.json';path.write_text('{}')
            guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
            guard.phase='fit_latent';guard.limit=16*g;guard.last=0.;guard.counter=0;guard.plan={}
            guard.baseline={'swap_used_bytes':0};guard.baseline_sha=execution.inputs.sha(path)
            guard.predecessors={};guard.captured={'pid':777}
            workers={'777':dict(bytes=2*g,rss_bytes=2*g)};current={'compressor_bytes':0,'swap_used_bytes':0}
            with patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'),patch.object(execution.os,'getpid',return_value=777),\
                 patch.object(execution,'output_remaining',side_effect=[OSError('output stat transient'),64*2**20]),\
                 patch.object(execution.resources,'footprints',return_value=workers),patch.object(execution.memory,'memory',return_value=current),\
                 patch.object(execution,'stop_child',return_value=True) as stop,\
                 patch.object(torch.mps,'driver_allocated_memory',return_value=2*g),patch.object(torch.mps,'current_allocated_memory',return_value=2*g):
                guard()
            stop.assert_called_once_with(guard.captured);self.assertEqual(guard.counter,1)
            record=json.loads((report/'fit_latent-worker-resource-request.json').read_text())
            self.assertIn('output stat transient',record['sample']['disk_accounting_error'])
            self.assertGreaterEqual(record['extra_disk_reserve'],320*2**20)
            self.assertEqual(guard.baseline_sha,execution.inputs.sha(path))

    def test_review_gate_binds_full_protocol_source_before_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source=root/'fixture.py';source.write_text('# synthetic\n')
            proof_path=root/'experiments/upstream-rbm400-neural-execution-protocol/verification.json';proof_path.parent.mkdir(parents=True)
            baseline=root/'experiments/upstream-factored-rbm-execution-deterministic-metal/first-scientific-baseline.json'
            baseline.parent.mkdir();baseline.write_text('{}')
            proof=dict(state='reviewed_full_original_budget_two_arm_neural_execution_pending_actual_data',test_exit_code=0,
                test_count=38,actual_MPS_tested=False,production_source_admission_run=False,production_models_fitted=0,
                qualifying_labels_read=False,independent_review='clean',source_sha256={'fixture.py':execution.inputs.sha(source)})
            with patch.object(execution,'ROOT',root),patch.object(execution,'source_files',return_value={source}):
                with self.assertRaises(FileNotFoundError):execution.frozen_plan()
                proof_path.write_text(json.dumps(proof));plan=execution.frozen_plan();execution.check_sources(plan)
                self.assertIn(str(proof_path.relative_to(root)),plan['source_sha256'])
                source.write_text('# changed\n')
                with self.assertRaises(ValueError):execution.frozen_plan()
                source.write_text('# synthetic\n');proof['actual_MPS_tested']=True;proof_path.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):execution.frozen_plan()

    def test_owned_receipt_binds_actual_known_selection_and_tie_rule(self):
        import upstream_rbm400_neural_canonical as canonical
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'experiments/upstream-own-rbm400-neural-selection/selection.json';path.parent.mkdir(parents=True)
            scientific=dict(state=canonical.SELECTION_STATE,qualifying_labels_read=False,both_arms_complete_before_selection=True,
                qualifying_labels_used_for_selection=False,known_fixed_half_scores={'scalar':.86,'latent':.859},selected_arm='latent')
            path.write_text(json.dumps(scientific))
            value=dict(state=execution.RESULTS['select'][1],execution_plan_sha256='1'*64,qualifying_labels_read=False,
                       scientific_proof_sha256=execution.inputs.sha(path))
            with patch.object(execution,'ROOT',root):
                execution.validate_result('select',value,'1'*64)
                with path.open('a') as stream:stream.write(' ')
                with self.assertRaises(ValueError):execution.validate_result('select',value,'1'*64)
                scientific['selected_arm']='scalar';path.write_text(json.dumps(scientific));value['scientific_proof_sha256']=execution.inputs.sha(path)
                with self.assertRaises(ValueError):execution.validate_result('select',value,'1'*64)

    def test_GPU_driver_cache_and_actual_overcap_preserve_original_frame(self):
        import torch
        g=2**30
        self.assertEqual(execution.cap('fit_latent'),16*g);self.assertEqual(execution.cap('canonical_scalar'),3*g)
        workers={'12':dict(bytes=2*g,rss_bytes=2*g),'13':dict(bytes=9*g,rss_bytes=g)}
        current={'compressor_bytes':0,'swap_used_bytes':0};baseline={'swap_used_bytes':0}
        self.assertFalse(execution.continuation_ready(workers,12,'fit_latent',current,baseline,100*g,64*2**20))
        workers['13']=dict(bytes=g,rss_bytes=7*g)
        self.assertTrue(execution.continuation_ready(workers,12,'fit_latent',current,baseline,100*g,64*2**20))
        for allocated,expected_stops in [([2*g],0),([17*g,2*g],1)]:
            with tempfile.TemporaryDirectory() as folder:
                report=Path(folder);path=report/'first-scientific-baseline.json';path.write_text('{}')
                guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
                guard.phase='fit_latent';guard.limit=16*g;guard.last=0.;guard.counter=0;guard.plan={};guard.baseline=baseline
                guard.baseline_sha=execution.inputs.sha(path);guard.predecessors={};guard.captured={'pid':777}
                sample={'777':dict(bytes=2*g,rss_bytes=2*g)}
                with patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'),patch.object(execution.os,'getpid',return_value=777),\
                     patch.object(execution.resources,'footprints',return_value=sample),patch.object(execution.memory,'memory',return_value=current),\
                     patch.object(execution,'stop_child',return_value=True) as stop,\
                     patch.object(torch.mps,'driver_allocated_memory',side_effect=[17*g,2*g]),\
                     patch.object(torch.mps,'current_allocated_memory',side_effect=allocated),patch.object(torch.mps,'empty_cache'):
                    guard()
                self.assertEqual(stop.call_count,expected_stops);self.assertEqual(guard.counter,expected_stops)
                self.assertEqual(guard.baseline_sha,execution.inputs.sha(path))

    def test_fresh_worker_import_makes_no_MPS_allocator_or_production_call(self):
        code="import torch; fail=lambda *a,**k: (_ for _ in ()).throw(AssertionError('No MPS allocator call')); torch.mps.current_allocated_memory=fail; torch.mps.driver_allocated_memory=fail; torch.mps.empty_cache=fail; import upstream_rbm400_neural_worker as w; print(w.execution.PHASES)"
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent))
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True)
        self.assertIn('select',result.stdout)


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
        workers={'12':dict(bytes=2*g,rss_bytes=2*g),'13':dict(bytes=22*g,rss_bytes=1*g)}
        self.assertFalse(execution.continuation_ready(workers,12,'prepare',current,baseline,100*g,64*2**20))
        workers['13']=dict(bytes=g,rss_bytes=22*g)
        self.assertFalse(execution.continuation_ready(workers,12,'prepare',current,baseline,100*g,64*2**20))
        workers['13']=dict(bytes=g,rss_bytes=g)
        self.assertTrue(execution.continuation_ready(workers,12,'prepare',current,baseline,100*g,64*2**20))


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
                code="import sys; from pathlib import Path; import upstream_rbm400_neural_execution as e; e.REPORT=Path(sys.argv[1]); e.exclusive_json=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('disk floor')); e.watchdog(2147483647,[int(x) for x in sys.argv[2:]])"
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


    def test_self_pause_preserves_guard_and_original_baseline_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder);baseline_path=report/'first-scientific-baseline.json';baseline_path.write_text('{}')
            guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
            guard.phase='prepare';guard.limit=3*2**30;guard.last=0.;guard.counter=0;guard.plan={};guard.baseline={}
            guard.baseline_sha=execution.inputs.sha(baseline_path);guard.predecessors={};guard.captured={'pid':777}
            workers={'777':dict(bytes=1,rss_bytes=1)};current={'compressor_bytes':0,'swap_used_bytes':0}
            with patch.object(execution,'REPORT',report),patch.object(execution,'check_sources'),patch.object(execution.os,'getpid',return_value=777),patch.object(execution.resources,'footprints',return_value=workers),patch.object(execution.memory,'memory',return_value=current),patch.object(execution,'safe',side_effect=[False,True]),patch.object(execution,'stop_child',return_value=True) as stop:
                guard(extra_disk_bytes=123)
            stop.assert_called_once_with(guard.captured);self.assertEqual(guard.counter,1)
            self.assertEqual(guard.baseline_sha,execution.inputs.sha(baseline_path))
            self.assertEqual(json.loads((report/'prepare-worker-resource-request.json').read_text())['extra_disk_reserve'],execution.output_remaining())


    def test_measurement_failures_preserve_the_same_worker_stack(self):
        failures=[subprocess.CalledProcessError(1,['vmmap']),ValueError('malformed memory number'),AttributeError('missing memory field'),OSError('disk query')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as folder:
                report=Path(folder);path=report/'first-scientific-baseline.json';path.write_text('{}')
                guard=execution.WorkerGuard.__new__(execution.WorkerGuard)
                guard.phase='prepare';guard.limit=3*2**30;guard.last=0.;guard.counter=0;guard.plan={}
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



if __name__=='__main__':unittest.main()
