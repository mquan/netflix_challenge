"""Small real calibration/verification fixtures; no production arrays/outcomes."""
from contextlib import ExitStack
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
from unittest.mock import patch
import numpy as np
import upstream_local_rbm_blend_metal as study
import verify_upstream_local_rbm_blend_metal as verify
import upstream_local_rbm_blend_queue_metal as queue


class ProtocolTests(unittest.TestCase):
    def test_imports_do_not_start_work_or_import_torch(self):
        code="import sys; import upstream_local_rbm_blend_queue_metal as q; assert 'torch' not in sys.modules; assert q.PHASES==('known','freeze','preaudit','audit','canonical'); print('metadata imports only')"
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)),capture_output=True,text=True,check=True)
        self.assertIn('metadata imports only',result.stdout)

    def test_asymmetric_continuation_reserves_each_full_metric_and_original_swap(self):
        g=2**30;workers={'1':dict(bytes=g/4,rss_bytes=g),'2':dict(bytes=23*g,rss_bytes=g)}
        current={'compressor_bytes':0,'swap_used_bytes':0};baseline={'swap_used_bytes':0}
        self.assertFalse(queue.continuation_ready(workers,1,current,baseline,100*g,64*2**20))
        workers['2']=dict(bytes=g,rss_bytes=23*g)
        self.assertFalse(queue.continuation_ready(workers,1,current,baseline,100*g,64*2**20))
        workers['2']=dict(bytes=g,rss_bytes=g)
        self.assertTrue(queue.continuation_ready(workers,1,current,baseline,100*g,64*2**20))
        current['swap_used_bytes']=128*2**20+1
        self.assertFalse(queue.continuation_ready(workers,1,current,baseline,100*g,64*2**20))

    def test_cpu_guard_accounting_error_returns_through_same_stack(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder).resolve();path=report/'first-scientific-baseline.json';path.write_text('{}')
            (report/'prerequisites.json').write_text(json.dumps({'source_sha256':{}}))
            guard=queue.Guard.__new__(queue.Guard);guard.phase='known';guard.plan={};guard.baseline={'swap_used_bytes':0}
            guard.baseline_sha=study.inputs.sha(path);guard.owner={'pid':123};guard.gate={'predecessor_sha256':{},'watchdog':{'pid':124}}
            guard.captured={'pid':777};guard.last=0.;guard.counter=0
            with patch.object(queue,'REPORT',report),patch.object(queue,'check_sources'),patch.object(queue.os,'getpid',return_value=777),patch.object(queue.rbm.resources,'footprints',side_effect=[subprocess.CalledProcessError(1,['vmmap']),{'777':dict(bytes=1,rss_bytes=1)}]),patch.object(queue.rbm.memory,'memory',return_value={'compressor_bytes':0,'swap_used_bytes':0}),patch.object(queue.rbm.identity,'process',return_value={}),patch.object(queue.rbm.identity,'same_owner',return_value=True),patch.object(queue.rbm,'stop_child',return_value=True) as stop:
                guard()
            stop.assert_called_once_with(guard.captured);self.assertEqual(guard.counter,1)
            self.assertEqual(guard.baseline_sha,study.inputs.sha(path))

    def test_owner_loss_retains_real_primary_lane_even_if_evidence_write_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder).resolve();owner={'pid':2147483647,'started':'missing','command':'missing'}
            (report/'monitor.json').write_text(json.dumps(owner))
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
            lane=(report/'lane.lock').open('a');watch=None
            try:
                fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                (report/'child-ownership.json').write_text(json.dumps(study.rbm.identity.process(child.pid)))
                code="import sys; from pathlib import Path; import upstream_local_rbm_blend_queue_metal as q; q.REPORT=Path(sys.argv[1]); q.rbm.exclusive_json=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('disk floor')); q.watchdog(2147483647,int(sys.argv[2]))"
                watch=subprocess.Popen([sys.executable,'-c',code,str(report),str(lane.fileno())],env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)),pass_fds=(lane.fileno(),))
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if 'T' in study.rbm.identity.process(child.pid)['state']:break
                    time.sleep(.05)
                self.assertIn('T',study.rbm.identity.process(child.pid)['state']);self.assertIsNone(watch.poll())
                lane.close()
                with (report/'lane.lock').open('a') as contender:
                    with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);watch.wait(timeout=5)
                self.assertEqual(watch.returncode,0)
                with (report/'lane.lock').open('a') as contender:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                if watch is not None and watch.poll() is None:watch.terminate();watch.wait(timeout=5)
                lane.close()

    def test_recovered_csv_cannot_enter_fitting_closure(self):
        with patch.object(study.rbm,'read',return_value={'synthetic_qualifying_payload_sha256':'a'*64}):
            for path in ('experiments/recovered-audit/qual_ratings.csv.gz','artifacts/upstream-original1200/source/data/qual_ratings/qual_ratings.csv.gz'):
                with self.assertRaisesRegex(ValueError,'Recovered qualifying targets'):
                    study.verify_bindings({path:'b'*64},lambda **kwargs:None)

    def test_gpu_preflight_expected_digest_cannot_be_rebased_to_new_receipt(self):
        from dataclasses import asdict
        from upstream_factored_rbm_preflight import REQUIRED_STRESS
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();report=root/'gpu-control';data=root/'gpu-study'
            report.mkdir();data.mkdir()
            stack.enter_context(patch.object(study,'ROOT',root));stack.enter_context(patch.object(study.rbm,'REPORT',report))
            stack.enter_context(patch.object(study.rbm,'STUDY',data));stack.enter_context(patch.object(study.rbm,'check_sources'))
            plan={'configuration':asdict(study.rbm.Config()),'source_sha256':{}}
            (report/'plan.json').write_text(json.dumps(plan));plan_sha=study.inputs.sha(report/'plan.json')
            base=dict(execution_plan_sha256=plan_sha,qualifying_labels_read=False)
            preflight=dict(base,state=study.rbm.RESULTS['mps_preflight'][1],device='mps',actual_MPS_tested=True,
                passed_for_production_device=True,full_required_stress_tested=True,fresh_mps_runtime_import=True,
                cases={'stress':{'shape':REQUIRED_STRESS}})
            (data/'mps-preflight.json').write_text(json.dumps(preflight));original_sha=study.inputs.sha(data/'mps-preflight.json')
            paired=dict(base,state=study.rbm.RESULTS['assemble'][1],configuration=plan['configuration'],models_fitted=2,
                epochs_each=15,score_computed=False,known={'rows':study.COUNTS['known']},qualifying={'rows':study.COUNTS['qualifying']},
                preflight_sha256=original_sha)
            (data/'paired-verification.json').write_text(json.dumps(paired))
            (report/'completion.json').write_text(json.dumps(dict(state='complete_full_own_gpu_rbm400_original15_15_and_independent_queries',
                verification_sha256=study.inputs.sha(data/'paired-verification.json'),qualifying_labels_read=False)))
            preflight['new_unrelated_receipt_marker']='must not replace original expected digest'
            (data/'mps-preflight.json').write_text(json.dumps(preflight))
            with self.assertRaisesRegex(ValueError,'Full original paired model population/recipe required'):
                study.gpu_sources({})


class FullSyntheticStudyTests(unittest.TestCase):
    def test_known_scoring_reclips_float32_lower_bound_before_error(self):
        value=np.array([1.05],dtype='<f4')
        self.assertLess(float(value[0]),1.05)
        self.assertAlmostEqual(verify.known_rmse(value,np.array([1.])),.05,places=14)

    def test_all16_real_fit_full65535_support_replay_freeze_and_separate_audits(self):
        rng=np.random.default_rng(828);nk,nq=80,61
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();report=root/'study';art=root/'art';old=root/'old';known=root/'known'
            for path in (report,art,old,known):path.mkdir()
            base=root/'base';base.mkdir();(base/'fresh-training-plan.json').write_text(json.dumps({'synthetic_qualifying_payload_sha256':'a'*64}))
            rows=[];bindings={}
            for index,name in enumerate(study.BASE_IDS+[study.FAMILY_ID]):
                values={'known':rng.normal(3,.7,nk).astype('<f4'),'qualifying':rng.normal(3,.7,nq).astype('<f4')}
                row={'id':name,'family':'synthetic'}
                for query,value in values.items():
                    filename='%s-%s.%s'%(index,query,'npy' if index==15 else 'f32');path=root/filename
                    if index==15:
                        with path.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                        row['format']='npy'
                    else:value.tofile(path)
                    row[query+'_prediction']=filename;bindings[filename]=study.inputs.sha(path)
                rows.append(row)
            y=rng.integers(1,6,nk,dtype=np.int8)
            with (known/'labels.npy').open('wb') as stream:np.save(stream,y,allow_pickle=False)
            bindings['known/labels.npy']=study.inputs.sha(known/'labels.npy')
            rng.normal(3,.4,nk).astype('<f4').tofile(old/'known-simplex.f32')
            rng.normal(3,.4,nq).astype('<f4').tofile(old/(study.preceding.CANDIDATES[0]+'.f32'))
            for path in old.iterdir():bindings[str(path.relative_to(root))]=study.inputs.sha(path)
            for obj,name,value in [(study,'ROOT',root),(study.rbm,'ROOT',root),(study,'REPORT',report),(study,'ART',art),(study,'COUNTS',{'known':nk,'qualifying':nq}),(study.preceding,'KNOWN',known),(study.preceding,'RUN',old),(study.preceding.u,'BASE',base),(verify,'GROUP_COUNTS',{0:30,1:31})]:
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(study,'source_declaration',return_value=(rows,bindings)))
            guard=lambda **kwargs:None
            outcomes=np.column_stack([rng.integers(1,6,nq,dtype=np.int16),np.r_[np.zeros(30,dtype=np.int16),np.ones(31,dtype=np.int16)]])
            def gated_outcomes(guard):
                self.assertTrue((report/'independent-preaudit-verification.json').is_file())
                return outcomes,'synthetic-fixture-only'
            label_reader=stack.enter_context(patch.object(verify,'labels',side_effect=gated_outcomes))
            study.known(guard);study.frozen(guard)
            with self.assertRaises(FileNotFoundError):verify.audit(guard)
            label_reader.assert_not_called()
            verify.verify(guard)
            pre=study.rbm.read(report/'independent-preaudit-verification.json')
            self.assertEqual(pre['simplex_supports_examined'],65535);self.assertEqual(pre['ridge_equations'],17)
            self.assertFalse(pre['qualifying_labels_read']);label_reader.assert_not_called()
            verify.audit(guard);verify.canonical(guard);self.assertEqual(label_reader.call_count,2)
            result=study.rbm.read(report/'independent-canonical-verification.json')
            self.assertEqual(result['primary'],study.CANDIDATES[0]);self.assertFalse(result['qualifying_labels_used_for_fitting_or_selection'])
            self.assertFalse(result['official_or_untouched_test']);self.assertEqual(result['groups']['source0']['rows'],30)
            self.assertEqual(result['groups']['source1']['rows'],31);self.assertEqual(len(result['results']),3)
            # Alter a vector and its recorded hash: actual independent arithmetic
            # must still detect the change, even with newly authenticated bytes.
            path=art/'known-simplex.f32';changed=np.fromfile(path,dtype='<f4');changed[0]+=np.float32(.125);changed.tofile(path)
            freeze=study.rbm.read(report/'frozen.json');freeze['outputs']['known-simplex']['sha256']=study.inputs.sha(path)
            (report/'frozen.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(ValueError,'Every actual stored arithmetic bit'):
                verify.verify(guard)

    def test_exact_population_schema_and_file_identity_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'values.npy'
            with path.open('wb') as stream:np.save(stream,np.array([1,2,3],dtype='<f4'),allow_pickle=False)
            records=study.inputs.verify_files(root,{'values.npy':study.inputs.sha(path)},lambda:None)
            with patch.object(study,'ROOT',root):
                with study.vector('values.npy',3,records,'npy') as value:np.testing.assert_array_equal(value,[1,2,3])
                with self.assertRaises(ValueError):
                    with study.vector('values.npy',2,records,'npy'):pass
                before=path.read_bytes();path.unlink();path.write_bytes(before)
                with self.assertRaises(ValueError):
                    with study.vector('values.npy',3,records,'npy'):pass


if __name__=='__main__':unittest.main()
