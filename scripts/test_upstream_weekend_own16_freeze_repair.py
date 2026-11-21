"""Full16 freeze regression on tiny arrays; no real predictions or labels."""
from contextlib import ExitStack,contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import upstream_own16_weekend_trajectory_calibration as original
import upstream_weekend_own16_freeze_repair as repair
import upstream_weekend_own16_queue_v2 as queue


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
        root=Path(folder).resolve();report=root/'report';run=root/'run';bag=root/'bag';bag_run=root/'bag-run'
        for path in (report,run,bag,bag_run):path.mkdir()
        for obj,name,value in ((original.u,'ROOT',root),(original,'REPORT',report),(original,'RUN',run),(original,'BAG',bag),(original,'BAG_RUN',bag_run),(original,'COUNTS',{'known':2,'qual':3})):
            stack.enter_context(patch.object(obj,name,value))
        rows=[]
        for index,name in enumerate(original.BASE_IDS+['nn-weekend-trajectory-fold-bag']):
            path=root/('%02d.f32'%index);np.array([1+index/10,3,4.9-index/50],dtype='<f4').tofile(path)
            rows.append({'id':name,'qualifying_prediction':str(path)})
        plan={'families':16,'primary':original.CANDIDATES[0],'candidates':original.CANDIDATES,'ridge_lambda':10.,
            'script_sha256':original.sha(Path(original.__file__)),
            'verifier_sha256':original.sha(Path(queue.verification.__file__)),
            'declaration_sha256':original.sha(original.DECL),'qualifying_labels_read':False}
        cal={'families':rows,'family_ids':[r['id'] for r in rows],'simplex_weights':[1/16]*16,'ridge_weights':[.01*(i+1) for i in range(16)],'ridge_intercept':.1,'source_sha256':{},'qualifying_labels_read':False,'known_score_is_nested_cross_validation':False}
        for name,value in (('plan.json',plan),('calibration.json',cal)):(report/name).write_text(json.dumps(value))
        outputs={}
        for name in ('known-simplex','known-ridge10','known-fixed-half'):
            path=run/(name+'.f32');np.array([2.5,3.5],dtype='<f4').tofile(path);outputs[name]=original.sha(path)
        done={'state':'complete_original16_known_only_coefficients_before_candidate_freeze','calibration_sha256':original.sha(report/'calibration.json'),
            'plan_sha256':original.sha(report/'plan.json'),'qualifying_labels_read':False,'known_predictions':outputs}
        (report/'known-completion.json').write_text(json.dumps(done))
        bag_vector=bag_run/'nn-bag.f32';bag_vector.write_bytes(Path(rows[-1]['qualifying_prediction']).read_bytes())
        proof={'state':'complete_independent_complete_trajectory_actual_fold_models_all_query_streams_and_bag','seed_slots':15,'rows':3,
            'all_original_model_and_input_streams_replayed':True,'uniform_five_fold_bag_bitwise_equal':True,'matched_B_raw_operators_on_both_queries':True,
            'known_OOF_and_qualifying_bag_constructions_distinct':True,'qualifying_labels_read':False,'models_fitted':0,'qualifying_score_claimed':False,
            'source_sha256':{},'prediction_sha256':original.sha(bag_vector)}
        (bag/'independent-verification.json').write_text(json.dumps(proof))
        yield root,report,run,rows,cal,outputs


class FreezeRegressionTests(unittest.TestCase):
    def test_original_rejects_all16_families_before_any_qualifying_write(self):
        with fixture() as (root,report,run,rows,cal,outputs):
            self.assertEqual(len(rows),16)
            with self.assertRaisesRegex(ValueError,'All complete finite qualifying families required'):original.freeze()
            self.assertEqual({p.stem for p in run.iterdir()},set(outputs))
            self.assertFalse((report/'frozen.json').exists())

    def test_repair_retains_every_family_original_coefficients_and_all_six_vector_bits(self):
        with fixture() as (root,report,run,rows,cal,known_outputs):
            original_metadata={p.name:p.read_bytes() for p in report.iterdir()}
            marker=root/'repair-source.txt';marker.write_text('synthetic source binding only')
            with patch.object(repair,'protocol_bindings',return_value={'repair-source.txt':original.sha(marker)}):
                repair.freeze(lambda **kwargs:None)
            frozen=original.u.read(report/'frozen.json')
            self.assertEqual(frozen['state'],'complete_all_three_own16_candidates_frozen_before_any_new_audit')
            self.assertEqual(len(frozen['predictions']),6);self.assertFalse(frozen['qualifying_labels_read'])
            self.assertEqual(frozen['freeze_cardinality_repair']['models_refitted'],0)
            for name,payload in original_metadata.items():self.assertEqual((report/name).read_bytes(),payload)
            for name,digest in known_outputs.items():self.assertEqual(original.sha(run/(name+'.f32')),digest)
            arrays=[np.fromfile(row['qualifying_prediction'],dtype='<f4') for row in rows]
            specs=[(arrays,cal['simplex_weights'],0.),(arrays,cal['ridge_weights'],cal['ridge_intercept']),([arrays[-1],arrays[5]],[.5,.5],0.)]
            for name,(columns,weights,bias) in zip(original.CANDIDATES,specs):
                values=[]
                for index in range(3):
                    value=float(bias)
                    for column,weight in zip(columns,weights):value+=float(column[index])*float(weight)
                    values.append(min(4.95,max(1.05,value)))
                expected=np.array(values,dtype='<f4');actual=np.fromfile(run/(name+'.f32'),dtype='<f4')
                np.testing.assert_array_equal(actual.view('u4'),expected.view('u4'))
                self.assertEqual(frozen['predictions'][name],original.sha(run/(name+'.f32')))
            self.assertIn('repair-source.txt',frozen['source_sha256'])
            verified_plan,verified_calibration,verified_freeze=queue.verification.evidence()
            self.assertEqual(verified_plan['families'],16)
            self.assertEqual(verified_calibration['family_ids'],original.BASE_IDS+['nn-weekend-trajectory-fold-bag'])
            self.assertEqual(verified_freeze,frozen)
            with self.assertRaises(ValueError):repair.freeze(lambda **kwargs:None)

    def test_fifteen_family_subset_or_nonfinite_or_truncated_query_is_rejected(self):
        for change in ('subset','nan','truncated'):
            with self.subTest(change=change),fixture() as (root,report,run,rows,cal,outputs):
                if change=='subset':
                    cal['families']=rows[:-1];cal['family_ids']=cal['family_ids'][:-1]
                    (report/'calibration.json').write_text(json.dumps(cal))
                    done=original.u.read(report/'known-completion.json');done['calibration_sha256']=original.sha(report/'calibration.json')
                    (report/'known-completion.json').write_text(json.dumps(done))
                else:
                    np.array([1.,np.nan,3.] if change=='nan' else [1.,2.],dtype='<f4').tofile(rows[0]['qualifying_prediction'])
                with patch.object(repair,'protocol_bindings',return_value={}):
                    with self.assertRaises(ValueError):repair.freeze(lambda **kwargs:None)
                self.assertEqual({p.stem for p in run.iterdir()},set(outputs))


class IdleHandoffTests(unittest.TestCase):
    def test_import_does_not_signal_or_launch_a_production_process(self):
        code="import sys; import upstream_weekend_own16_queue_v2 as q; assert 'torch' not in sys.modules; assert q.command('freeze')[-2].endswith('upstream_weekend_own16_freeze_repair.py'); print('metadata only')"
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)),capture_output=True,text=True,check=True)
        self.assertIn('metadata only',result.stdout)

    def test_existing_scientific_namespace_prevents_idle_retirement(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();old=root/'old';old.mkdir();science=root/'science';science.mkdir()
            owner={'pid':7,'pgid':7,'started':'original','command':' '.join([sys.executable,'-u',str(root/'scripts/upstream_own16_weekend_trajectory_queue.py')]),'state':'S'}
            (old/'monitor.json').write_text(json.dumps(owner))
            with patch.object(queue,'ROOT',root),patch.object(queue,'OLD_REPORT',old),patch.object(queue.study,'REPORT',science),patch.object(queue.rbm.identity,'process',return_value=owner),patch.object(queue.os,'kill') as kill:
                with self.assertRaisesRegex(ValueError,'No prior scientific fit'):queue.inspect_idle_predecessor()
                kill.assert_not_called()

    def test_actual_idle_owner_and_sole_watchdog_retire_without_any_scientific_signal(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder).resolve();old=root/'old';protocol=root/'protocol';scripts=root/'scripts'
            for path in (old,protocol,scripts):path.mkdir()
            script=scripts/'upstream_own16_weekend_trajectory_queue.py'
            script.write_text('''import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if len(sys.argv)>1:
    owner=sys.argv[2]
    while True:
        state=subprocess.run(['ps','-p',owner,'-o','stat='],text=True,capture_output=True).stdout.strip()
        if not state or state.startswith('Z'):break
        time.sleep(.05)
else:
    def end(signum,frame):
        (root/'old/failure.json').write_text(json.dumps({'error':'synthetic idle termination','child':None}))
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,end)
    subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid())],start_new_session=True)
    while True:time.sleep(.05)
''')
            (old/'plan.json').write_text(json.dumps({'source_sha256':{str(script.relative_to(root)):queue.inputs.sha(script)}}))
            fake_queue=scripts/'upstream_weekend_own16_queue_v2.py';fake_repair=scripts/'upstream_weekend_own16_freeze_repair.py'
            original_fit=scripts/'upstream_own16_weekend_trajectory_calibration.py'
            document=root/'repair.md';test_file=scripts/'test_upstream_weekend_own16_freeze_repair.py'
            for path in (fake_queue,fake_repair,document,test_file,original_fit):path.write_text('synthetic control fixture only')
            proof={'state':'verified_weekend16_original_failure_and_corrected_full_freeze_CPU_checks_only',
                'source_sha256':{str(path.relative_to(root)):queue.inputs.sha(path) for path in (script,fake_queue,fake_repair,document,test_file,original_fit)}}
            (protocol/'protocol-verification.json').write_text(json.dumps(proof))
            owner=subprocess.Popen([sys.executable,'-u',str(script)],start_new_session=True)
            try:
                actual=queue.rbm.identity.process(owner.pid);(old/'monitor.json').write_text(json.dumps(actual))
                for obj,name,value in ((queue,'ROOT',root),(queue,'OLD_REPORT',old),(queue,'PROTOCOL',protocol),(queue,'__file__',str(fake_queue)),
                    (queue.study,'__file__',str(fake_repair)),(queue.study,'DECL',document),(queue.study,'REPORT',root/'science'),(queue.study,'ART',root/'art'),(queue.rbm,'ROOT',root)):
                    stack.enter_context(patch.object(obj,name,value))
                deadline=time.monotonic()+5
                while True:
                    try:before=queue.inspect_idle_predecessor();break
                    except ValueError:
                        if time.monotonic()>deadline:raise
                        time.sleep(.05)
                real_kill=os.kill
                with patch.object(queue.os,'kill',wraps=real_kill) as signals:queue.retire_idle_predecessor()
                self.assertTrue(signals.call_args_list)
                self.assertEqual({call.args[0] for call in signals.call_args_list},{owner.pid})
                owner.wait(timeout=5);self.assertEqual(owner.returncode,0)
                handoff=json.loads((protocol/'idle-handoff.json').read_text())
                self.assertFalse(handoff['scientific_work_dispatched']);self.assertFalse(handoff['baseline_reset'])
                self.assertTrue(handoff['original_owner_and_watchdog_terminal'])
                self.assertIsNone(queue.rbm.identity.process(before['watchdog']['pid']))
                with self.assertRaises(ValueError):queue.retire_idle_predecessor()
            finally:
                if owner.poll() is None:owner.terminate();owner.wait(timeout=5)


if __name__=='__main__':unittest.main()
