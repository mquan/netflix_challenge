"""Owned query-controller fixtures and actual root bootstrap; no production run."""
from contextlib import contextmanager,ExitStack
import copy
from types import SimpleNamespace
import hashlib
import io
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
import upstream_full_context_catboost_bag_execution_v1 as e


class ExecutionTests(unittest.TestCase):

    def test_all_four_complete_phases_original_family_budget_and_full_output_reserve(self):
        self.assertEqual(e.PHASES,('prepare','infer','replay','admit'))
        self.assertEqual(e.FAMILY_LIMIT,16*2**30)
        self.assertEqual(e.OUTPUT_RESERVE,768*2**20)
        self.assertGreaterEqual(e.OUTPUT_INVENTORY_MiB['prediction_vectors']*2**20,(6*2817131+1128164)*4)
        self.assertEqual(e.policy()['maximum_swap_growth_bytes'],128*2**20)
        self.assertEqual(e.policy()['disk_floor_bytes'],int(18.75*2**30))
        for phase in e.PHASES:self.assertEqual(e.command(phase)[-1],phase)
        for phase in ('prepare','admit','partial'):
            with self.assertRaises(ValueError):e.producer_command(phase,10)
        self.assertNotIn('torch',sys.modules)

    def test_source_policy_dimensions_and_runtime_cannot_be_rebased(self):
        with self.fixture() as (root,control):
            plan=self.plan(root);e.check_sources(plan)
            for field,value in [('known_rows',20),('seed_slots',14),('fold_slots',4),('block_rows',4096),
                    ('models_fitted',1),('qualifying_labels_read',True),('known_only_selection_required',False)]:
                bad=copy.deepcopy(plan);bad[field]=value
                with self.assertRaises(ValueError):e.check_sources(bad)
            bad=copy.deepcopy(plan);bad['policy']['family_bytes']=17*2**30
            with self.assertRaises(ValueError):e.check_sources(bad)
            (root/'fixture-source.py').write_text('# newer source cannot replace expected bytes')
            with self.assertRaisesRegex(ValueError,'source changed'):e.check_sources(plan)

    def test_completed_known_selection_is_required_and_primary_pass_can_skip_only_with_full_audit(self):
        with self.fixture() as (root,control),patch.object(e.known_owner,'REPORT',root/'previous'):
            self.write(e.SELECTION,{'synthetic_selection':True})
            known=dict(arm='latent',selection_sha256=e.study.sha(e.SELECTION),source_sha256={'selection.json':e.study.sha(e.SELECTION)})
            with patch.object(e.selected,'load',return_value=known) as load:
                result=e.prerequisites();self.assertTrue(result['start_followup']);self.assertEqual(result['selected_arm'],'latent')
                load.assert_called_once_with(e.study.sha(e.SELECTION),e.metadata_guard)
            not_needed=e.known_owner.REPORT/'not-needed.json'
            self.write(not_needed,dict(state='own17_primary_already_meets_original_target_no_tree_study_started'))
            with patch.object(e.known_owner,'prerequisites',return_value=dict(start_followup=False,source_sha256={})):
                result=e.prerequisites();self.assertFalse(result['start_followup']);self.assertFalse(result['qualifying_outcomes_read'])
            with patch.object(e.known_owner,'prerequisites',return_value=dict(start_followup=True,source_sha256={})):
                with self.assertRaisesRegex(ValueError,'primary target audit'):e.prerequisites()

    def test_exact_full_query_source_request_and_no_outcome_additions(self):
        with self.fixture() as (root,control),ExitStack() as stack:
            worker=root/'upstream_fixture_query_worker.py';worker.write_text('# worker')
            code=root/'upstream_fixture_pipeline.py';code.write_text('# pipeline')
            query=root/'metadata.json';self.write(query,{'synthetic':True})
            digest=e.study.sha(query);expected={'admission_sha256':'a'*64}
            known=dict(arm='scalar',selection_sha256='b'*64,source_sha256={code.name:e.study.sha(code)})
            metadata=dict(source_sha256=dict(known['source_sha256']))
            captured=dict(pid=10,ppid=1,pgid=10,state='R',started='root',command='root-worker')
            role=dict(pid=11,ppid=10,pgid=10,state='R',started='source',command='source-worker')
            self.write(control/'infer-source-bootstrap.json',dict(scientific_root_pid=10,roles={'producer':role}))
            stack.enter_context(patch.object(e,'WORKER',worker));stack.enter_context(patch.object(e,'authorize',return_value=captured))
            stack.enter_context(patch.object(e,'query_inputs',return_value=(known,query,digest,metadata,expected)))
            stack.enter_context(patch.object(e.pipeline,'code_paths',return_value=[code]))
            stack.enter_context(patch.object(e.identity,'process',return_value=dict(role,state='S')))
            bindings={**known['source_sha256'],query.name:digest,worker.name:e.study.sha(worker)}
            request=dict(mode='infer',arm='scalar',command=e.producer_command('infer',10),scientific_root=captured,producer=role,
                selection_sha256='b'*64,completion_sha256=None,query_metadata_path=str(query),query_metadata_sha256=digest,
                source_metadata_sha256='a'*64,contract_sha256=hashlib.sha256(e.study.wire.encoded(expected)).hexdigest(),
                source_sha256=bindings,qualifying_labels_read=False)
            gate=e.admit_source('infer',captured,request,lambda **kw:None)
            self.assertEqual(e.read(gate['path'])['request'],request)
            # A source captured just before its bootstrap write must remain in
            # the same admission frame until that exact role is registered.
            (control/'infer-source-bootstrap.json').unlink()
            (control/'infer-roles.json').unlink();(control/'infer-source-authorization.json').unlink()
            samples=[]
            def delayed_registration(**kw):
                if (control/'infer-source-bootstrap.json').exists():return
                samples.append(True)
                self.write(control/'infer-source-bootstrap.json',dict(scientific_root_pid=10,roles={'producer':role}))
            e.admit_source('infer',captured,request,delayed_registration)
            self.assertEqual(samples,[True])
            bad=copy.deepcopy(request);bad['source_sha256']['unapproved-qualifying-outcomes']='c'*64
            with self.assertRaisesRegex(ValueError,'complete original query source closure'):
                e.admit_source('infer',captured,bad,lambda **kw:None)
            bad=copy.deepcopy(request);bad['source_metadata_sha256']=digest
            with self.assertRaisesRegex(ValueError,'selection/metadata'):
                e.admit_source('infer',captured,bad,lambda **kw:None)

    def test_all_four_worker_dispatches_and_separate_source_preparation(self):
        import upstream_full_context_catboost_bag_worker_v1 as worker
        with self.fixture() as (root,control),ExitStack() as stack:
            events=[];guard=lambda **kw:None;capture={'pid':10}
            self.write(control/'plan.json',{'synthetic':True});self.write(control/'prerequisites.json',{'selection_sha256':'a'*64})
            q=root/'query.json';self.write(q,{'synthetic':True})
            run=root/'bag-art';report=root/'bag-report';report.mkdir()
            self.write(report/'completion.json',{'synthetic':True})
            known=dict(arm='scalar',selection_sha256='a'*64,arms={'scalar':{'metadata':{'actual_known_metadata':True}}})
            metadata={'arm':'scalar','original_known_source_metadata':known['arms']['scalar']['metadata']}
            stack.enter_context(patch.object(e,'authorize',return_value=capture))
            stack.enter_context(patch.object(e,'control',return_value=SimpleNamespace(cached_roles={})))
            stack.enter_context(patch.object(e.supervision,'Guard',return_value=guard))
            stack.enter_context(patch.object(e.selected,'load',return_value=known))
            stack.enter_context(patch.object(e.bag,'paths',return_value=(run,report)))
            stack.enter_context(patch.object(e,'query_inputs',return_value=(known,q,e.study.sha(q),metadata,{})))
            def prepare(arm,g):
                events.append(('prepare',arm));return dict(metadata=metadata,contract={'admission_sha256':'b'*64})
            stack.enter_context(patch.dict(sys.modules,{'upstream_full_context_catboost_query_producer':SimpleNamespace(prepare=prepare)}))
            def pipeline(mode,selection,path,digest,**kwargs):
                events.append((mode,selection));self.assertEqual(kwargs['command'],e.producer_command(mode,10))
                if mode=='replay':self.assertEqual(kwargs['completion_sha256'],e.study.sha(report/'completion.json'))
            stack.enter_context(patch.object(e.pipeline,'run',side_effect=pipeline))
            stack.enter_context(patch.object(e,'admit_pair',side_effect=lambda g:events.append(('admit',None))))
            validate=stack.enter_context(patch.object(e,'validate_result'))
            scientific=root/'synthetic.json';self.write(scientific,{'synthetic_scientific_body':True})
            stack.enter_context(patch.object(e,'scientific_path',return_value=(scientific,'synthetic')))
            with patch('builtins.print'):
                for phase in e.PHASES:
                    worker.main(phase);value=e.read(e.RESULTS/(phase+'.json'))
                    self.assertEqual(value['state'],'complete_owned_original_tree_pair_'+phase)
            self.assertEqual(events,[('prepare','scalar'),('infer','a'*64),('replay','a'*64),('admit',None)])
            preparation=e.read(e.SOURCE_REPORT/'preparation.json')
            self.assertEqual(preparation['logical_source_admission_sha256'],'b'*64)
            self.assertEqual(preparation['metadata_sha256'],e.study.sha(e.SOURCE_REPORT/'metadata.json'))
            self.assertEqual(validate.call_count,4);self.assertNotIn('torch',sys.modules)




    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();control=root/'control';control.mkdir()
            for module,name,value in [(e,'ROOT',root),(e.study,'ROOT',root),(e,'REPORT',control),(e,'RESULTS',root/'results'),
                    (e,'SOURCE_REPORT',root/'query-source'),(e,'SELECTION',root/'selection.json')]:
                stack.enter_context(patch.object(module,name,value))
            yield root,control

    def plan(self,root):
        source=root/'fixture-source.py';source.write_text('# original synthetic source\n')
        return dict(state=e.PLAN_STATE,phases=list(e.PHASES),environment=e.ENV,policy=e.policy(),libraries=e.study.versions(),python=sys.version,
            known_rows=1128164,qualifying_rows=2817131,fold_slots=5,seed_slots=15,block_rows=8192,
            models_fitted=0,qualifying_labels_read=False,known_only_selection_required=True,
            source_sha256={'fixture-source.py':e.study.sha(source)})


    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return e.study.sha(path)


    def test_early_and_registered_role_tables_cannot_replace_original_lifetime(self):
        with self.fixture() as (root,control):
            captured={'pid':10};role={'pid':11,'started':'original'}
            self.write(control/'infer-source-bootstrap.json',dict(scientific_root_pid=10,roles={'producer':role}))
            self.write(control/'infer-roles.json',dict(scientific_root_pid=10,roles={'producer':role}))
            self.assertEqual(e.roles('infer',captured),{'producer':role})
            self.write(control/'infer-roles.json',dict(scientific_root_pid=10,roles={'producer':dict(role,pid=12)}))
            with self.assertRaises(ValueError):e.roles('infer',captured)


    def test_live_source_state_may_change_but_lifetime_or_command_may_not(self):
        role=dict(pid=11,ppid=10,pgid=10,started='original',command='python upstream_source.py',state='R')
        self.assertTrue(e.same_role(role,dict(role,state='S')))
        for field,value in [('pid',12),('started','recycled'),('command','other'),('pgid',11)]:
            self.assertFalse(e.same_role(role,dict(role,**{field:value})))


    def bootstrap_case(self,valid):
        with self.fixture() as (root,control):
            plan=self.plan(root);helper=root/'upstream_fixture_owned_worker.py';marker=root/'authorized.json'
            directory=str(Path(e.__file__).parent)
            helper.write_text("import sys,json,os\nsys.path.insert(0,"+repr(directory)+")\nfrom pathlib import Path\nimport upstream_full_context_catboost_bag_execution_v1 as e\ne.ROOT=Path("+repr(str(root))+")\ne.REPORT=Path("+repr(str(control))+")\ne.WORKER=Path(__file__)\ne.authorize('infer')\nPath("+repr(str(marker))+").write_text(json.dumps(dict(pid=os.getpid(),torch_imported='torch' in sys.modules)))\n")
            plan['source_sha256'][helper.name]=e.study.sha(helper)
            self.write(control/'plan.json',plan);owner=e.identity.process(os.getpid());self.write(control/'monitor.json',owner)
            (control/'first-scientific-baseline.json').write_bytes(e.study.BASELINE.read_bytes())
            self.write(control/'prerequisites.json',{'source_sha256':{}})
            body="import sys\nsys.path.insert(0,"+repr(directory)+")\nfrom pathlib import Path\nimport upstream_full_context_catboost_bag_execution_v1 as e\ne.ROOT=Path("+repr(str(root))+")\ne.REPORT=Path("+repr(str(control))+")\ne.WORKER=Path("+repr(str(helper))+")\ne.bootstrap('infer',"+str(owner['pid'])+")\n"
            child=subprocess.Popen([sys.executable,'-c',body],env=dict(os.environ,**e.ENV),start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+15;path=control/'infer-ownership.json'
                while not path.exists() and time.monotonic()<deadline:
                    if child.poll() is not None:
                        output,errors=child.communicate();self.fail((output+errors).decode())
                    time.sleep(.05)
                self.assertTrue(path.exists());captured=e.read(path);self.assertEqual(captured['pid'],child.pid)
                self.assertFalse(marker.exists());self.assertIsNone(child.poll())
                with patch.object(e,'WORKER',helper):
                    gate=dict(pid=child.pid,phase='infer',command=e.command('infer'),
                        execution_plan_sha256=e.study.sha(control/'plan.json') if valid else 'b'*64,
                        owner_sha256=e.study.sha(control/'monitor.json'),baseline_sha256=e.study.BASELINE_SHA,
                        watchdog=owner,predecessor_sha256={})
                self.write(control/'infer-gate.json',gate)
                output,errors=child.communicate(timeout=15)
                if valid:
                    self.assertEqual(child.returncode,0,(output+errors).decode())
                    actual=e.read(marker);self.assertEqual(actual['pid'],child.pid);self.assertFalse(actual['torch_imported'])
                else:
                    self.assertNotEqual(child.returncode,0);self.assertFalse(marker.exists())
                    self.assertIn('phase gate required',errors.decode())
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=10)
                child.stdout.close();child.stderr.close()


    def test_actual_root_waits_for_matching_gate_then_executes_with_same_PID(self):
        self.bootstrap_case(True)


    def test_actual_wrong_source_gate_cannot_dispatch_root(self):
        self.bootstrap_case(False)


    def test_observed_peaks_keep_both_domains_and_never_invent_unsampled_peak(self):
        peaks={}
        first=dict(family_usage=dict(bytes=100,rss_bytes=200),workers={'1':dict(bytes=100,rss_bytes=200),'2':dict(bytes=300,rss_bytes=200)})
        second=dict(family_usage=dict(bytes=80,rss_bytes=210),workers={'1':dict(bytes=80,rss_bytes=210),'2':dict(bytes=200,rss_bytes=250)})
        e.observe_peaks(peaks,first);e.observe_peaks(peaks,second)
        self.assertEqual(peaks,dict(family_physical_bytes=100,family_RSS_bytes=210,aggregate_physical_bytes=400,aggregate_RSS_bytes=460,samples=2))
        with self.assertRaises(ValueError):e.observe_peaks({},dict(first,family_usage=dict(bytes=-1,rss_bytes=1)))


    def test_source_release_requires_registered_role_and_complete_request_before_return(self):
        for good in (True,False):
            with self.subTest(good=good),self.fixture() as (root,control):
                captured=dict(pid=10,ppid=1,pgid=10,state='S',started='root',command='root-worker')
                role=dict(pid=os.getpid(),ppid=10,pgid=10,state='S',started='source',command=' '.join(e.producer_command('infer',10)))
                self.write(control/'infer-ownership.json',captured)
                self.write(control/'infer-roles.json',dict(scientific_root_pid=10,roles={'producer':role}))
                request=dict(producer=role,scientific_root=captured)
                self.write(control/'infer-source-authorization.json',dict(state='authorized_original_full_context_tree_query_producer_role' if good else 'foreign',request=request))
                with patch.object(e,'phase_check'),patch.object(e.identity,'process',side_effect=lambda pid:role if pid==os.getpid() else captured), \
                        patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'G'))),patch.object(e,'validate_source_request') as validate:
                    if good:
                        self.assertEqual(e.source_bootstrap('infer',10),captured)
                        validate.assert_called_once_with('infer',captured,request,e.metadata_guard)
                    else:
                        with self.assertRaisesRegex(ValueError,'request authorization'):e.source_bootstrap('infer',10)
                        validate.assert_not_called()
                self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
