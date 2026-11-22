"""Owned-controller metadata, conditions and dispatch; no production admission."""
from contextlib import contextmanager,ExitStack
import copy
import io
from types import SimpleNamespace
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import upstream_full_context_catboost_execution_v1 as e


class ExecutionTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();control=root/'control';control.mkdir()
            for module,name,value in [(e,'ROOT',root),(e,'REPORT',control),(e,'RESULTS',root/'results'),
                    (e,'PREDECESSOR',root/'previous-owner'),(e,'PREDECESSOR_STUDY',root/'previous-study')]:
                stack.enter_context(patch.object(module,name,value))
            yield root,control

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return e.study.sha(path)

    def plan(self,root):
        source=root/'fixture-source.py';source.write_text('# original synthetic source\n')
        return dict(state=e.PLAN_STATE,phases=list(e.PHASES),environment=e.ENV,policy=e.policy(),libraries=e.study.versions(),python=sys.version,
            parameters=e.study.backend.PARAMETERS,max_rounds=2000,patience=100,seeds=[2026,7,13],folds=5,known_rows=1128164,
            qualifying_rows=2817131,tie_rule='scalar',tie_tolerance=1e-12,selection_uses_only_known_CV=True,
            source_sha256={'fixture-source.py':e.study.sha(source)})

    def predecessor(self,root,scores=(.86,.861)):
        dependency=root/'dependency.json';self.write(dependency,{'synthetic_completed_dependency':True})
        proof=dict(state='complete_independent_selected_RBM400_own17_full_source_models_coefficients_and_frozen_group_scores',primary=e.PRIMARY,
            simplex_supports_examined=131071,ridge_equations=18,all_six_vectors_independently_verified=True,
            full_inherited_source_closure_rehashed=True,qualifying_labels_used_for_fitting_or_selection=False,official_or_untouched_test=False,
            groups={k:dict(rows=n,rmse=s) for (k,n),s in zip(e.GROUP_COUNTS.items(),scores)},
            both_groups_below_target=all(s<.85725 for s in scores),bindings={'dependency.json':e.study.sha(dependency)})
        path=e.PREDECESSOR_STUDY/'independent-canonical-verification.json';digest=self.write(path,proof)
        self.write(e.PREDECESSOR/'completion.json',dict(state='complete_own17_known_only_calibration_full_preaudit_and_separate_fixed_recovered_canonical_audit',
            verification_sha256=digest,qualifying_labels_used_for_fitting_or_selection=False))
        return path,proof

    def test_all_nine_phases_original_budgets_and_complete_output_inventory(self):
        self.assertEqual(len(e.PHASES),9);self.assertEqual(e.PHASES[-1],'select')
        self.assertEqual(e.PHASES[:4],('prepare_scalar','fit_scalar','replay_scalar','canonical_scalar'))
        self.assertEqual(e.FAMILY_LIMIT,16*2**30);self.assertEqual(e.policy()['maximum_swap_growth_bytes'],128*2**20)
        self.assertGreaterEqual(e.OUTPUT_RESERVE,(30*64+128+512+64+64)*2**20)
        for phase in e.PHASES:self.assertEqual(e.command(phase)[-1],phase)
        with self.assertRaises(ValueError):e.command('partial')
        with self.assertRaises(ValueError):e.producer_command('prepare_scalar',10)
        self.assertNotIn('torch',sys.modules)

    def test_frozen_source_policy_and_runtime_cannot_be_rebased(self):
        with self.fixture() as (root,control):
            plan=self.plan(root);e.check_sources(plan)
            for field,value in [('max_rounds',1999),('patience',99),('known_rows',20),('tie_rule','latent')]:
                bad=copy.deepcopy(plan);bad[field]=value
                with self.assertRaises(ValueError):e.check_sources(bad)
            bad=copy.deepcopy(plan);bad['policy']['family_bytes']=17*2**30
            with self.assertRaises(ValueError):e.check_sources(bad)
            (root/'fixture-source.py').write_text('# newer bytes must not become the old expected source')
            with self.assertRaisesRegex(ValueError,'source changed'):e.check_sources(plan)

    def test_primary_only_strict_both_group_condition_and_completed_dependencies(self):
        with self.fixture() as (root,control):
            for scores,start in [((.86,.861),True),((.85,.85),False),((.85,.85725),True)]:
                self.predecessor(root,scores);result=e.prerequisites()
                self.assertEqual(result['start_followup'],start)
                self.assertFalse(result['qualifying_outcomes_read']);self.assertFalse(result['score_values_used_for_parameters_or_selection'])
            (root/'dependency.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'dependency changed'):e.prerequisites()

    def test_partial_or_comparison_candidate_cannot_start_followup(self):
        with self.fixture() as (root,control):
            path,proof=self.predecessor(root)
            for field,value in [('primary','rbm400-own17-ridge10'),('simplex_supports_examined',15),('all_six_vectors_independently_verified',False),
                    ('qualifying_labels_used_for_fitting_or_selection',True),('both_groups_below_target',True)]:
                bad=copy.deepcopy(proof);bad[field]=value;digest=self.write(path,bad)
                done=e.read(e.PREDECESSOR/'completion.json');done['verification_sha256']=digest;self.write(e.PREDECESSOR/'completion.json',done)
                with self.assertRaises(ValueError):e.prerequisites()

    def test_early_and_registered_role_tables_cannot_replace_original_lifetime(self):
        with self.fixture() as (root,control):
            captured={'pid':10};role={'pid':11,'started':'original'}
            self.write(control/'fit_scalar-source-bootstrap.json',dict(scientific_root_pid=10,roles={'producer':role}))
            self.write(control/'fit_scalar-roles.json',dict(scientific_root_pid=10,roles={'producer':role}))
            self.assertEqual(e.roles('fit_scalar',captured),{'producer':role})
            self.write(control/'fit_scalar-roles.json',dict(scientific_root_pid=10,roles={'producer':dict(role,pid=12)}))
            with self.assertRaises(ValueError):e.roles('fit_scalar',captured)

    def test_live_source_state_may_change_but_lifetime_or_command_may_not(self):
        role=dict(pid=11,ppid=10,pgid=10,started='original',command='python upstream_source.py',state='R')
        self.assertTrue(e.same_role(role,dict(role,state='S')))
        for field,value in [('pid',12),('started','recycled'),('command','other'),('pgid',11)]:
            self.assertFalse(e.same_role(role,dict(role,**{field:value})))

    def test_actual_role_request_must_bind_complete_original_source_closure(self):
        with self.fixture() as (root,control),ExitStack() as stack:
            arm=root/'arm';arm.mkdir();run=root/'art';run.mkdir()
            worker=root/'upstream_fixture_worker.py';worker.write_text('# source worker fixture')
            code=root/'upstream_fixture_pipeline.py';code.write_text('# pipeline fixture')
            metadata={'source_sha256':{'upstream_fixture_pipeline.py':e.study.sha(code)}}
            expected={'admission_sha256':'a'*64};manifest=dict(source_metadata_sha256='a'*64,outputs={})
            for name in ('source','contract','weights'):
                path=arm/(name+'.json');self.write(path,{name:True});manifest['outputs'][name]=dict(path=str(path),sha256=e.study.sha(path))
            self.write(arm/'manifest.json',manifest)
            captured=dict(pid=10,ppid=1,pgid=10,state='R',started='root',command='root-worker')
            role=dict(pid=11,ppid=10,pgid=10,state='R',started='source',command='source-worker')
            self.write(control/'fit_scalar-source-bootstrap.json',dict(scientific_root_pid=10,roles={'producer':role}))
            stack.enter_context(patch.object(e,'WORKER',worker));stack.enter_context(patch.object(e,'authorize',return_value=captured))
            stack.enter_context(patch.object(e.study,'paths',return_value=(run,arm)))
            stack.enter_context(patch.object(e.study,'load',return_value=(manifest,metadata,expected)))
            stack.enter_context(patch.object(e.pipeline,'code_paths',return_value=[code]))
            stack.enter_context(patch.object(e.identity,'process',return_value=dict(role,state='S')))
            bindings=dict(metadata['source_sha256'])
            for row in manifest['outputs'].values():bindings[str(Path(row['path']).relative_to(root))]=row['sha256']
            bindings['arm/manifest.json']=e.study.sha(arm/'manifest.json');bindings[worker.name]=e.study.sha(worker)
            request=dict(mode='fit',arm='scalar',command=e.producer_command('fit_scalar',10),scientific_root=captured,producer=role,
                manifest_sha256=e.study.sha(arm/'manifest.json'),source_metadata_sha256='a'*64,
                contract_sha256=hashlib.sha256(e.study.wire.encoded(expected)).hexdigest(),source_sha256=bindings,qualifying_labels_read=False)
            gate=e.admit_source('fit_scalar',captured,request,lambda **kw:None)
            self.assertEqual(e.study.sha(gate['path']),gate['sha256'])
            self.assertEqual(e.read(gate['path'])['request'],request)
            bad=copy.deepcopy(request);bad['source_sha256']['unapproved-qualifying-outcomes']='b'*64
            with self.assertRaisesRegex(ValueError,'complete original producer source closure'):e.admit_source('fit_scalar',captured,bad,lambda **kw:None)

    def test_partial_fit_or_wrong_pipeline_cannot_be_promoted_by_rehashed_receipt(self):
        with self.fixture() as (root,control):
            path=root/'fit-completion.json';phase='fit_scalar'
            actual=dict(state=e.study.FITTED,arm='scalar',qualifying_labels_read=False,all_five_folds_complete=True,original_seed_slots=15)
            self.write(path,actual);self.write(root/'fit-pipeline-completion.json',dict(scientific_proof_sha256=e.study.sha(path)))
            value=dict(state='complete_owned_original_tree_'+phase,execution_plan_sha256='a'*64,scientific_proof_sha256=e.study.sha(path),qualifying_labels_read=False)
            with patch.object(e,'scientific_path',return_value=(path,e.study.FITTED)):
                e.validate_result(phase,value,'a'*64)
                actual['original_seed_slots']=14;self.write(path,actual);value['scientific_proof_sha256']=e.study.sha(path)
                self.write(root/'fit-pipeline-completion.json',dict(scientific_proof_sha256=e.study.sha(path)))
                with self.assertRaisesRegex(ValueError,'fold/seed'):e.validate_result(phase,value,'a'*64)


    def bootstrap_case(self,valid):
        with self.fixture() as (root,control):
            plan=self.plan(root);helper=root/'upstream_fixture_owned_worker.py';marker=root/'authorized.json'
            directory=str(Path(e.__file__).parent)
            helper.write_text("import sys,json,os\nsys.path.insert(0,"+repr(directory)+")\nfrom pathlib import Path\nimport upstream_full_context_catboost_execution_v1 as e\ne.ROOT=Path("+repr(str(root))+")\ne.REPORT=Path("+repr(str(control))+")\ne.WORKER=Path(__file__)\ne.authorize('fit_scalar')\nPath("+repr(str(marker))+").write_text(json.dumps(dict(pid=os.getpid(),torch_imported='torch' in sys.modules)))\n")
            plan['source_sha256'][helper.name]=e.study.sha(helper)
            self.write(control/'plan.json',plan);owner=e.identity.process(os.getpid());self.write(control/'monitor.json',owner)
            (control/'first-scientific-baseline.json').write_bytes(e.study.BASELINE.read_bytes())
            self.write(control/'prerequisites.json',{'source_sha256':{}})
            body="import sys\nsys.path.insert(0,"+repr(directory)+")\nfrom pathlib import Path\nimport upstream_full_context_catboost_execution_v1 as e\ne.ROOT=Path("+repr(str(root))+")\ne.REPORT=Path("+repr(str(control))+")\ne.WORKER=Path("+repr(str(helper))+")\ne.bootstrap('fit_scalar',"+str(owner['pid'])+")\n"
            child=subprocess.Popen([sys.executable,'-c',body],env=dict(os.environ,**e.ENV),start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+15;path=control/'fit_scalar-ownership.json'
                while not path.exists() and time.monotonic()<deadline:
                    if child.poll() is not None:
                        output,errors=child.communicate();self.fail((output+errors).decode())
                    time.sleep(.05)
                self.assertTrue(path.exists());captured=e.read(path);self.assertEqual(captured['pid'],child.pid)
                self.assertFalse(marker.exists());self.assertIsNone(child.poll())
                with patch.object(e,'WORKER',helper):
                    gate=dict(pid=child.pid,phase='fit_scalar',command=e.command('fit_scalar'),
                        execution_plan_sha256=e.study.sha(control/'plan.json') if valid else 'b'*64,
                        owner_sha256=e.study.sha(control/'monitor.json'),baseline_sha256=e.study.BASELINE_SHA,
                        watchdog=owner,predecessor_sha256={})
                self.write(control/'fit_scalar-gate.json',gate)
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
                role=dict(pid=os.getpid(),ppid=10,pgid=10,state='S',started='source',command=' '.join(e.producer_command('fit_scalar',10)))
                self.write(control/'fit_scalar-ownership.json',captured)
                self.write(control/'fit_scalar-roles.json',dict(scientific_root_pid=10,roles={'producer':role}))
                request=dict(producer=role,scientific_root=captured)
                self.write(control/'fit_scalar-source-authorization.json',dict(state='authorized_original_full_context_tree_producer_role' if good else 'foreign',request=request))
                with patch.object(e,'phase_check'),patch.object(e.identity,'process',side_effect=lambda pid:role if pid==os.getpid() else captured), \
                        patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(b'G'))),patch.object(e,'validate_source_request') as validate:
                    if good:
                        self.assertEqual(e.source_bootstrap('fit_scalar',10),captured)
                        validate.assert_called_once_with('fit_scalar',captured,request,e.metadata_guard)
                    else:
                        with self.assertRaisesRegex(ValueError,'request authorization'):e.source_bootstrap('fit_scalar',10)
                        validate.assert_not_called()
                self.assertNotIn('torch',sys.modules)


    def test_worker_dispatches_all_nine_full_phases_and_keeps_root_Torch_free(self):
        import upstream_full_context_catboost_worker_v1 as worker
        with self.fixture() as (root,control),ExitStack() as stack:
            guard=lambda **kw:None;capture={'pid':10};events=[]
            self.write(control/'plan.json',{'synthetic_dispatch_plan_only':True})
            locations={arm:(root/'art'/arm,root/'reports'/arm) for arm in e.study.ARMS}
            for run,report in locations.values():
                run.mkdir(parents=True);report.mkdir(parents=True)
                for name in ('manifest.json','fit-completion.json','independent-verification.json','independent-canonical-verification.json'):
                    self.write(report/name,{'synthetic_dispatch_fixture':True})
            stack.enter_context(patch.object(e.study,'paths',side_effect=lambda arm:locations[arm]))
            stack.enter_context(patch.object(e,'authorize',return_value=capture));stack.enter_context(patch.object(e,'control',return_value=SimpleNamespace(cached_roles={})))
            stack.enter_context(patch.object(e.supervision,'Guard',return_value=guard));validate=stack.enter_context(patch.object(e,'validate_result'))
            stack.enter_context(patch.dict(sys.modules,{'upstream_full_context_catboost_producer_v2':SimpleNamespace(prepare=lambda arm,g:dict(arm=arm))}))
            def prepare(arm,prepared,owner,g):
                self.assertEqual(prepared,{'arm':arm});self.assertEqual(set(owner),{'plan','monitor','baseline','gate','ownership'})
                events.append(('prepare',arm))
            def pipeline(mode,arm,manifest_sha,**kwargs):
                self.assertEqual(manifest_sha,e.study.sha(locations[arm][1]/'manifest.json'))
                self.assertEqual(kwargs['command'],e.producer_command(mode+'_'+arm,10));events.append((mode,arm))
                if mode=='replay':self.assertEqual(kwargs['fit_sha256'],e.study.sha(locations[arm][1]/'fit-completion.json'))
            def canonical(arm,*args):events.append(('canonical',arm))
            def select(hashes,g):
                self.assertEqual(set(hashes),{'scalar','latent'});events.append(('select',None))
                self.write(root/'experiments/upstream-full-context-catboost-selection/selection.json',{'synthetic_selection_dispatch_only':True})
            stack.enter_context(patch.object(e.study,'prepare_arm',side_effect=prepare));stack.enter_context(patch.object(e.pipeline,'run',side_effect=pipeline))
            stack.enter_context(patch.object(e.canonical,'verify',side_effect=canonical));stack.enter_context(patch.object(e.canonical,'select',side_effect=select))
            # Prepare ownership files are exact path/hash bindings even though
            # this dispatch-only fixture stubs the numerical bodies.
            for phase in e.PHASES:
                self.write(control/(phase+'-gate.json'),{'synthetic':True});self.write(control/(phase+'-ownership.json'),{'synthetic':True})
            self.write(control/'monitor.json',{'synthetic':True});(control/'first-scientific-baseline.json').write_bytes(e.study.BASELINE.read_bytes())
            with patch('builtins.print') as printed:
                for phase in e.PHASES:
                    worker.main(phase);value=e.read(e.RESULTS/(phase+'.json'))
                    self.assertFalse(value['qualifying_labels_read']);self.assertEqual(value['state'],'complete_owned_original_tree_'+phase)
                self.assertEqual(printed.call_count,9)
            self.assertEqual(events,[(kind,arm) for arm in ('scalar','latent') for kind in ('prepare','fit','replay','canonical')]+[('select',None)])
            self.assertEqual(validate.call_count,9);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
