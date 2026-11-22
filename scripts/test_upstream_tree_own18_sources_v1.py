"""Own18 provider fixtures; upstream owner/numerical admission is mocked."""
from contextlib import contextmanager,ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_tree_own18_sources_v1 as module


class SourcesTests(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
        return module.tree_owner.study.sha(path)

    @contextmanager
    def fixture(self):
        e=module.tree_owner
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();control=root/'owner';results=root/'results';source=root/'source';previous=root/'previous'
            run=root/'tree-art';run.mkdir();code=root/'fixture-sources.py';code.write_text('# synthetic component binding')
            for obj,key,value in [(module,'ROOT',root),(module,'COUNTS',{'known':20,'qualifying':23}),(module,'__file__',str(code)),
                    (e,'ROOT',root),(e,'REPORT',control),(e,'RESULTS',results),(e,'SOURCE_REPORT',source),(module.previous,'REPORT',previous)]:
                stack.enter_context(patch.object(obj,key,value))
            rows=[dict(id=ident,family='synthetic-original-family',known_prediction='base.f32',qualifying_prediction='base.f32') for ident in module.BASE_IDS]
            base=root/'base.f32';base.write_bytes(np.full(20,3.,dtype='<f4').tobytes())
            calibration=dict(family_ids=list(module.BASE_IDS),families=rows,source_sha256={'base.f32':e.study.sha(base)})
            stack.enter_context(patch.object(module.previous_replay,'require_preaudit',return_value=({},calibration,{},{},{})))
            stack.enter_context(patch.object(module.previous,'verify_bindings',side_effect=lambda b,g:e.consumer.verify_files(root,b,g)))
            stack.enter_context(patch.object(e,'check_sources'));validate=stack.enter_context(patch.object(e,'validate_result'))
            stack.enter_context(patch.object(e.family,'live_roles',return_value={}))
            stack.enter_context(patch.object(e,'roles',return_value={}))
            stack.enter_context(patch.object(e.bag,'paths',return_value=(run,root/'tree-report')))
            outputs={}
            for query,count,filename in [('known',20,'known-prediction.f32'),('qualifying',23,'prediction.f32')]:
                path=run/filename;path.write_bytes(np.full(count,3.,dtype='<f4').tobytes())
                outputs[query]=dict(path=str(path),bytes=count*4,sha256=e.study.sha(path))
            admitted=dict(state=e.ADMITTED,selection_sha256='a'*64,arm='scalar',known_rows=20,qualifying_rows=23,fold_slots=5,seed_slots=15,
                models_fitted=0,qualifying_labels_read=False,production_data_admitted=False,
                every_original_raw_ridge_model_fold_bag_and_matched_known_bit_independently_verified=True,
                both_actual_owned_source_exit_and_authorization_closures_verified=True,outputs=outputs,source_sha256={})
            digest=self.write(source/'pair-admission.json',admitted)
            self.write(control/'plan.json',dict(source_sha256={code.name:e.study.sha(code)}))
            completed={}
            for phase in e.PHASES:
                path=results/(phase+'.json');completed[str(path.relative_to(root))]=self.write(path,{'synthetic_scientific_wrapper':True})
                path=control/(phase+'-resource-peaks.json')
                completed[str(path.relative_to(root))]=self.write(path,dict(state='complete_observed_native_original_family_resource_samples',phase=phase,
                    continuous_peak_claimed=False,original_baseline_sha256=e.study.BASELINE_SHA))
                self.write(control/(phase+'-ownership.json'),dict(pid=100+len(completed)))
            done=dict(state=e.COMPLETE,verification_sha256=digest,production_pair_admitted=True,models_fitted=0,known_rows=20,qualifying_rows=23,
                qualifying_labels_read=False,selection_sha256='a'*64,selected_arm='scalar',completed_phase_and_observed_resource_proof_sha256=completed)
            self.write(control/'completion.json',done)
            self.write(control/'owner-release.json',dict(state='all_original_scientific_roles_terminal_after_complete_selected_pair_replay'))
            for name in ('plan.json','calibration.json','known-completion.json','frozen.json','independent-preaudit-verification.json'):
                self.write(previous/name,{'synthetic_original17_preaudit_closure':True})
            yield dict(root=root,control=control,source=source,calibration=calibration,admitted=admitted,done=done,validate=validate)

    def test_every_original_family_retained_and_control_matches_the_appended_pair(self):
        with self.fixture() as f:
            rows,bindings,controls=module.declaration(lambda **kw:None)
            self.assertEqual([r['id'] for r in rows],module.BASE_IDS+[module.FAMILY_ID]);self.assertEqual(len(rows),18)
            self.assertEqual(rows[:-1],f['calibration']['families'])
            for query in ('known','qualifying'):
                self.assertEqual(rows[-1][query+'_prediction'],str(Path(controls[query]['path']).relative_to(f['root'])))
                self.assertEqual(controls[query]['format'],'raw')
            self.assertEqual(f['validate'].call_count,4)
            self.assertIn('owner/completion.json',bindings)

    def test_component_receipt_without_completed_owned_admission_cannot_supply_a_family(self):
        with self.fixture() as f:
            f['done']['production_pair_admitted']=False;self.write(f['control']/'completion.json',f['done'])
            with self.assertRaisesRegex(ValueError,'Complete owned full tree pair'):module.declaration(lambda **kw:None)

    def test_missing_phase_or_live_source_is_rejected(self):
        with self.fixture() as f:
            f['done']['completed_phase_and_observed_resource_proof_sha256'].pop('results/replay.json')
            self.write(f['control']/'completion.json',f['done'])
            with self.assertRaisesRegex(ValueError,'All four actual pair'):module.declaration(lambda **kw:None)
        with self.fixture(),patch.object(module.tree_owner.family,'live_roles',return_value={'producer':{'pid':10}}):
            with self.assertRaisesRegex(ValueError,'still-live'):module.declaration(lambda **kw:None)

    def test_dropped_or_reordered_original_family_and_changed_actual_vector_rejected(self):
        with self.fixture() as f:
            f['calibration']['families'].reverse()
            with self.assertRaisesRegex(ValueError,'original order'):module.declaration(lambda **kw:None)
        with self.fixture() as f:
            Path(f['admitted']['outputs']['qualifying']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):module.declaration(lambda **kw:None)


if __name__=='__main__':unittest.main()
