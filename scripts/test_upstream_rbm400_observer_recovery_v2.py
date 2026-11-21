"""Recovery boundaries and complete source equivalence; no production signals."""
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import upstream_rbm400_observer_recovery_v2 as recovery
import upstream_rbm400_metadata_retirement_v2 as retirement
import upstream_factored_rbm_execution_native_v2 as successor


def functions(text):
    return {n.name: ast.dump(n) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


class RecoveryTests(unittest.TestCase):
    def test_worker_only_changes_execution_import(self):
        recovery.equivalent_worker()

    def test_original_numerical_and_resource_functions_unchanged(self):
        old = functions(Path(recovery.original.__file__).read_text())
        new = functions(Path(successor.__file__).read_text())
        for name in old.keys()-{'check_sources','frozen_plan','main','command'}:
            self.assertEqual(old[name],new[name],name)
        self.assertEqual(successor.PHASES,recovery.original.PHASES)
        self.assertEqual(successor.ENV,recovery.original.ENV)
        self.assertEqual(successor.RESULTS,recovery.original.RESULTS)
        self.assertEqual(successor.TOTAL_LIMIT,recovery.original.TOTAL_LIMIT)
        self.assertEqual(successor.FLOOR,recovery.original.FLOOR)
        for phase in successor.PHASES:
            self.assertEqual(successor.cap(phase),recovery.original.cap(phase))
            expected=recovery.original.command(phase)
            expected[2]=expected[2].replace('_worker.py','_worker_native_v2.py')
            self.assertEqual(successor.command(phase),expected)

    def test_consumer_all_scientific_functions_equivalent(self):
        for stem in ('upstream_local_rbm_blend','verify_upstream_local_rbm_blend',
                     'upstream_local_rbm_blend_queue'):
            old=(recovery.ROOT/'scripts'/f'{stem}.py').read_text()
            new=(recovery.ROOT/'scripts'/f'{stem}_native_v2.py').read_text()
            # Normalize only the explicitly declared provenance/namespace map.
            for a,b in sorted(recovery.REPLACEMENTS.items(),key=lambda pair:-len(pair[1])):
                new=new.replace(b,a)
            new=new.replace('-native-observer-v2','').replace('OWN_LOCAL_RBM_NATIVE_OBSERVER_V2_BLEND_PLAN.md','OWN_LOCAL_RBM_BLEND_PLAN.md')
            a,b=functions(old),functions(new)
            for name in a.keys()-({'main'} if stem.endswith('_queue') else set()):
                self.assertEqual(a[name],b[name],stem+':'+name)

    def test_original_runtime_and_source_still_validate(self):
        plan=recovery.original_plan()
        self.assertEqual(plan['configuration'],successor.asdict(successor.Config()))

    def test_no_fresh_swap_baseline_call_in_successor_mains(self):
        for name in ('upstream_factored_rbm_execution_native_v2','upstream_local_rbm_blend_queue_native_v2'):
            text=(recovery.ROOT/'scripts'/f'{name}.py').read_text()
            main=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            for node in ast.walk(main):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='baseline' for t in node.targets):
                    self.assertNotIn('memory.memory',ast.unparse(node.value))

    def test_copy_original_baseline_preserves_exact_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'old';source.mkdir()
            payload=b'{ "swap_used_bytes": 123.5, "compressor_bytes": 42 }\n'
            (source/'first-scientific-baseline.json').write_bytes(payload)
            target=Path(td)/'new.json'
            with patch.object(recovery,'original_plan'),patch.object(recovery.original,'REPORT',source),patch.object(recovery,'ROOT',Path(td)):
                recovery.copy_baseline(target)
            self.assertEqual(target.read_bytes(),payload)

    def test_changed_caps_and_omitted_original_source_fail(self):
        old=dict(policy={'cap':24},source_sha256={'a':'hash'})
        p=dict(policy=old['policy'],source_sha256=old['source_sha256'],recovery={'baseline_reset':False})
        with patch.object(recovery,'original_plan',return_value=old),patch.object(recovery,'previous_plan',return_value=old),patch.object(recovery,'equivalent_worker'),patch.object(recovery,'retirement'),patch.object(recovery,'recovery_fields',return_value=p['recovery']):
            recovery.check_declaration(p)
            for change in (dict(policy={'cap':25}),dict(source_sha256={}),dict(recovery={'baseline_reset':True})):
                with self.assertRaises(ValueError):recovery.check_declaration(dict(p,**change))

    def test_exact_identity_checks_pid_start_command_group_parent(self):
        expected=dict(pid=1,started='original',command='worker',pgid=1,ppid=2)
        with patch.object(retirement.old.identity,'process',return_value=expected):
            retirement.exact(expected)
        for key,value in [('pid',3),('started','other'),('command','other'),('pgid',4),('ppid',5)]:
            with patch.object(retirement.old.identity,'process',return_value=dict(expected,**{key:value})):
                with self.assertRaises(ValueError):retirement.exact(expected)

    def test_metadata_only_rejects_artifacts_advanced_data_or_consumer_admission(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            root=Path(td);study=root/'study';study.mkdir();art=root/'art';art.mkdir()
            report=root/'producer';report.mkdir();consumer=root/'consumer';consumer.mkdir()
            (study/'data-started.json').write_text(json.dumps(dict(phase='data',execution_plan_sha256=recovery.PREVIOUS_PLAN_SHA)))
            for name in ('data-ownership.json','child-ownership.json'):(report/name).write_text('{}')
            (report/'first-scientific-baseline.json').write_text('{}')
            (consumer/'plan.json').write_text('{}')
            (consumer/'first-scientific-baseline.json').write_text('{}')
            (consumer/'status.json').write_text(json.dumps(dict(state='waiting_full_actual_RBM_and_original_own15_verification')))
            stack.enter_context(patch.object(recovery,'original_plan'))
            stack.enter_context(patch.object(recovery,'previous_plan'))
            stack.enter_context(patch.object(retirement.consumer,'check_sources'))
            for obj,name,value in ((retirement.old,'STUDY',study),(retirement.old,'ART',art),(retirement.old,'REPORT',report),(retirement.consumer,'REPORT',consumer),(retirement.consumer.study,'REPORT',root/'absent-study'),(retirement.consumer.study,'ART',root/'absent-art'),(recovery.original,'REPORT',report)):
                stack.enter_context(patch.object(obj,name,value))
            self.assertFalse(retirement.metadata_only()['learned_state_existed'])
            for path in (art/'known.pt',study/'data-admission.json',consumer/'prerequisites.json'):
                path.write_text('{}')
                with self.assertRaises(ValueError):retirement.metadata_only()
                path.unlink()

    def test_imports_do_not_load_torch_or_start_work(self):
        code="import sys; import upstream_factored_rbm_worker_native_v2; import upstream_local_rbm_blend_queue_native_v2; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=recovery.ROOT/'scripts',check=True)

    def test_unreviewed_retirement_never_signals(self):
        with patch.object(retirement.old,'read',side_effect=[{},dict(state='not-reviewed',findings=[]),{}]),patch.object(retirement.os,'kill') as kill,patch.object(retirement.os,'killpg') as killpg:
            with self.assertRaises(ValueError):retirement.execute()
            kill.assert_not_called();killpg.assert_not_called()

    def test_changed_metadata_after_freeze_never_kills_worker(self):
        import signal
        processes={key:dict(pid=n,ppid=99,pgid=n,started='same',command='original',worker_command='worker')
                   for n,key in enumerate(('owner','worker','watchdog','consumer_owner','consumer_watchdog','observer_reaper'),1)}
        metadata=dict(data_start_sha256='start',original_plan_sha256='plan',original_baseline_sha256='baseline',learned_state_existed=False)
        plan=dict(processes=processes,**metadata)
        review=dict(state='complete_independent_native_observer_successor_review',findings=[],protocol_sha256='bound')
        with patch.object(retirement.old,'read',side_effect=[plan,review,dict(source_sha256={})]),patch.object(retirement.old.inputs,'sha',return_value='bound'),patch.object(retirement,'capture',return_value=dict(processes=processes)),patch.object(retirement,'metadata_only',side_effect=[metadata,dict(metadata,learned_state_existed=True)]),patch.object(retirement,'journal'),patch.object(retirement,'exact'),patch.object(retirement,'wait_state'),patch.object(retirement.os,'kill') as kill,patch.object(retirement.os,'killpg') as killpg:
            with self.assertRaises(ValueError):retirement.execute()
            self.assertEqual(kill.call_args_list[0].args,(6,signal.SIGTERM))
            self.assertEqual(kill.call_args_list[1].args,(1,signal.SIGSTOP))
            self.assertEqual(killpg.call_args_list[0].args,(2,signal.SIGSTOP))
            self.assertEqual(killpg.call_count,1)


if __name__=='__main__':unittest.main()
