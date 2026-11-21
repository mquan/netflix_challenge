"""Source/ownership admission for the full original-budget Metal successor."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import upstream_rbm400_metal_declaration as declaration
import upstream_factored_rbm_execution_metal as execution
import upstream_factored_rbm_execution as original


def definitions(path):
    return {n.name:ast.dump(n) for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


class ProductionTests(unittest.TestCase):
    def test_worker_training_terminal_change_only_declared_imports(self):
        declaration.equivalent_modules()

    def test_original_full_config_phases_runtime_and_limits_unchanged(self):
        self.assertEqual(execution.asdict(execution.Config()),original.asdict(original.Config()))
        self.assertEqual(execution.PHASES,original.PHASES)
        self.assertEqual(execution.ENV,original.ENV)
        self.assertEqual(execution.RESULTS,original.RESULTS)
        for phase in original.PHASES:self.assertEqual(execution.cap(phase),original.cap(phase))
        self.assertEqual(execution.TOTAL_LIMIT,original.TOTAL_LIMIT)
        self.assertEqual(execution.FLOOR,original.FLOOR)

    def test_original_guard_and_supervision_AST_unchanged(self):
        a,b=definitions(original.__file__),definitions(execution.__file__)
        for name in ('WorkerGuard','safe','ready','continuation_ready','authorize','bootstrap','watchdog','stop_child','same_child'):
            self.assertEqual(a[name],b[name],name)

    def test_actual_full_width_device_proof_and_previous_closures_validate(self):
        declaration.validate_history()
        self.assertEqual(declaration.DEVICE_PROOF_SHA,'a8ab47ca345a93213de0dd7c9cdb94d0cdeca2b87fb05d8f241a8a1572c40a24')

    def test_no_new_swap_baseline_in_main(self):
        main=next(n for n in ast.parse(Path(execution.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        for node in ast.walk(main):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='baseline' for t in node.targets):
                self.assertNotIn('memory.memory',ast.unparse(node.value))

    def test_metadata_imports_do_not_load_torch_or_start_a_job(self):
        code="import sys; import upstream_factored_rbm_worker_metal; import upstream_local_rbm_blend_queue_metal; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=declaration.ROOT/'scripts',check=True)

    def test_idle_retirement_without_clean_review_cannot_signal(self):
        with tempfile.TemporaryDirectory() as td,patch.object(declaration,'REPORT',Path(td)),patch.object(declaration,'read',side_effect=[dict(state='not-reviewed',findings=[]),{},{}]),patch.object(declaration.os,'kill') as kill:
            with self.assertRaises(ValueError):declaration.retire_idle()
            kill.assert_not_called()


if __name__=='__main__':unittest.main()
