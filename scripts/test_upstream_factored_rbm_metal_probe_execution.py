"""Verify the owned probe cannot be mistaken for full production training."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
import upstream_factored_rbm_metal_probe_execution as probe
import upstream_factored_rbm_execution as original
import upstream_factored_rbm_preflight as before
import upstream_factored_rbm_preflight_metal as after


def definitions(path):
    return {n.name:ast.dump(n) for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


class ProbeTests(unittest.TestCase):
    def test_probe_only_allows_actual_device_phase(self):
        self.assertEqual(probe.PHASES,('mps_preflight',))
        with self.assertRaises(ValueError):probe.command('fit_known')
        with self.assertRaises(ValueError):probe.command('data')
        self.assertEqual(probe.cap('mps_preflight'),8*2**30)

    def test_original_resource_and_owner_guards_unchanged(self):
        a,b=definitions(original.__file__),definitions(probe.__file__)
        for name in ('WorkerGuard','safe','ready','continuation_ready','same_child','stop_child','authorize','bootstrap','watchdog'):
            self.assertEqual(a[name],b[name],name)
        self.assertEqual(probe.ENV,original.ENV)
        self.assertEqual(probe.TOTAL_LIMIT,original.TOTAL_LIMIT)
        self.assertEqual(probe.FLOOR,original.FLOOR)

    def test_full_stress_and_tolerances_never_shrunk_or_relaxed(self):
        for name in ('REQUIRED_STRESS','SMALL_ATOL','SMALL_RTOL','STRESS_ATOL','STRESS_RTOL'):
            self.assertEqual(getattr(before,name),getattr(after,name))
        a,b=definitions(before.__file__),definitions(after.__file__)
        for name in ('fixture','stress_fixture','reference_prediction','run'):
            self.assertEqual(a[name],b[name],name)

    def test_metadata_imports_never_load_torch(self):
        code="import sys; import upstream_factored_rbm_metal_probe_worker; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],cwd=probe.ROOT/'scripts',check=True)

    def test_main_cannot_take_new_first_baseline(self):
        tree=ast.parse(Path(probe.__file__).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        for node in ast.walk(main):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='baseline' for t in node.targets):
                self.assertNotIn('memory.memory',ast.unparse(node.value))


if __name__=='__main__':unittest.main()
