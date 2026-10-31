"""Exercise the real shell queue with fixture receipts and no training binary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/e1_bayes_control_wait.sh'


class BayesControlQueueTests(unittest.TestCase):
    def fixture(self, root, delta=-0.00001, matched=True):
        for name in ('scripts', 'experiments', 'bin'):
            (root/name).mkdir()
        shutil.copyfile(SOURCE, root/'scripts/e1_bayes_control_wait.sh')
        producer = root/'scripts/e1_bayes_chain.sh'
        producer.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == build/netflix-bayesweights && "$2" == w1 ]]
touch launched
printf '{"returncode":0}\\n' > experiments/E1-B-B4-w1.json
''')
        # Existing real K1 jobs must not affect the isolated fixture.
        pgrep = root/'bin/pgrep'
        pgrep.write_text('#!/usr/bin/env bash\nexit 1\n')
        pgrep.chmod(0o755)
        (root/'comparison.json').write_text(json.dumps(dict(plan_sha256='expected' if matched else 'changed',
            arms={'w2': {'comparison': {'delta_rmse': delta}}, 'w3': {'comparison': {'delta_rmse': 0.0001}}})))
        (root/'plan.json').write_text(json.dumps(dict(comparison='comparison.json',
            scoring_plan_sha256='expected', frozen={})))
        return dict(os.environ, PATH=str(root/'bin')+os.pathsep+os.environ['PATH'])

    def run_queue(self, root, env):
        return subprocess.run(['bash', 'scripts/e1_bayes_control_wait.sh', 'plan.json'],
                              cwd=root, env=env, capture_output=True, text=True, timeout=10)

    def state(self, root):
        return json.loads((root/'experiments/E1-bayes-control-queue-20261030.state.json').read_text())['state']

    def test_some_gain_launches_only_the_declared_control_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root)
            result = self.run_queue(root, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root/'launched').exists())
            self.assertEqual(self.state(root), 'complete')
            self.assertNotEqual(self.run_queue(root, env).returncode, 0)

    def test_no_gain_does_not_launch_or_stop_anything(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root, delta=0)
            self.assertEqual(self.run_queue(root, env).returncode, 0)
            self.assertEqual(self.state(root), 'skipped_no_k1_gain')
            self.assertFalse((root/'launched').exists())

    def test_mismatched_plan_rejects_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root, matched=False)
            self.assertNotEqual(self.run_queue(root, env).returncode, 0)
            self.assertEqual(self.state(root), 'failed')
            self.assertFalse((root/'launched').exists())

    def test_existing_control_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.fixture(root)
            existing = root/'experiments/E1-B-B4-w1.log'
            existing.write_text('existing evidence')
            self.assertEqual(self.run_queue(root, env).returncode, 0)
            self.assertEqual(self.state(root), 'skipped_existing_control')
            self.assertEqual(existing.read_text(), 'existing evidence')
            self.assertFalse((root/'launched').exists())


if __name__ == '__main__':
    unittest.main()
