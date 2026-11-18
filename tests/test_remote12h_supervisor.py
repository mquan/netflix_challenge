import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import remote12h_supervisor as sup


def iso(seconds):
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).isoformat()


class SupervisorTest(unittest.TestCase):
    def run_sup(self, deadline, command, **extra):
        d = Path(tempfile.mkdtemp())
        args = ['--name', 'toy', '--out-dir', str(d), '--deadline', deadline, '--sample-every', '0.5',
                '--grace', '2', '--min-disk-gb', '0']
        for k, v in extra.items():
            args += ['--' + k.replace('_', '-'), str(v)]
        code = sup.main(args + ['--', *command])
        return code, json.loads((d/'status.json').read_text()), d

    def test_completed(self):
        code, status, d = self.run_sup(iso(60), [sys.executable, '-c', 'print(1)'])
        self.assertEqual((code, status['state'], status['exit_code']), (0, 'completed', 0))

    def test_failure_recorded(self):
        code, status, _ = self.run_sup(iso(60), [sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual((status['state'], status['exit_code']), ('failed', 7))

    def test_deadline_kills_whole_tree_including_term_ignoring_grandchild(self):
        # Parent spawns a grandchild that ignores SIGTERM; both must be gone.
        script = ('import subprocess,sys,time;'
                  'subprocess.Popen([sys.executable,"-c","import signal,time;'
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(600)"]);time.sleep(600)')
        t0 = time.time()
        code, status, d = self.run_sup(iso(3), [sys.executable, '-c', script])
        self.assertEqual(status['state'], 'interrupted-deadline')
        self.assertEqual(status['leftover_group_members'], [])
        self.assertEqual(sup.group_members(status['pgid']), [])
        self.assertLess(time.time() - t0, 30)
        self.assertIn('SIGKILL', (d/'supervisor.log').read_text())
        self.assertTrue((d/'resources.jsonl').read_text().strip())

    def test_past_deadline_not_launched(self):
        code, status, _ = self.run_sup(iso(-5), [sys.executable, '-c', 'print(1)'])
        self.assertEqual(status['state'], 'not-launched-deadline-passed')


if __name__ == '__main__':
    unittest.main()
