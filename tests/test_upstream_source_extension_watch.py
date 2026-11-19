"""An idle source watcher must not monopolize the shared fit/preparation lock."""
import fcntl
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_source_extension_watch as m


class SourceWatcherLockTest(unittest.TestCase):
    def fixture(self,root):
        (root/'source').mkdir();(root/'parent').mkdir()
        (root/'parent/independent-verification.json').write_text('{}')
        return patch.object(m,'REPORT',root/'source'),patch.object(m.experiment.parent,'REPORT',root/'parent')

    def test_waiting_watcher_leaves_shared_lock_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,parent=self.fixture(root)
            with report,parent,patch.object(m.experiment,'inventory',return_value={'aex-128':False}),patch.object(m.experiment,'prepare') as prepare:
                self.assertFalse(m.step());prepare.assert_not_called()
                with (root/'source/run.lock').open('a') as other:
                    fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertIs(m.u.read(root/'source/status.json')['fitting_lock_held'],False)

    def test_ready_watcher_cannot_prepare_during_another_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,parent=self.fixture(root)
            with report,parent,patch.object(m.experiment,'inventory',return_value={'aex-128':True}),patch.object(m.experiment,'prepare') as prepare:
                with (root/'source/run.lock').open('a') as other:
                    fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    self.assertFalse(m.step());prepare.assert_not_called()
                    self.assertEqual(m.u.read(root/'source/status.json')['state'],'waiting_for_prepare_lock')

    def test_actual_preparation_holds_shared_lock_then_releases_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,parent=self.fixture(root)
            def prepare():
                with (root/'source/run.lock').open('a') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with report,parent,patch.object(m.experiment,'inventory',return_value={'aex-128':True}),patch.object(m.experiment,'prepare',side_effect=prepare) as called:
                self.assertTrue(m.step());called.assert_called_once()
                with (root/'source/run.lock').open('a') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)


if __name__=='__main__':unittest.main()
