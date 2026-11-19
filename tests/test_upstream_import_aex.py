"""Verify serialization and actual scheduler identity after the primary handoff."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_import_aex as m


class AexImportTest(unittest.TestCase):
    def test_both_completed_predecessors_are_required_and_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'experiments/upstream-import-backfill';old.mkdir(parents=True)
            handoff=root/'handoff';handoff.mkdir();primary=root/'primary';primary.mkdir()
            with patch.object(m.u,'ROOT',root),patch.object(m.handoff,'REPORT',handoff),patch.object(m.transfer,'PRIMARY',primary):
                self.assertIsNone(m.prior_transition())
                m.u.write(old/'result.json',dict(state='complete'))
                m.u.write(old/'status.json',dict(state='complete'))
                self.assertIsNone(m.prior_transition())
                m.u.write(primary/'launch.json',dict(pid=123))
                result=dict(state='successor_launched',launch=dict(pid=123),auxiliary_import_sha256=m.u.sha(old/'result.json'))
                m.u.write(handoff/'result.json',result)
                self.assertEqual(m.prior_transition(),result)
                m.u.write(old/'result.json',dict(state='changed'))
                with self.assertRaisesRegex(ValueError,'identity'):m.prior_transition()

    def test_stale_successor_pid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'experiments/upstream-import-backfill';old.mkdir(parents=True)
            handoff=root/'handoff';handoff.mkdir();primary=root/'primary';primary.mkdir()
            m.u.write(old/'result.json',dict(state='complete'));m.u.write(old/'status.json',dict(state='complete'))
            m.u.write(primary/'launch.json',dict(pid=999))
            m.u.write(handoff/'result.json',dict(state='successor_launched',launch=dict(pid=123),auxiliary_import_sha256=m.u.sha(old/'result.json')))
            with patch.object(m.u,'ROOT',root),patch.object(m.handoff,'REPORT',handoff),patch.object(m.transfer,'PRIMARY',primary):
                with self.assertRaisesRegex(ValueError,'identity'):m.prior_transition()

    def test_actual_successor_role_is_accepted_and_predecessor_never_paused(self):
        children=[]
        try:
            for marker in ['scripts/upstream_full_reproduce.py','scripts/upstream_primary_guarded.py']:
                children.append(subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',marker],start_new_session=True))
            saved={k:getattr(m.transfer,k) for k in ['REPORT','auxiliary','followon','PRIMARY_FRAGMENT']}
            with patch.multiple(m.transfer,**saved):
                m.configure()
                with self.assertRaisesRegex(ValueError,'changed'):
                    with m.transfer.paused_primary(children[0].pid):self.fail('Predecessor must be rejected')
                with m.transfer.paused_primary(children[1].pid):
                    row=next(r for r in m.transfer.process_rows() if r['pid']==children[1].pid)
                    self.assertIn('T',row['state'])
                for child in children:
                    row=next(r for r in m.transfer.process_rows() if r['pid']==child.pid)
                    self.assertNotIn('T',row['state']);self.assertIsNone(child.poll())
        finally:
            for child in children:
                if child.poll() is None:
                    os.kill(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
