from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_cfnade_canonical as c


class CanonicalChecks(unittest.TestCase):
    def test_canonical_waits_for_verified_voting_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'experiments/upstream-voting-blend/status.json'
            c.u.write(p,dict(state='training'))
            with patch.object(c.u,'ROOT',root),patch.object(c,'BASE',root),patch.object(c,'other_rss',return_value=0):
                self.assertFalse(c.memory_ready())
                c.u.write(p,dict(state='complete'))
                with patch.object(c.os,'statvfs') as disk:
                    disk.return_value.f_bavail=40;disk.return_value.f_frsize=2**30
                    self.assertTrue(c.memory_ready())
                    with patch.object(c,'other_rss',return_value=17*2**30):self.assertFalse(c.memory_ready())

    def test_resource_groups_are_counted_once_and_completed_jobs_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c.u.write(root/'experiments/full-upstream-reproduction/status.json',dict(state='training',child_pid=12))
            c.u.write(root/'experiments/full-upstream-reproduction/mapped-training/tsvdx5-150o.status.json',dict(state='training',pid=12))
            c.u.write(root/'experiments/full-upstream-reproduction/qualifying-backfill/status.json',dict(state='complete',pid=13))
            with patch.object(c.u,'ROOT',root),patch.object(c,'group_rss',return_value=2048) as rss:
                self.assertEqual(c.other_rss(),2048);rss.assert_called_once_with(12)


if __name__=='__main__':unittest.main()
