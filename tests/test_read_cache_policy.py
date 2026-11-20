import fcntl
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_read_cache_policy as policy


class CachePolicyRoundtrip(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.report=self.root/'report';self.report.mkdir();self.file=self.root/'features.npy';self.file.write_bytes(bytes(range(256))*256)
        self.before=policy.identity(self.file);self.digest=policy.u.sha(self.file)
        self.row=dict(path=str(self.file),expected_sha256=self.digest,identity=self.before)
        self.patches=[patch.object(policy,'LOCK',self.root/'lease.lock'),patch.object(policy,'OWNER',self.root/'owner.json'),
            patch.object(policy,'candidates',return_value=[self.row]),
            patch.object(policy.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr=''))]
        for p in self.patches:p.start()
        self.actor=policy.Policy(self.report,dict(pid=123,started='private fixture',command='fixture'))

    def tearDown(self):
        self.actor.restore()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def assert_restored(self):
        with self.file.open('rb') as reader:self.assertEqual(fcntl.fcntl(reader.fileno(),55,0),0)
        self.assertEqual(policy.identity(self.file),self.before);self.assertEqual(policy.u.sha(self.file),self.digest)

    def test_complete_lease_is_idempotently_restored_with_exact_bytes(self):
        self.actor.apply()
        with self.file.open('rb') as reader:self.assertTrue(fcntl.fcntl(reader.fileno(),55,1))
        self.actor.restore();original=(self.report/'cache-policy-restoration.json').read_bytes()
        self.actor.restore();self.assertEqual((self.report/'cache-policy-restoration.json').read_bytes(),original)
        self.assert_restored()

    def test_exception_after_policy_change_restores_it(self):
        real=policy.write
        def failing(path,value):
            if path.name=='cache-policy.json':raise RuntimeError('injected journal failure')
            return real(path,value)
        with patch.object(policy,'write',side_effect=failing),self.assertRaisesRegex(RuntimeError,'injected'):
            self.actor.apply()
        self.assert_restored()

    def test_watchdog_restores_journal_after_owner_handles_are_lost(self):
        self.actor.apply()
        for fd in self.actor.handles:policy.os.close(fd)
        self.actor.handles=[];self.actor.lock.close();self.actor.lock=None
        policy.restore_after_owner_loss(self.report);self.assert_restored()

    def test_writable_descriptor_prevents_policy_application(self):
        with patch.object(policy.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='aw\n',stderr='')):
            with self.assertRaisesRegex(ValueError,'writable'):self.actor.apply()
        self.assertFalse(self.actor.records);self.assert_restored()

    def test_preexisting_uncached_state_is_preserved(self):
        with self.file.open('rb') as reader:
            fcntl.fcntl(reader.fileno(),55,1)
            try:
                self.actor.apply();self.assertTrue(self.actor.records[0]['prior_global_state'])
                self.actor.restore();self.assertTrue(fcntl.fcntl(reader.fileno(),55,1))
            finally:fcntl.fcntl(reader.fileno(),55,0)


if __name__=='__main__':unittest.main()
