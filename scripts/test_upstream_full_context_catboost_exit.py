"""Real child lifecycle checks; no model/data/queue admission."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import upstream_full_context_catboost_exit as module


class ExitTests(unittest.TestCase):
    @contextmanager
    def fixture(self, status=0):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'source.py';source.write_text('# synthetic frozen source\n')
            # EOF precedes actual exit. The child remains blocked on stdin,
            # allowing the real parent to prove EOF alone cannot admit a fit.
            script='import os,sys; os.close(1); sys.stdin.buffer.read(1); sys.exit('+str(status)+')'
            child=subprocess.Popen([sys.executable,'-c',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            try:
                with patch.object(module.study,'ROOT',root):
                    captured=module.consumer.identity.process(child.pid)
                    owner=module.consumer.identity.process(os.getpid())
                    assert captured and owner
                    metadata='2'*64;contract=dict(admission_sha256=metadata)
                    record=module.capture(child,owner=owner,group=captured['pgid'],expected_contract=contract,
                        metadata_sha256=metadata,source_sha256={'source.py':module.study.sha(source)},guard=lambda **kw:None)
                    yield root,child,record,source
            finally:
                if child.poll() is None:
                    child.stdin.write(b'x');child.stdin.flush()
                child.wait(timeout=10)
                child.stdin.close();child.stdout.close()

    def test_real_EOF_is_insufficient_but_actual_reaped_zero_exit_publishes_once(self):
        with self.fixture() as (root,child,record,source):
            self.assertEqual(child.stdout.read(),b'')
            path=root/'receipt.json'
            self.assertIsNone(module.poll(record,path,guard=lambda **kw:None))
            self.assertFalse(path.exists());self.assertIsNone(child.poll())
            child.stdin.write(b'x');child.stdin.flush();child.wait(timeout=10)
            gate=module.poll(record,path,guard=lambda **kw:None)
            proof=json.loads(path.read_text())
            self.assertTrue(proof['observed_by_actual_parent_waitpid'])
            self.assertFalse(proof['production_study_admitted'])
            self.assertEqual(gate['producer'],record.producer)
            self.assertEqual(proof['exit_code'],0)
            with self.assertRaisesRegex(ValueError,'overwrite'):module.poll(record,path,guard=lambda **kw:None)

    def test_nonzero_exit_never_creates_success_receipt(self):
        with self.fixture(status=7) as (root,child,record,source):
            child.stdin.write(b'x');child.stdin.flush();child.wait(timeout=10)
            with self.assertRaisesRegex(ValueError,'failed'):module.poll(record,root/'receipt.json',guard=lambda **kw:None)
            self.assertFalse((root/'receipt.json').exists())

    def test_source_or_owner_change_blocks_receipt_even_after_zero_exit(self):
        with self.fixture() as (root,child,record,source):
            child.stdin.write(b'x');child.stdin.flush();child.wait(timeout=10)
            source.write_text('# changed frozen source\n')
            with self.assertRaises(ValueError):module.poll(record,root/'receipt.json',guard=lambda **kw:None)
            self.assertFalse((root/'receipt.json').exists())
            from dataclasses import replace
            wrong=replace(record,owner=dict(record.owner,pid=record.owner['pid']+1000000))
            with self.assertRaisesRegex(ValueError,'parent owner'):module.poll(wrong,root/'receipt.json',guard=lambda **kw:None)

    def test_live_identity_change_and_wrong_handle_are_not_exit_evidence(self):
        with self.fixture() as (root,child,record,source):
            from dataclasses import replace
            wrong=replace(record,producer=dict(record.producer,pid=record.producer['pid']+1000000))
            with self.assertRaisesRegex(ValueError,'child handle'):module.poll(wrong,root/'receipt.json',guard=lambda **kw:None)
            actual=dict(record.producer,started='different lifetime')
            with patch.object(module.consumer.identity,'process',side_effect=lambda pid:record.owner if pid==os.getpid() else actual):
                with self.assertRaisesRegex(ValueError,'lifetime'):module.poll(record,root/'receipt.json',guard=lambda **kw:None)
            self.assertIsNone(child.poll());self.assertFalse((root/'receipt.json').exists())
            self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
