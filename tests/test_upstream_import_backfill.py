"""Receipt integrity, transactional linking and real scheduler-signal recovery."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_import_backfill as m


class ImportBackfillTest(unittest.TestCase):
    def fixture(self,root):
        source,target=root/'source',root/'target';receipts=root/'source-receipts';dest=root/'target-receipts'
        receipts.mkdir();target.mkdir();files=['preds_new/own.probex.npy','preds_new/own.qual.npy']
        for i,name in enumerate(files):
            p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(bytes(range(16+i)))
        jobs={'own':dict(command='fresh native own',outputs=files)}
        record=dict(job='own',source_commit=m.u.PIN,command=jobs['own']['command'],outputs={name:m.u.sha(source/name) for name in files})
        m.u.write(receipts/'own.json',record)
        return source,target,receipts,dest,jobs,record

    def test_hardlinks_keep_source_lineage_and_existing_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,receipts,dest,jobs,record=self.fixture(Path(tmp))
            plan=m.transfer_plan(source,receipts,target,dest,jobs)
            self.assertEqual(m.apply_transfers(plan,source,target,dest,jobs),['own'])
            result=m.u.read(dest/'own.json');m.validate_record('own',result,target,jobs)
            self.assertEqual(result['imported_from']['native_receipt_sha256'],m.u.sha(receipts/'own.json'))
            for name in record['outputs']:self.assertTrue(os.path.samefile(source/name,target/name))
            original=(dest/'own.json').read_bytes()
            self.assertEqual(m.apply_transfers(plan,source,target,dest,jobs),[])
            self.assertEqual(original,(dest/'own.json').read_bytes())

    def test_changed_source_or_conflicting_destination_fails_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,receipts,dest,jobs,record=self.fixture(Path(tmp))
            name=next(iter(record['outputs']));p=target/name;p.parent.mkdir(parents=True);p.write_bytes(b'other owned data')
            with self.assertRaisesRegex(ValueError,'conflicts'):m.transfer_plan(source,receipts,target,dest,jobs)
            self.assertEqual(p.read_bytes(),b'other owned data');self.assertFalse(dest.exists())
            (source/name).write_bytes(b'corrupt source')
            with self.assertRaisesRegex(ValueError,'output changed'):m.transfer_plan(source,receipts,target,dest,jobs)

    def test_link_failure_rolls_back_only_unpublished_new_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,target,receipts,dest,jobs,record=self.fixture(Path(tmp))
            plan=m.transfer_plan(source,receipts,target,dest,jobs);real_link=os.link;calls=[]
            def interrupted(a,b):
                calls.append(b)
                if len(calls)==2:raise OSError('simulated link failure')
                real_link(a,b)
            with patch.object(m.os,'link',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'simulated'):m.apply_transfers(plan,source,target,dest,jobs)
            self.assertFalse((dest/'own.json').exists())
            self.assertFalse(any((target/name).exists() for name in record['outputs']))
            for name,digest in record['outputs'].items():self.assertEqual(m.u.sha(source/name),digest)

    def test_actual_pause_is_released_when_body_raises(self):
        marker='owned-boundary-signal-test'
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',marker],start_new_session=True)
        try:
            with patch.object(m,'PRIMARY_FRAGMENT',marker):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    with m.paused_primary(child.pid):
                        row=next(r for r in m.process_rows() if r['pid']==child.pid)
                        self.assertIn('T',row['state']);raise RuntimeError('injected')
            row=next(r for r in m.process_rows() if r['pid']==child.pid)
            self.assertNotIn('T',row['state']);self.assertIsNone(child.poll())
        finally:
            os.kill(child.pid,signal.SIGCONT);child.terminate();child.wait(timeout=5)

    def test_unknown_pid_is_never_suspended(self):
        with self.assertRaisesRegex(ValueError,'changed'):
            with m.paused_primary(os.getpid()):self.fail('Must not enter')


if __name__=='__main__':unittest.main()
