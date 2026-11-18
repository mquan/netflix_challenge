"""Receipt integrity and process ownership for the long native queue."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_full_reproduce as f


class FullReproduceChecks(unittest.TestCase):
    def test_receipt_requires_complete_and_unchanged_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(f.u,'BASE',root/'base'),patch.object(f.u,'FRESH',root/'fresh'):
                (f.u.BASE/'fresh-receipts').mkdir(parents=True)
                f.u.FRESH.mkdir()
                out=f.u.FRESH/'model.npy';out.write_bytes(b'fresh prediction')
                jobs={'model':{'command':'train model','outputs':['model.npy']}}
                self.assertFalse(f.receipt_valid('model',jobs))
                record=dict(source_commit=f.u.PIN,job='model',command='train model',outputs={})
                receipt=f.u.BASE/'fresh-receipts/model.json'
                receipt.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'Incomplete'):f.receipt_valid('model',jobs)
                record['outputs']['model.npy']=f.u.sha(out)
                receipt.write_text(json.dumps(record))
                self.assertTrue(f.receipt_valid('model',jobs))
                out.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Changed'):f.receipt_valid('model',jobs)

    def test_rss_counts_only_owned_process_group(self):
        with patch.object(f.subprocess,'check_output',return_value=' 42 100\n 42 200\n 43 900000\n'):
            self.assertEqual(f.group_rss(42),300*1024)

    def test_stop_never_signals_finished_process(self):
        class Finished:
            def poll(self):return 0
        with patch.object(f.os,'killpg') as kill:
            f.stop(Finished());kill.assert_not_called()


if __name__=='__main__':unittest.main()
