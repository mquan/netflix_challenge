from pathlib import Path
import tempfile
import unittest
import upstream_remote_temporal_archive as r


class CompleteArchiveTest(unittest.TestCase):
    def test_all63_output_hashes_are_required_and_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'preds_new').mkdir();outputs={}
            for index in range(63):
                p=root/'preds_new'/f'tsvdx5-150o-part{index}.out';p.write_text(str(index))
                outputs[str(p.relative_to(root))]=r.io.sha(p)
            self.assertEqual(len(r.checked_outputs(root,dict(outputs=outputs))),63)
            p=root/'preds_new/tsvdx5-150o-part62.out';p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'output changed'):r.checked_outputs(root,dict(outputs=outputs))
            p.write_text('62');bad=dict(outputs);bad.pop(str(p.relative_to(root)));bad['../outside']=r.io.sha(p)
            with self.assertRaisesRegex(ValueError,'outside'):r.checked_outputs(root,dict(outputs=bad))
            with self.assertRaisesRegex(ValueError,'Complete63'):r.checked_outputs(root,dict(outputs={}))


if __name__=='__main__':unittest.main()
