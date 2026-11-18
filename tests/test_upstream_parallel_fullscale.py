from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_parallel_fullscale as p


class ParallelFullscaleChecks(unittest.TestCase):
    def test_resource_accounting_includes_registered_python_fitter_once(self):
        rows='''10 11 100 /tmp/target/release/run -n model
12 11 200 ./target/release/model
20 22 400 ./target/release/cfnade-stage x
30 33 500 python fitter.py
31 33 50 python helper.py
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p.u.write(root/'experiments/upstream-extension-run/worker-status.json',dict(pid=33))
            p.u.write(root/'experiments/upstream-production/status.json',dict(pid=33))
            with patch.object(p.u,'ROOT',root),patch.object(p.subprocess,'check_output',return_value=rows):
                self.assertEqual(p.other_rss(),1250*1024)
                self.assertEqual(p.other_rss(exclude=11),950*1024)

    def test_data_aliases_are_isolated_idempotent_and_reject_wrong_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'candidate';fresh=root/'fresh';base=root/'base'
            (tree/'data').mkdir(parents=True);(base/'target').mkdir(parents=True)
            for split in ['raw','train','probe','trainx','probex','fulltrain','qual']:
                (fresh/'data'/split).mkdir(parents=True)
            with patch.object(p,'TREE',tree),patch.object(p,'BASE',base),patch.object(p.u,'FRESH',fresh):
                p.setup();p.setup()
                self.assertEqual((tree/'data/qual').resolve(),(fresh/'data/qual').resolve())
                (tree/'data/probex').unlink();(tree/'data/probex').symlink_to(fresh/'data/qual')
                with self.assertRaisesRegex(ValueError,'Incorrect parallel data alias'):p.setup()

    def test_changed_source_cannot_reuse_a_passing_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';tree=root/'source';report=root/'report'
            (tree/'src').mkdir(parents=True);(tree/'src/tx.rs').write_text('changed')
            prepared=dict(source_pin=p.u.PIN,script_sha256=p.u.sha(Path(p.candidate.__file__)),
                          generated={'src/tx.rs':'wrong-hash'})
            p.u.write(base/'parallel-preparation.json',prepared)
            p.u.write(report/'verification.json',dict(pass_all=True,preparation=prepared))
            with patch.object(p,'BASE',base),patch.object(p,'TREE',tree),patch.object(p.candidate,'REPORT',report),\
                 patch.object(p.u,'validate_fresh'):
                with self.assertRaisesRegex(ValueError,'Parallel candidate changed'):p.verify_candidate()


if __name__=='__main__':unittest.main()
