"""Reject changes to published scientific provenance using private byte fixtures."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from build_research_journey import verify_own14_publication_sources


class SourceIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.files=['scripts/upstream_own14_matched391_calibration.py',
                    'scripts/verify_upstream_own14_matched391_calibration.py',
                    'artifacts/upstream-own391-paired-h128-mps/nn/0/model.pt',
                    'artifacts/upstream-own391-paired-h128-mps/nn/0/scaler.npz',
                    'artifacts/upstream-own391-fold-bag/nn-bag.f32',
                    'experiments/upstream-own391-paired-h128-mps/independent-verification.json',
                    'experiments/upstream-own391-fold-bag/independent-verification.json']
        self.bindings={}
        for name in self.files:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(('fixture:'+name).encode())
            self.bindings[name]=hashlib.sha256(p.read_bytes()).hexdigest()

    def test_complete_sources(self):
        self.assertEqual(sum(verify_own14_publication_sources(self.root,self.bindings).values()),7)

    def test_code_model_scaler_bag_and_replay_tampering(self):
        for name in self.files:
            with self.subTest(name=name):
                p=self.root/name;original=p.read_bytes();p.write_bytes(original+b'changed')
                try:
                    with self.assertRaises(ValueError):verify_own14_publication_sources(self.root,self.bindings)
                finally:p.write_bytes(original)

    def test_missing_bag_expectation(self):
        del self.bindings['artifacts/upstream-own391-fold-bag/nn-bag.f32']
        with self.assertRaises(ValueError):verify_own14_publication_sources(self.root,self.bindings)

    def test_external_source_path(self):
        self.bindings['../external.py']='0'*64
        with self.assertRaises(ValueError):verify_own14_publication_sources(self.root,self.bindings)


if __name__=='__main__':unittest.main()
