"""Private fixtures prove destructive cleanup rejects protected/misclassified caches."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import upstream_retire_public_stat_caches as retire


class RetirementTests(unittest.TestCase):
    def fixture(self,root):
        root=root.resolve()
        own=root/'own';old=root/'old';public=old/'artifacts/public-new';public.mkdir(parents=True)
        (own/'experiments').mkdir(parents=True);checkpoint=own/'checkpoint.bin';checkpoint.write_bytes(b'own actual model')
        manifest=public/'production-dev-manifest.json';manifest.write_text('{"historical_public_bank":true}')
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        for name in retire.NAMES:
            (public/(name+'.stats.npz.zst')).write_bytes(bytes.fromhex('28b52ffd')+b'obsolete public statistics')
            (public/(name+'.inputs.json')).write_text(json.dumps(dict(config=dict(output='artifacts/public-new/'+name,
                public='artifacts/public-new/production-dev-manifest.json'),public_manifest_sha256=digest,
                inputs={str(public/'columns'/('%03d.f32'%i)):'a'*64 for i in range(500)})))
            (public/(name+'.json')).write_text('{"historical_metadata":true}')
        proof=own/'experiments/frozen.json';proof.write_text(json.dumps(dict(source_sha256={'checkpoint.bin':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})))
        return own,old,public,checkpoint,proof

    def run_case(self,mutate=None,closed=True):
        with tempfile.TemporaryDirectory() as tmp:
            own,old,public,checkpoint,proof=self.fixture(Path(tmp));report=own/'experiments/retirement'
            if mutate:mutate(public,proof)
            with mock.patch.object(retire,'OLD',old),mock.patch.object(retire,'PUBLIC',public),mock.patch.object(retire,'REPORT',report),\
                 mock.patch.object(retire,'CLOSURES',['frozen.json']),mock.patch.object(retire.u,'ROOT',own),\
                 mock.patch.object(retire.subprocess,'check_output',return_value=b''),mock.patch.object(retire,'closed',return_value=closed):
                if mutate or not closed:
                    with self.assertRaises(ValueError):retire.main()
                    self.assertTrue(all((public/(name+'.stats.npz.zst')).exists() for name in retire.NAMES))
                else:
                    retire.main();self.assertTrue(all(not(public/(name+'.stats.npz.zst')).exists() for name in retire.NAMES))
                    record=json.loads((report/'verification.json').read_text());self.assertEqual(len(record['removed']),5)
                    self.assertFalse(record['models_or_checkpoints_removed'])
                self.assertEqual(checkpoint.read_bytes(),b'own actual model')
                self.assertTrue(all((public/(name+'.inputs.json')).exists() and (public/(name+'.json')).exists() for name in retire.NAMES))

    def test_only_named_public_statistics_removed_own_and_metadata_retained(self):self.run_case()

    def test_actual_protected_inode_rejected_before_any_unlink(self):
        def protected(public,proof):
            path=public/(retire.NAMES[2]+'.stats.npz.zst')
            proof.write_text(json.dumps(dict(source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})))
        self.run_case(protected)

    def test_wrong_provenance_rejected_before_any_unlink(self):
        def wrong(public,proof):
            path=public/(retire.NAMES[2]+'.inputs.json');record=json.loads(path.read_text())
            record['config']['output']='artifacts/own-fresh/model';path.write_text(json.dumps(record))
        self.run_case(wrong)

    def test_open_cache_rejected_before_any_unlink(self):self.run_case(closed=False)


if __name__=='__main__':unittest.main()
