"""Protect own/shared/tracked files and require complete public provenance."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import upstream_retire_remaining_public_stats as run


class PublicProvenance(unittest.TestCase):
    def test_actual_retained_manifests(self):
        for spec in run.SPECS:
            with self.subTest(prefix=spec[0]):
                record=json.loads((run.OLD/(spec[0]+'.inputs.json')).read_text())
                self.assertEqual(run.classify(spec,record),spec[3])

    def test_wrong_output_manifest_or_public_input_hash(self):
        spec=run.SPECS[0];original=json.loads((run.OLD/(spec[0]+'.inputs.json')).read_text())
        for kind in ('output','manifest','input'):
            with self.subTest(kind=kind):
                v=copy.deepcopy(original)
                if kind=='output':v['config']['output']='artifacts/own-model'
                elif kind=='manifest':v['public_manifest_sha256']='0'*64
                else:
                    manifest=json.loads((run.OLD/spec[1]).read_text());v['inputs'][manifest['columns'][0]['path']]='0'*64
                with self.assertRaises(ValueError):run.classify(spec,v)


class FileProtection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.path=self.root/'cache.zst';self.path.write_bytes(b'private fixture')

    def test_regular_untracked_fixture(self):
        self.assertEqual(run.regular_unbound(self.path,set(),{},self.root).st_nlink,1)

    def test_protected_own_inode(self):
        s=self.path.stat()
        with self.assertRaises(ValueError):run.regular_unbound(self.path,set(),{(s.st_dev,s.st_ino):'own model'},self.root)

    def test_hardlink_and_symlink(self):
        alias=self.root/'alias';os.link(self.path,alias)
        with self.assertRaises(ValueError):run.regular_unbound(self.path,set(),{},self.root)
        alias.unlink();alias.symlink_to(self.path)
        with self.assertRaises(ValueError):run.regular_unbound(alias,set(),{},self.root)

    def test_tracked_file(self):
        with self.assertRaises(ValueError):run.regular_unbound(self.path,{b'cache.zst'},{},self.root)


if __name__=='__main__':unittest.main()
