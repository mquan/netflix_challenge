"""Private downloaded-array eligibility fixtures; no production files deleted."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_retire_public_qual_arrays as c


class Eligibility(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.folder=self.root/'public';self.folder.mkdir();self.path=self.folder/'sample.qual.npy'
        np.save(self.path,np.zeros(2817131,dtype='<f4'),allow_pickle=False)
        self.entry=dict(path='preds_old/sample.qual.npy',size=self.path.stat().st_size,md5=c.sha(self.path,'md5'))
        self.row=dict(path=str(self.path),source=self.entry['path'],md5=self.entry['md5'],sha256=c.sha(self.path))
        self.anchors={str(self.path):self.row};self.index={self.entry['path']:self.entry}
        self.patches=[patch.object(c,'OLD',self.root),patch.object(c,'ALLOWED',(self.folder,))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def classify(self,paths=None,protected=None,tracked=None):
        return c.classify(paths or [self.path],self.anchors,self.index,protected or set(),tracked or set())

    def test_positive_original_full_array_and_complete_hardlink_aliases(self):
        alias=self.folder/'alias/sample.qual.npy';alias.parent.mkdir();os.link(self.path,alias)
        result=self.classify([self.path,alias])
        self.assertEqual(result['sha256'],self.row['sha256']);self.assertEqual(result['metadata']['nlink'],2)
        self.assertEqual(result['download_manifest_entries'],[self.row])

    def test_protected_own_inode_and_tracked_file_are_retained(self):
        s=self.path.stat()
        with self.assertRaisesRegex(ValueError,'Own scientific'):self.classify(protected={(s.st_dev,s.st_ino)})
        with self.assertRaises(ValueError):self.classify(tracked={str(self.path.relative_to(self.root)).encode()})

    def test_missing_or_nonpublic_hardlink_is_retained(self):
        alias=self.root/'outside.qual.npy';os.link(self.path,alias)
        with self.assertRaisesRegex(ValueError,'Every hardlink'):self.classify()
        with self.assertRaises(ValueError):self.classify([self.path,alias])

    def test_manifest_or_index_mutation_is_rejected(self):
        for target,key in ((self.row,'sha256'),(self.row,'md5'),(self.entry,'md5'),(self.entry,'size')):
            value=target[key];target[key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):self.classify()
            target[key]=value
        self.anchors.clear()
        with self.assertRaisesRegex(ValueError,'download-manifest alias'):self.classify()

    def test_changed_numerical_byte_is_retained(self):
        with self.path.open('r+b') as f:f.seek(-1,2);f.write(b'\x01')
        with self.assertRaisesRegex(ValueError,'numerical bytes'):self.classify()

    def test_wrong_shape_remains_ineligible_even_if_catalog_hashes_agree(self):
        np.save(self.path,np.zeros(5,dtype='<f4'),allow_pickle=False)
        self.entry.update(size=self.path.stat().st_size,md5=c.sha(self.path,'md5'))
        self.row.update(md5=self.entry['md5'],sha256=c.sha(self.path))
        with self.assertRaisesRegex(ValueError,'full qualifying prediction'):self.classify()

    def test_symlink_and_parent_redirect_are_retained(self):
        link=self.folder/'link.qual.npy';link.symlink_to(self.path)
        with self.assertRaises(ValueError):self.classify([link])
        redirected=self.folder/'redirect';redirected.symlink_to(self.folder,target_is_directory=True)
        path=redirected/self.path.name
        self.anchors[str(path)]=dict(self.row,path=str(path))
        with self.assertRaises(ValueError):self.classify([path])

    def test_coherently_replaced_download_catalog_cannot_become_new_expectation(self):
        p=self.root/'replacement-index.toml';p.write_text('[meta]\nversion=1\n')
        with patch.object(c,'INDEX',p),self.assertRaisesRegex(ValueError,'Fixed historical'):
            c.manifests()


if __name__=='__main__':unittest.main()
