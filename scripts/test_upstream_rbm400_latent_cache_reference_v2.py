"""Reference-alias regression plus all original cache adversarial fixtures."""
from pathlib import Path
import tempfile
import numpy as np
import unittest
import upstream_rbm400_latent_cache as original
import upstream_rbm400_latent_cache_reference_v2 as cache
import test_upstream_rbm400_latent_cache as fixtures
sha=fixtures.sha


class ReferenceTests(fixtures.CacheTests):
    def load(self):return cache.load(self.root,'manifest.json',self.digest,self.guard)

    def add_native_alias(self):
        name,target=next(iter(cache.APPROVED_REFERENCE_ALIASES.items()))
        canonical=self.root/target;canonical.parent.mkdir(parents=True)
        np.save(canonical,np.arange(17,dtype='<i4'),allow_pickle=False)
        logical=self.root/name;logical.parent.parent.mkdir(parents=True)
        logical.parent.symlink_to(canonical.parent,target_is_directory=True)
        self.manifest['references'][name]=self.manifest['references'][target]=sha(canonical)
        self.freeze();return name,target

    def test_original_failure_reproduced_and_only_declared_same_hash_twin_admitted(self):
        self.add_native_alias()
        with self.assertRaisesRegex(ValueError,'traverse symlinks'):
            with original.load(self.root,'manifest.json',self.digest,self.guard):pass
        with self.load() as (contexts,manifest):
            self.assertFalse(manifest['production_provider_admitted'])
            self.assertEqual(contexts['known'].users.shape,(3,50))
            self.assertEqual(len(manifest['references']),3)

    def test_missing_or_different_canonical_twin_cannot_replace_original_reference(self):
        name,target=self.add_native_alias();self.manifest['references'].pop(target);self.freeze();self.reject()
        self.manifest['references'][target]='0'*64;self.freeze();self.reject()

    def test_alias_cannot_escape_root_or_resolve_to_an_unapproved_equal_byte_copy(self):
        name,target=self.add_native_alias();logical=self.root/name;canonical=self.root/target
        alternate=self.root/'alternate';alternate.mkdir();(alternate/logical.name).write_bytes(canonical.read_bytes())
        logical.parent.unlink();logical.parent.symlink_to(alternate,target_is_directory=True);self.reject()
        with tempfile.TemporaryDirectory() as folder:
            outside=Path(folder).resolve();(outside/logical.name).write_bytes(canonical.read_bytes())
            logical.parent.unlink();logical.parent.symlink_to(outside,target_is_directory=True);self.reject()

    def test_reference_alias_never_grants_a_cache_payload_symlink(self):
        name,target=self.add_native_alias()
        self.manifest['arrays']['known/items']={'path':name,'sha256':self.manifest['references'][name]}
        self.freeze();self.reject()

    def test_alias_retarget_on_exceptional_exit_is_checked_and_maps_close(self):
        name,target=self.add_native_alias();logical=self.root/name;canonical=self.root/target
        alternate=self.root/'alternate';alternate.mkdir();(alternate/logical.name).write_bytes(canonical.read_bytes())
        with self.assertRaisesRegex(ValueError,'canonical source twin'):
            with self.load() as (contexts,_):
                mapped=contexts['known'].users.base
                logical.parent.unlink();logical.parent.symlink_to(alternate,target_is_directory=True)
                raise RuntimeError('Synthetic caller failure')
        self.assertTrue(mapped._mmap.closed)


if __name__=='__main__':unittest.main()
