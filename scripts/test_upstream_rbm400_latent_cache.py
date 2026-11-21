"""IO adversarial checks on synthetic caches; no real provider admission."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_rbm400_latent_cache as cache


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'reference.json').write_text('{"synthetic": true}')
        self.manifest = dict(format=cache.FORMAT, users=3, items=51,
                             qualifying_labels_read=False, production_provider_admitted=False,
                             references={'reference.json': sha(self.root / 'reference.json')}, arrays={})
        self.values = {}
        for phase in ('known', 'qualifying'):
            users = np.arange(150, dtype='<f4').reshape(3, 50)
            users[1] = np.nan
            for name, value in [('users', users), ('items', np.ones((51, 50), dtype='<f4')),
                                ('covered', np.array([True, False, True]))]:
                self.values[phase + '/' + name] = value
        self.values.update(item_map=np.eye(50, dtype='<f8'), user_map=np.eye(50, dtype='<f8'))
        for name, value in self.values.items():
            relative = name.replace('/', '-') + '.npy'
            np.save(self.root / relative, value, allow_pickle=False)
            self.manifest['arrays'][name] = dict(path=relative, sha256=sha(self.root / relative))
        self.calls = 0
        self.freeze()

    def guard(self):
        self.calls += 1

    def freeze(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        self.digest = sha(self.root / 'manifest.json')

    def load(self):
        return cache.load(self.root, 'manifest.json', self.digest, self.guard)

    def reject(self):
        with self.assertRaises((ValueError, KeyError)):
            with self.load():
                self.fail('Malformed cache accepted')

    def replace_array(self, name, value):
        row = self.manifest['arrays'][name]
        np.save(self.root / row['path'], value, allow_pickle=False)
        row['sha256'] = sha(self.root / row['path'])
        self.freeze()

    def test_all_context_bits_and_readonly_maps(self):
        with self.load() as (contexts, manifest):
            self.assertFalse(manifest['production_provider_admitted'])
            for phase, context in contexts.items():
                users, items = np.array([2, 0, 2]), np.array([50, 1, 50])
                u = self.values[phase + '/users'][users]
                i = self.values[phase + '/items'][items]
                expected = np.concatenate([u, i, u * i], axis=1)
                self.assertEqual(context.values(users, items).tobytes(), expected.tobytes())
                self.assertFalse(context.users.flags.writeable)
                with self.assertRaises(ValueError):
                    context.values(np.array([1]), np.array([0]))
                held = context.users.base
        self.assertTrue(held._mmap.closed)
        self.assertGreater(self.calls, 20)

    def test_changed_manifest_payload_and_reference_fail(self):
        for relative in ('manifest.json', 'reference.json', 'known-users.npy'):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                self.reject()
                path.write_bytes(original)

    def test_changed_file_during_use_fails_and_closes_maps(self):
        with self.assertRaises(ValueError):
            with self.load() as (contexts, _):
                held = contexts['known'].users.base
                (self.root / 'reference.json').write_text('changed reference')
        self.assertTrue(held._mmap.closed)

    def test_trailing_bytes_fail_even_when_hash_matches(self):
        row = self.manifest['arrays']['qualifying/items']
        path = self.root / row['path']
        with path.open('ab') as stream:
            stream.write(b'extra')
        row['sha256'] = sha(path)
        self.freeze()
        self.reject()

    def test_exceptional_caller_exit_still_checks_closure_and_closes_maps(self):
        for relative in ('reference.json', 'manifest.json'):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    with self.load() as (contexts, _):
                        held = contexts['known'].users.base
                        path.write_bytes(original + b'changed')
                        raise RuntimeError('Caller fails during cache use')
                self.assertTrue(held._mmap.closed)
                path.write_bytes(original)

    def test_wrong_dtype_shape_and_fortran_order_fail(self):
        for value in (np.ones((51, 50), dtype='<f8'), np.ones((51, 49), dtype='<f4'),
                      np.asfortranarray(np.ones((51, 50), dtype='<f4'))):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                self.replace_array('known/items', value)
                self.reject()

    def test_path_escape_symlink_and_duplicate_payload_fail(self):
        row = self.manifest['arrays']['known/users']
        original = dict(row)
        (self.root / 'alias.npy').symlink_to(self.root / original['path'])
        for relative in ('../outside.npy', str(self.root / original['path']), 'alias.npy',
                         self.manifest['arrays']['qualifying/users']['path']):
            row['path'] = relative
            self.freeze()
            self.reject()
        row.update(original)

    def test_missing_fields_or_production_claims_fail(self):
        for key in ('qualifying_labels_read', 'production_provider_admitted'):
            self.manifest[key] = True
            self.freeze()
            self.reject()
            self.manifest[key] = False
        del self.manifest['arrays']['user_map']
        self.freeze()
        self.reject()

    def test_nonfinite_context_and_invalid_dual_map_fail(self):
        value = self.values['known/items'].copy()
        value[4, 6] = np.nan
        self.replace_array('known/items', value)
        self.reject()
        self.replace_array('known/items', self.values['known/items'])
        self.replace_array('user_map', np.eye(50, dtype='<f8') * 2)
        self.reject()


if __name__ == '__main__':
    unittest.main()
