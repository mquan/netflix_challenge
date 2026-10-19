"""Frozen feature semantics, query ordering, and rejection of incompatible inputs."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_joint import dataset
from gbdt_blend import HEADER, PROBE, load
import production_latent as prod
import round9_latent as old


def fixture(root):
    data = root / 'dev.bin'
    dataset(data)
    d = load(data)
    rng = np.random.default_rng(92)
    users, items, rank, sessions = d['users'], d['items'], 4, 500
    lengths = (users, users, users, items, items*30, items*8,
               sessions, sessions, users*rank, items*rank, items*rank, users*rank)
    model = root / 'model.bin'
    with model.open('wb') as f:
        f.write(struct.pack('<QQII', 0x4E46504A4F494E31, 42, rank, 3))
        for size in lengths:
            f.write(struct.pack('<Q', size))
            rng.normal(size=size).astype('<f4').tofile(f)
    m, info = old.read_model(model, users, items)
    cols, basis, _ = old.features(d, m, 3)
    lat = root / 'latent'
    lat.mkdir()
    np.save(lat / 'basis.npy', basis)
    outputs = {}
    for name, values in cols.items():
        path = lat / (name+'.f32')
        np.asarray(values, dtype='<f4').tofile(path)
        outputs[str(path.relative_to(root))] = old.digest(path)
    ref = dict(config=dict(data='dev.bin', model='model.bin', output='latent', dimensions=3),
               model=info, inputs={'dev.bin': old.digest(data), 'model.bin': old.digest(model)},
               features=list(outputs), outputs=outputs)
    manifest = lat / 'manifest.json'
    manifest.write_text(json.dumps(ref))
    production = root / 'production'
    production.mkdir()
    query_path = production / 'train-probe-qualifying.bin'
    # Fixture preserves the context but reverses query rows and hides all labels.
    raw = bytearray(data.read_bytes())
    offset = HEADER.itemsize + 8*(users+1) + 5*len(d['train_ratings'])
    queries = np.frombuffer(raw, dtype=PROBE, count=len(d['probe']), offset=offset)
    queries[:] = queries[::-1].copy()
    queries['rating'] = 0
    query_path.write_bytes(raw)
    pm = production / 'manifest.json'
    pm.write_text(json.dumps(dict(state='complete', source_sha256=old.digest(data),
        qualifying_count=len(queries), outputs={query_path.name: old.digest(query_path)})))
    return dict(reference_root=root, latent_manifest=manifest, data=data,
                model=model, production_manifest=pm), d, m, basis, cols


class ProductionLatentTests(unittest.TestCase):
    def test_full_export_matches_retained_and_preserves_query_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, d, m, basis, expected = fixture(root)
            report = prod.export(**args, output=root / 'out')
            self.assertEqual(report['feature_order'], list(expected))
            for name, values in expected.items():
                dev = np.fromfile(report['stages']['development_replay'][name]['path'], '<f4')
                query = np.fromfile(report['stages']['qualifying'][name]['path'], '<f4')
                np.testing.assert_array_equal(dev, np.asarray(values, '<f4'))
                np.testing.assert_array_equal(query, dev[::-1])
            with self.assertRaises(FileExistsError):
                prod.export(**args, output=root / 'out')
            changed = d['probe'].copy()
            changed['rating'] = 5
            for name, values in prod.features(d, m, basis, changed).items():
                np.testing.assert_array_equal(values, expected[name])

    def test_rotated_basis_and_changed_model_rejected_by_replay_or_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _, basis, _ = fixture(root)
            basis[:, 0] *= -1
            np.save(root / 'latent/basis.npy', basis)
            with self.assertRaisesRegex(ValueError, 'development feature replay differs'):
                prod.export(**args, output=root / 'rotated')
            model = args['model']
            raw = bytearray(model.read_bytes())
            raw[-1] ^= 1
            model.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'input digest mismatch'):
                prod.export(**args, output=root / 'changed')

    def test_unknown_triple_and_context_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, d, m, basis, _ = fixture(Path(temp))
            query = d['probe'][:1].copy()
            query['item'] = 1  # Known item and day, but never this user's pair.
            with self.assertRaisesRegex(ValueError, 'query absent'):
                prod.features(d, m, basis, query)
            for name in ('users', 'items', 'min_day', 'max_day'):
                changed = dict(d, **{name: d[name]+1})
                with self.assertRaisesRegex(ValueError, 'changed context'):
                    prod.same_context(d, changed)
            for name in ('implicit_offsets', 'implicit_items', 'implicit_days'):
                values = d[name].copy()
                values[-1] += 1
                with self.assertRaisesRegex(ValueError, 'changed context'):
                    prod.same_context(d, dict(d, **{name: values}))


if __name__ == '__main__':
    unittest.main()
