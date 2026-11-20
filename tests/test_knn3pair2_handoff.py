"""Check full-size source admission and exact virtual-column preservation."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_knn3pair2_screen as screen
import upstream_knn3pair2_virtual_features as features
import upstream_knn3pair2_virtual_mps_context as model
import upstream_attention2_virtual_mps_context as original


class Knn3PairTests(unittest.TestCase):
    def test_original_full_model_recipe_retained(self):
        before, after = dict(original.CONFIG), dict(model.CONFIG)
        before.pop('architecture');after.pop('architecture')
        self.assertEqual(before, after)
        self.assertEqual(model.CONFIG['nn_hidden'], 128)
        self.assertEqual(model.CONFIG['context_rank'], 32)
        self.assertEqual(model.SIGNALS, original.SIGNALS)
        self.assertEqual(model.CONTEXT, original.CONTEXT)
        self.assertEqual(max(model.SIGNALS), 399)

    def test_virtual_columns_preserve_entire_parent(self):
        rng = np.random.default_rng(839)
        old = rng.normal(size=(13, 183))
        context = rng.normal(size=(13, 397)).astype(np.float32)
        raw = [np.linspace(-1, 7, 13, dtype=np.float32), np.linspace(8, -2, 13, dtype=np.float32)]
        z, f = features.block((old, context, raw), 2, 11)
        self.assertEqual(z.shape, (9, 185))
        self.assertEqual(f.shape, (9, 399))
        np.testing.assert_array_equal(z[:, :182], old[2:11, :182])
        np.testing.assert_array_equal(z[:, -1], old[2:11, -1])
        np.testing.assert_array_equal(f[:, :243], context[2:11, :243])
        np.testing.assert_array_equal(f[:, 245:], context[2:11, 243:])
        for j in range(2):
            expected = np.clip(raw[j][2:11], 0, 6)
            np.testing.assert_array_equal(z[:, 182+j], expected)
            np.testing.assert_array_equal(f[:, 243+j], expected-np.float32(3.6))

    def proofs(self):
        rows = {}
        for name in screen.NEW:
            known, qual = f'artifacts/fixture/{name}.known.npy', f'artifacts/fixture/{name}.qual.npy'
            rows[screen.PROOFS[name]] = dict(state='complete_original_two_phase_knn3_chain',
                job=name, parent=screen.PARENTS[name], source_pin=screen.u.PIN,
                both_full_native_neighbor_builds_verified=True, original_default_configuration_verified=True,
                complete_parent_terminal_inputs_bound=True, full_known_projection_recomputed=True,
                known_training_disjointness_bound_to_unchanged_original_data=True,
                qualifying_targets_zero=True, qualifying_labels_read=False, public_prediction_values_read=False,
                projected_pairs=[dict(name=name, probex=known, qual=qual,
                                      probex_sha256='exact', qual_sha256='exact')],
                bindings={known: 'exact', qual: 'exact'})
        return rows

    def test_admit_only_both_complete_original_chain_lineages(self):
        rows = self.proofs()
        with patch.object(screen.u, 'read', side_effect=lambda p: rows[p]), patch.object(screen.u, 'sha', return_value='exact'):
            result, _ = screen.verified_inputs()
        self.assertEqual([r['name'] for r in result], screen.NEW)
        for field in ['both_full_native_neighbor_builds_verified', 'original_default_configuration_verified',
                      'complete_parent_terminal_inputs_bound', 'full_known_projection_recomputed',
                      'known_training_disjointness_bound_to_unchanged_original_data', 'qualifying_targets_zero']:
            broken = copy.deepcopy(rows)
            broken[screen.PROOFS[screen.NEW[0]]][field] = False
            with patch.object(screen.u, 'read', side_effect=lambda p: broken[p]), patch.object(screen.u, 'sha', return_value='exact'):
                with self.assertRaises(ValueError):
                    screen.verified_inputs()
        broken = copy.deepcopy(rows)
        broken[screen.PROOFS[screen.NEW[1]]]['parent'] = 'unrelated-model'
        with patch.object(screen.u, 'read', side_effect=lambda p: broken[p]), patch.object(screen.u, 'sha', return_value='exact'):
            with self.assertRaises(ValueError):
                screen.verified_inputs()

    def test_missing_source_proof_never_admitted(self):
        with patch.object(screen.u, 'read', side_effect=FileNotFoundError('native producer not complete')):
            with self.assertRaises(FileNotFoundError):
                screen.verified_inputs()


if __name__ == '__main__':
    unittest.main()
