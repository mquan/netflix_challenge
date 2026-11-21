"""Synthetic lifecycle/FD checks; no actual source or MPS provider admitted."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_rbm400_latent_provider as provider


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.freeze_key = str((provider.original.REPORT / 'frozen.json').relative_to(provider.ROOT))
        self.done = dict(state='complete_local_rbm16_known_only_calibration_freeze_full_replay_and_independent_recovered_audit',
                         verification_sha256='1' * 64, qualifying_labels_used_for_fitting_or_selection=False)
        self.canonical = dict(state='complete_independent_local_rbm16_full_source_models_coefficients_and_frozen_group_scores',
                              qualifying_labels_used_for_fitting_or_selection=False,
                              all_six_vectors_independently_verified=True, full_inherited_source_closure_rehashed=True,
                              simplex_supports_examined=65535, ridge_equations=17,
                              primary=provider.original.CANDIDATES[0], bindings={self.freeze_key: '2' * 64})

    def require(self, done=None, canonical=None):
        return provider.require_consumer_completion(self.done if done is None else done,
                                                    self.canonical if canonical is None else canonical,
                                                    '1' * 64, '2' * 64)

    def test_full_lifecycle_contract_without_accessing_scores(self):
        self.require()
        class NoScores(dict):
            def __getitem__(self, key):
                if key in ('groups', 'results', 'both_groups_below_target'):
                    raise AssertionError('Recovered scores cannot influence extraction')
                return super().__getitem__(key)
        self.require(canonical=NoScores(self.canonical))

    def test_partial_or_changed_lifecycle_rejected(self):
        for key, bad in [('state', 'running'), ('verification_sha256', '3' * 64),
                         ('qualifying_labels_used_for_fitting_or_selection', True)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.require(done=dict(self.done, **{key: bad}))
        for key, bad in [('state', 'partial'), ('qualifying_labels_used_for_fitting_or_selection', True),
                         ('all_six_vectors_independently_verified', False),
                         ('full_inherited_source_closure_rehashed', False),
                         ('simplex_supports_examined', 32767), ('ridge_equations', 16),
                         ('primary', 'selected-after-audit')]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.require(canonical=dict(self.canonical, **{key: bad}))
        bad = copy.deepcopy(self.canonical)
        bad['bindings'][self.freeze_key] = '4' * 64
        with self.assertRaises(ValueError):
            self.require(canonical=bad)

    def test_phase_and_environment_rejected_before_sources_or_torch(self):
        with patch.object(provider, 'sources', side_effect=AssertionError('Cannot start source admission')):
            with self.assertRaises(ValueError):
                provider.extract_phase('partial', lambda: None)
            with patch.dict(provider.os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': '1'}):
                with self.assertRaises(ValueError):
                    provider.extract_phase('known', lambda: None)

    def terminal_fixture(self, root):
        import torch
        path = root / 'known.pt'
        torch.save({'model': {'p': torch.tensor([1., 2.])}, 'synthetic': True}, path)
        return path, provider.inputs.verify_files(root, {'known.pt': provider.inputs.sha(path)}, lambda: None)['known.pt']

    def test_authenticated_fd_load_is_readonly_and_preserves_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path, expected = self.terminal_fixture(root)
            with provider.terminal_payload(path, expected, lambda: None) as record:
                self.assertTrue(record['synthetic'])
                self.assertEqual(record['model']['p'].tolist(), [1., 2.])
            self.assertEqual(provider.inputs.sha(path), expected['sha256'])

    def test_replacement_symlink_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path, expected = self.terminal_fixture(root)
            alias = root / 'alias.pt'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                with provider.terminal_payload(alias, expected, lambda: None):
                    self.fail('Symlink accepted')
            with self.assertRaises(ValueError):
                with provider.terminal_payload(path, expected, lambda: None):
                    with path.open('ab') as stream:
                        stream.write(b'changed')
            with self.assertRaises(ValueError):
                with provider.terminal_payload(path, expected, lambda: None):
                    self.fail('Changed terminal accepted')

    def test_exceptional_exit_still_checks_original_model_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path, expected = self.terminal_fixture(root)
            with self.assertRaisesRegex(ValueError, 'changed during extraction'):
                with provider.terminal_payload(path, expected, lambda: None):
                    with path.open('ab') as stream:
                        stream.write(b'changed')
                    raise RuntimeError('Caller fails')


if __name__ == '__main__':
    unittest.main()
