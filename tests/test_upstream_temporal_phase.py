from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_temporal_phase as phase
import upstream_phase_bank as bank
import upstream_phase_production as production


class TemporalPhaseChecks(unittest.TestCase):
    def test_original_config_is_preserved_and_only_fit_invocation_changes(self):
        source=phase.mapped.pinned(phase.SOURCE)
        text=phase.driver(source)
        original=source[source.index('fn run_ord('):source.index('\nfn run_nlpp(')]
        self.assertEqual(text[text.index('    let cfg'):text.index('    if std::env')],
                         original[original.index('    let cfg'):original.index('    fit2!')])
        self.assertIn('run_ord(150, 25, 120, 1.2, job_name)',text)
        self.assertIn('fit::<TxModel>(cfg,"rtg","fulltrain","qual",job_name,',text)
        self.assertNotIn('fit2!',text)
        # The source's second phase itself calls this generic fitting function.
        lib=phase.mapped.pinned('src/lib.rs')
        self.assertIn('fit::<M>(cfg, target, split.fulltrain_tr, split.fulltrain_pr, model_name,',lib)

    def test_unrecognized_original_recipe_is_rejected(self):
        source=phase.mapped.pinned(phase.SOURCE)
        with self.assertRaisesRegex(ValueError,'Missing original ordinal dispatch'):
            phase.driver(source.replace('"tsvdx5-150o" =>','"other" =>'))
        with self.assertRaisesRegex(ValueError,'fit invocation changed'):
            phase.driver(source.replace('save_train: true, save_subscores: true, save_probe_each_epoch: true',
                                        'save_train: false, save_subscores: true, save_probe_each_epoch: true'))

    def test_adapter_preserves_native_resolution_for_unrelated_columns(self):
        with patch.object(bank,'native_pair',return_value={'native':True}) as native:
            row=dict(name='mf-61',clip=True)
            self.assertEqual(bank.choose_pair(row,{},True),{'native':True})
            native.assert_called_once_with(row,{},True)

    def test_phase_bank_requires_bound_receipt_for_all_qual_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/'report';report.mkdir();(report/'paired-receipt.json').write_text('{}')
            for name in ['probe.npy','qual.npy']:(root/name).write_bytes(b'fixture')
            receipt=dict(pairs={phase.JOB:dict(probex='probe.npy',qual='qual.npy')},
                         outputs={'probe.npy':'a','qual.npy':'b'})
            with patch.object(bank.u,'ROOT',root),patch.object(phase,'REPORT',report),\
                 patch.object(phase,'verify_receipt',return_value=receipt) as verify:
                result=bank.choose_pair(dict(name=phase.JOB,clip=True),{},True)
                verify.assert_called_once_with(verify_files=True)
                self.assertEqual(result['kind'],'our-independent-source-phases')
                self.assertEqual(result['sha256'],dict(probex='a',qual='b'))
                (root/'qual.npy').unlink()
                self.assertIsNone(bank.choose_pair(dict(name=phase.JOB,clip=True),{}))

    def test_production_hashes_bind_original_core_and_adapters(self):
        hashes=production.code_hashes()
        for name in ['scripts/upstream_production.py','scripts/upstream_production_bank.py',
                     'scripts/upstream_phase_production.py','scripts/upstream_phase_bank.py',
                     'scripts/upstream_temporal_phase.py']:
            self.assertIn(name,hashes)


if __name__=='__main__':unittest.main()
