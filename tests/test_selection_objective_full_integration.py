"""Real CPU trainer and blend subprocesses through all six prepared stage validators."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_objective as so
import selection_objective_full as full
from gbdt_blend import HEADER
from test_joint import dataset


class FullPipelineIntegrationTests(unittest.TestCase):
    def test_real_six_stage_reports_and_predictions(self):
        torch.set_num_threads(1)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'data.bin'
            dataset(data)
            raw = bytearray(data.read_bytes())
            # Leave two movies unobserved so the choice task has real negatives.
            np.frombuffer(raw, dtype=HEADER, count=1)['items'] = 7
            data.write_bytes(raw)
            d = full.load(data)
            c = so.ms.Config(rank=4, hidden=8, d_model=8, heads=2, ff=16, layers=1,
                context='session', time_bins=4, dropout=.1, batch_users=16, batch_tokens=256,
                seq_len=16, max_targets=4, pad_width=4, pad_rows=4, device='cpu')
            torch.manual_seed(7)
            model = so.ms.SequenceModel(d['items'], c, so.ms.weeks(d))
            warm, warm_report = root/'warm.pt', root/'warm.json'
            torch.save(dict(config=dataclasses.asdict(c), items=d['items'], weeks=so.ms.weeks(d),
                            epoch=12, state=model.state_dict()), warm)
            warm_report.write_text(json.dumps(dict(config=dataclasses.asdict(c), selected_epoch=12,
                data_sha256=full.digest(data), checkpoint_sha256=full.digest(warm))))
            bank = root/'bank'
            (bank/'experiments').mkdir(parents=True)
            np.random.default_rng(21).uniform(2, 4, len(d['probe'])).astype('<f4').tofile(bank/'base.f32')
            (bank/'experiments/UNION-round-manifest.json').write_text(json.dumps(
                {'reports': {'union': {'inputs': {'base.f32': full.digest(bank/'base.f32')}}}}))
            (bank/'experiments/BOOST-support-inputs.json').write_text('{}')
            tree = dict(data=str(data), output=str(root/'reference-s3'), bank_root=str(bank),
                no_union=False, extra=[], extra_support=[], rounds=2, learning_rate=.05,
                leaves=3, min_leaf=5, processes=1, seeds=[7], extra_meta=True)
            neural = dict(data=str(data), output=str(root/'reference-NN2'), like=tree['output']+'.json',
                hidden=8, layers=1, dropout=.1, lr=.001, weight_decay=.01, batch=32,
                epochs=1, patience=1, seeds=[7], threads=1)
            def launch(command):
                result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            launch(full.arguments('round4_boost.py', tree))
            launch(full.arguments('round8_nn_blend.py', neural))
            base_report, neural_report = Path(tree['output']+'.json'), Path(neural['output']+'.json')
            base, nn = full.read(base_report), full.read(neural_report)
            p2 = dict(settings=dict(warm_stage='selection', epochs=2, lr=.0003, seed=19,
                positives=2, negatives=4, threads=1, device='cpu', max_batches=0), score_epoch=2,
                arms={'control': {'weight': 0.0}, 'treatment': {'weight': .1}},
                libraries={'torch': torch.__version__, 'numpy': np.__version__})
            p2_path = root/'p2.json'
            p2_path.write_text(json.dumps(p2))
            output = root/'pair'
            output.mkdir()
            stages = full.make_stages(p2, base, nn, root, data, warm_report, warm, output, 'fixture', 1)
            inputs = [data, warm, warm_report, p2_path, base_report, neural_report]
            inputs += [ROOT/'scripts'/name for name in ('selection_objective.py', 'masked_sequence.py',
                'masked_history.py', 'gbdt_blend.py')]
            plan = dict(data=str(data), rows=len(d['probe']), p2_plan=str(p2_path),
                base_report=str(base_report), neural_report=str(neural_report),
                frozen={str(path): full.digest(path) for path in inputs})
            generated, completed = {}, []
            def check():
                full.verify(plan['frozen'])
                full.verify(generated)
            def validate(stage):
                generated.update(full.verify_training(stage, plan) if stage['kind']=='training'
                                 else full.verify_blend(stage, plan))
                completed.append(stage['name'])
            # Scheduling/gate rejection is tested separately. These are tiny CPU
            # fixtures; this does not call or bypass the Netflix full-run gate.
            full.execute(stages, check, lambda stage: None, lambda stage: launch(stage['command']), validate)
            check()
            self.assertEqual(len(completed), 6)
            self.assertEqual(len(generated), 22)
            values = {}
            for arm in ('duplicate-control', 'treatment'):
                avg = np.fromfile(output/(arm+'-NN2.avg.f32'), '<f4').astype(float)
                nn_values = np.fromfile(output/(arm+'-NN2.nn.f32'), '<f4').astype(float)
                tree_values = np.fromfile(output/(arm+'-s3.refit.f32'), '<f4').astype(float)
                np.testing.assert_allclose(avg, (nn_values+tree_values)/2, rtol=0, atol=1e-6)
                values[arm] = avg
            reference = np.fromfile(Path(neural['output']+'.avg.f32'), '<f4').astype(float)
            result = full.compare_full(d['probe']['rating'].astype(float), d['probe']['user'], reference,
                values['duplicate-control'], values['treatment'], .0003)
            self.assertEqual(result['primary']['rows'], len(d['probe']))
            self.assertTrue(all(v is not None and np.isfinite(v) for v in result['primary']['fold_deltas']))
            control = torch.load(output/'control.e2.pt', weights_only=True)['state']
            treatment = torch.load(output/'treatment.e2.pt', weights_only=True)['state']
            self.assertFalse(torch.equal(control['choice_bias'], treatment['choice_bias']))


if __name__ == '__main__':
    unittest.main()
