"""Contracts for contextual RBM and user-specific temporal neighborhoods."""
import array
import pathlib
import struct
import tempfile
import unittest
from test_joint import dataset, run

class PushTests(unittest.TestCase):
    def test_rbm_context_reload_isolation_and_zero_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            for name in ('a', 'b', 'zero', 'control'):
                dataset(p/(name+'.data'), change_fold=0 if name == 'b' else None)
                args = ['rbm', '--data', p/(name+'.data'), '--model', p/(name+'.model'),
                        '--predictions', p/(name+'.f32'), '--hidden', 4, '--threads', 2,
                        '--batch', 20, '--iterations', 1, '--visible-context', int(name != 'control')]
                if name == 'zero':
                    args += ['--user-visible-rate', 0, '--day-visible-rate', 0, '--frequency-visible-rate', 0]
                run(*args)
            self.assertEqual((p/'a.model').read_bytes()[16:], (p/'b.model').read_bytes()[16:])
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            self.assertEqual((p/'zero.f32').read_bytes(), (p/'control.f32').read_bytes())
            self.assertNotEqual((p/'a.f32').read_bytes(), (p/'control.f32').read_bytes())
            run('eval-rbm', '--data', p/'a.data', '--model', p/'a.model', '--predictions', p/'reload', '--threads', 2)
            self.assertEqual((p/'reload').read_bytes(), (p/'a.f32').read_bytes())
            run('rbm', '--data', p/'control.data', '--model', p/'warm', '--init-model', p/'control.model',
                '--hidden', 4, '--iterations', 2, '--start-epoch', 2, '--visible-context', 1, '--threads', 2)

    def test_temporal_reload_isolation_and_zero_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            for name in ('a', 'b', 'zero', 'control'):
                dataset(p/(name+'.data'), change_fold=0 if name == 'b' else None, correlated_train=True)
                run('graph', '--data', p/(name+'.data'), '--graph', p/(name+'.graph'), '--neighbors', 3, '--history-cap', 3)
                args = ['integrated', '--data', p/(name+'.data'), '--model', p/(name+'.model'),
                        '--graph', p/(name+'.graph'), '--factors', 4, '--iterations', 1,
                        '--predictions', p/(name+'.f32'), '--temporal-neighbors', int(name != 'control')]
                if name == 'zero': args += ['--neighbor-decay-rate', 0]
                run(*args)
            for ext in ('.model', '.model.nbr'):
                self.assertEqual((p/('a'+ext)).read_bytes()[16:], (p/('b'+ext)).read_bytes()[16:])
            self.assertEqual((p/'zero.f32').read_bytes(), (p/'control.f32').read_bytes())
            self.assertEqual((p/'a.f32').read_bytes(), (p/'b.f32').read_bytes())
            beta = array.array('f'); beta.frombytes((p/'a.model.nbr').read_bytes()[-100*4:])
            self.assertTrue(all(x >= 0 for x in beta))
            run('eval-integrated', '--data', p/'a.data', '--model', p/'a.model', '--graph', p/'a.graph', '--predictions', p/'reload')
            self.assertEqual((p/'a.f32').read_bytes(), (p/'reload').read_bytes())

if __name__ == '__main__': unittest.main()
