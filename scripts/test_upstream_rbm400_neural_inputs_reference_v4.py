"""Full-width scalar/latent materialization checks; no production admission."""
import hashlib
import unittest
import numpy as np
import upstream_rbm400_neural_inputs_reference_v4 as module
import test_upstream_rbm400_context_inputs as fixtures


class NeuralInputTests(unittest.TestCase):
    def setUp(self):self.fixture=fixtures.InputTests();self.fixture.setUp()

    def test_every_raw_neural_bit_and_float32_baseline_for_all_four_widths(self):
        rng=np.random.default_rng(128)
        for keep in (False,True):
            for latent in (False,True):
                with self.subTest(keep=keep,latent=latent):
                    reader=self.fixture.reader(keep,latent);z,f=reader.block(0,9)
                    weights=rng.normal(size=z.shape[1]).astype(np.float64)
                    expected_streams=[hashlib.sha256(a.tobytes()).hexdigest() for a in (z,f)]
                    values,baseline,proof=module.materialize(reader,weights,expected_streams,
                        retain_scalar=keep,latent=latent,guard=lambda:None)
                    expected_baseline=np.clip(z@weights,1.05,4.95)
                    expected=np.empty((9,f.shape[1]+1),dtype='<f4');expected[:,:-1]=f;expected[:,-1]=expected_baseline
                    self.assertEqual(values.tobytes(),expected.tobytes())
                    self.assertEqual(baseline.tobytes(),expected_baseline.tobytes())
                    self.assertEqual(proof['raw_neural_input_sha256'],hashlib.sha256(expected.tobytes()).hexdigest())
                    self.assertEqual(proof['width'],713+int(keep)+150*int(latent))

    def test_operator_hash_mismatch_or_wrong_ridge_width_never_returns_inputs(self):
        reader=self.fixture.reader(True,True);weights=np.ones(563,dtype=np.float64)
        with self.assertRaises(ValueError):module.materialize(reader,weights,['0'*64,'1'*64],retain_scalar=True,latent=True,guard=lambda:None)
        with self.assertRaises(ValueError):module.materialize(reader,np.ones(562),['0'*64,'1'*64],retain_scalar=True,latent=True,guard=lambda:None)

    def test_alias_scalar_inputs_match_original_append_baseline_arithmetic(self):
        reader=self.fixture.reader(False,False);z,f=reader.block(0,9);weights=np.arange(562,dtype=np.float64)/562
        values,baseline,_=module.materialize(reader,weights,[hashlib.sha256(a.tobytes()).hexdigest() for a in (z,f)],
                                              retain_scalar=False,latent=False,guard=lambda:None)
        original=np.empty((9,713),dtype=np.float32);original[:,:-1]=f;original[:,-1]=np.clip(z@weights,1.05,4.95)
        self.assertEqual(values.tobytes(),original.tobytes());self.assertEqual(values[:,-1].tobytes(),baseline.astype('<f4').tobytes())


if __name__=='__main__':unittest.main()
