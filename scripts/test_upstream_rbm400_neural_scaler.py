"""Full-width exact scaler CPU fixtures; no real neural/model admission."""
import unittest
import numpy as np
import upstream_rbm400_neural_scaler as current
import upstream_exact_trajectory_scaler as previous
import upstream_exact_neural_scaling as transformation


def independent(raw,eligible):
    result={key:np.empty(raw.shape[1],dtype=np.float64) for key in ('lo','hi','mean','std')}
    for column in range(raw.shape[1]):
        values=raw[eligible,column]
        lo,hi=np.percentile(values,[1,99]);result['lo'][column]=lo;result['hi'][column]=hi
        clipped=np.clip(values.astype(np.float64),lo,hi);clipped[0]+=0.
        mean=np.add.accumulate(clipped)[-1]/len(clipped)
        squared=np.clip(values.astype(np.float64),lo,hi)-mean
        np.multiply(squared,squared,out=squared);squared[0]+=0.
        result['mean'][column]=mean;result['std'][column]=max(np.sqrt(np.add.accumulate(squared)[-1]/len(squared)),1e-6)
    return result


class ScalerTests(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(128);self.calls=0
    def guard(self):self.calls+=1
    def equal(self,a,b):
        self.assertEqual(set(a),set(b))
        for key in a:self.assertEqual(a[key].tobytes(),b[key].tobytes(),key)

    def test_all_four_declared_widths_and_full_independent_column_statistics(self):
        for width in current.WIDTHS:
            with self.subTest(width=width):
                raw=self.rng.normal(size=(8201,width)).astype('<f4');raw[:,-2]=3.;raw[0,0]=-0.
                eligible=np.arange(len(raw))%3!=1
                result=current.fit_scaler(raw,eligible,guard=self.guard)
                self.equal(result,independent(raw,eligible))
                scaled=current.apply_scaler(raw,result,guard=self.guard)
                expected=np.asarray((np.clip(raw,result['lo'],result['hi'])-result['mean'])/result['std'],dtype=np.float32,order='C')
                self.assertEqual(scaled.tobytes(),expected.tobytes())
        self.assertGreater(self.calls,3000)

    def test_both_original_supported_widths_keep_every_existing_scaler_transform_bit(self):
        for width in (713,714):
            raw=self.rng.normal(size=(321,width)).astype('<f4');mask=np.arange(len(raw))%2==0
            old=previous.fit_scaler(raw,mask);new=current.fit_scaler(raw,mask,guard=self.guard)
            self.equal(old,new)
            self.assertEqual(current.apply_scaler(raw,new,guard=self.guard).tobytes(),transformation.apply_scaler(raw,old).tobytes())

    def test_held_rows_do_not_change_eligible_stats_and_row_blocking_preserves_bits(self):
        raw=self.rng.normal(size=(67,864)).astype('<f4');eligible=np.arange(len(raw))%2==1
        before=raw.tobytes();expected=current.fit_scaler(raw,eligible,guard=self.guard)
        other=current.fit_scaler(raw,eligible,guard=self.guard,block_rows=7);self.equal(expected,other)
        self.assertEqual(raw.tobytes(),before)
        changed=raw.copy();changed[~eligible]=np.float32(1e20)
        self.equal(expected,current.fit_scaler(changed,eligible,guard=self.guard))

    def test_exact_inplace_normalization_matches_separate_output(self):
        raw=self.rng.normal(size=(101,863)).astype('<f4');mask=np.ones(len(raw),dtype=bool)
        scaler=current.fit_scaler(raw,mask,guard=self.guard)
        expected=current.apply_scaler(raw,scaler,guard=self.guard)
        alias=raw.copy();result=current.apply_scaler(alias,scaler,guard=self.guard,output=alias)
        self.assertIs(result,alias);self.assertEqual(alias.tobytes(),expected.tobytes())

    def test_rejects_missing_columns_empty_eligibility_nonfinite_and_wrong_output(self):
        raw=np.ones((17,864),dtype='<f4');mask=np.ones(17,dtype=bool)
        for value in (raw[:,:862].copy(),raw.astype('<f8'),np.asfortranarray(raw)):
            with self.assertRaises(ValueError):current.fit_scaler(value,mask,guard=self.guard)
        with self.assertRaises(ValueError):current.fit_scaler(raw,np.zeros(17,dtype=bool),guard=self.guard)
        broken=raw.copy();broken[16,863]=np.inf
        with self.assertRaises(ValueError):current.fit_scaler(broken,mask,guard=self.guard)
        scaler=current.fit_scaler(raw,mask,guard=self.guard)
        with self.assertRaises(ValueError):current.apply_scaler(raw,scaler,guard=self.guard,output=np.zeros((17,863),dtype='<f4'))


if __name__=='__main__':unittest.main()
