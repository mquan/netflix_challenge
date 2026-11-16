import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qual_matrix import verified_matrix,predict_matrix_ridge
from streaming_blend import digest,predict_ridge


class QualMatrixTests(unittest.TestCase):
    def test_overlay_matches_physical_matrix_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=np.arange(30,dtype='f4').reshape(6,5);column=np.arange(6,dtype='f4')+.5
            np.save(root/'base.npy',base);column.tofile(root/'extra.f32');prefix=str(root/'qual')
            spec=dict(base_path=str(root/'base.npy'),base_sha256=digest(root/'base.npy'),
                      column_path=str(root/'extra.f32'),column_sha256=digest(root/'extra.f32'),
                      insert_at=3,shape=[6,6])
            Path(prefix+'.matrix.json').write_text(json.dumps(spec))
            report=dict(matrix_layout='column_insertion',composite_spec_sha256=digest(prefix+'.matrix.json'))
            overlay=verified_matrix(prefix,report);physical=np.column_stack([base[:,:3],column,base[:,3:]])
            for rows in [slice(None),slice(1,4),np.array([4,0,2]),3]:
                np.testing.assert_array_equal(overlay[rows],physical[rows])
            context=np.ones((6,2));w=np.arange(10)*.01
            np.testing.assert_allclose(predict_matrix_ridge(overlay,context,4,w),predict_ridge(physical[:,:4],context,w))
            report.update(state='complete',qualifying_labels_read=False)
            Path(prefix+'.json').write_text(json.dumps(report))
            second=column+2;second.tofile(root/'second.f32');next_prefix=str(root/'qual2')
            spec2=dict(base_prefix=prefix,base_manifest_sha256=digest(prefix+'.json'),
                       column_path=str(root/'second.f32'),column_sha256=digest(root/'second.f32'),
                       insert_at=4,shape=[6,7])
            Path(next_prefix+'.matrix.json').write_text(json.dumps(spec2))
            report2=dict(matrix_layout='column_insertion',composite_spec_sha256=digest(next_prefix+'.matrix.json'))
            overlay2=verified_matrix(next_prefix,report2)
            expected=np.column_stack([physical[:,:4],second,physical[:,4:]])
            np.testing.assert_array_equal(overlay2[:],expected)
            report2.update(state='complete',qualifying_labels_read=False)
            Path(next_prefix+'.json').write_text(json.dumps(report2))
            chosen=[6,0,3];selected_prefix=str(root/'selected')
            selection=dict(base_prefix=next_prefix,base_manifest_sha256=digest(next_prefix+'.json'),columns=chosen,shape=[6,3])
            Path(selected_prefix+'.matrix.json').write_text(json.dumps(selection))
            sr=dict(matrix_layout='column_selection',composite_spec_sha256=digest(selected_prefix+'.matrix.json'))
            selected=verified_matrix(selected_prefix,sr)
            for rows in [slice(None),slice(1,4),np.array([4,0,2]),3]:
                np.testing.assert_array_equal(selected[rows],expected[rows][...,chosen])
            sr.update(state='complete',qualifying_labels_read=False)
            Path(selected_prefix+'.json').write_text(json.dumps(sr))
            np.save(root/'block.npy',second);block_prefix=str(root/'block')
            block=dict(base_prefix=selected_prefix,base_manifest_sha256=digest(selected_prefix+'.json'),
                columns=[dict(path=str(root/'extra.f32'),sha256=digest(root/'extra.f32'),format='f32'),
                         dict(path=str(root/'block.npy'),sha256=digest(root/'block.npy'),format='npy')],
                insert_at=2,shape=[6,5])
            Path(block_prefix+'.matrix.json').write_text(json.dumps(block))
            br=dict(matrix_layout='column_block_insertion',composite_spec_sha256=digest(block_prefix+'.matrix.json'))
            combined=verified_matrix(block_prefix,br)
            physical_block=np.column_stack([expected[:,chosen[:2]],column,second,expected[:,chosen[2:]]])
            for rows in [slice(None),np.array([4,0,2]),3]:
                np.testing.assert_array_equal(combined[rows],physical_block[rows])
            np.save(root/'block.npy',second+1)
            with self.assertRaisesRegex(ValueError,'changed'):verified_matrix(block_prefix,br)
            np.save(root/'base.npy',base+1)
            with self.assertRaisesRegex(ValueError,'changed'):verified_matrix(prefix,report)
            with self.assertRaisesRegex(ValueError,'changed'):verified_matrix(next_prefix,report2)
            with self.assertRaisesRegex(ValueError,'changed'):verified_matrix(selected_prefix,sr)


if __name__=='__main__':unittest.main()
