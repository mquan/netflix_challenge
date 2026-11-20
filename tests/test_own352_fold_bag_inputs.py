from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_own352_fold_bag as production


class CompleteQualifyingOperators(unittest.TestCase):
    def test_retains_every_parent_column_and_adds_all22_before_intercept(self):
        z=np.arange(5*406,dtype=np.float64).reshape(5,406)
        f=np.arange(5*556,dtype=np.float32).reshape(5,556)
        class Parent:
            def __init__(self,query,rows):
                if query!='qual':raise ValueError('qualifying only')
            def block(self,lo,hi):return z[lo:hi],f[lo:hi]
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[];values=[]
            for j in range(22):
                name='%s.npy'%j;a=np.array([-1,2,j/10,8,4],dtype=np.float32)
                np.save(root/name,a);rows.append(dict(paired_files=dict(qual=name)));values.append(np.clip(a,0,6))
            with patch.object(production,'ROWS',5),patch.object(production.u,'ROOT',root),patch.object(production.base,'Inputs',Parent):
                inputs=production.Inputs([],rows)
                try:
                    actual_z,actual_f=inputs.block(1,5);extra=np.column_stack(values)[1:5]
                    np.testing.assert_array_equal(actual_z[:,:405],z[1:5,:405])
                    np.testing.assert_array_equal(actual_z[:,405:427],extra)
                    np.testing.assert_array_equal(actual_z[:,-1],z[1:5,-1])
                    np.testing.assert_array_equal(actual_f[:,:556],f[1:5])
                    np.testing.assert_array_equal(actual_f[:,556:],extra-np.float32(3.6))
                    self.assertEqual(actual_z.shape,(4,428));self.assertEqual(actual_f.shape,(4,578))
                finally:inputs.close()
                with self.assertRaises(ValueError):production.Inputs([],rows[:-1])


if __name__=='__main__':unittest.main()
