import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gbdt_blend import HEADER, PROBE


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.binary=Path(cls.tmp.name)/'splitmap'
        subprocess.run(['clang++','-O2','-std=c++17',str(ROOT/'src/public_split3_map.cpp'),'-o',str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_upstream_rng_reference_vectors(self):
        subprocess.run([str(self.binary),'--self-test'],check=True,capture_output=True)

    def test_original_queries_preserved_and_all_labels_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.array([99,2,77,5],dtype='<u4').tofile(root/'raw.u32')
            outputs=[]
            for variant in range(2):
                h=np.zeros(1,dtype=HEADER)
                for key,value in dict(magic=b'NFPDAT2',users=4,items=5,train_count=16,
                                      probe_count=4,implicit_count=0,min_day=1,max_day=3,mean=3.6).items(): h[key]=value
                q=np.zeros(4,dtype=PROBE)
                q['user']=np.arange(4); q['item']=5; q['day']=3; q['rating']=1+variant*4
                with (root/f'{variant}.bin').open('wb') as f:
                    h.tofile(f)
                    np.arange(0,17,4,dtype='<u8').tofile(f)
                    np.tile(np.arange(1,5,dtype='<u2'),4).tofile(f)
                    np.tile(np.array([1,1,2,2],dtype='<u2'),4).tofile(f)
                    np.full(16,1+variant*4,dtype='u1').tofile(f)
                    q.tofile(f)
                    np.zeros(5,dtype='<u8').tofile(f)
                prefix=root/f'map{variant}'
                subprocess.run([str(self.binary),str(root/f'{variant}.bin'),str(root/'raw.u32'),str(prefix)],check=True,capture_output=True)
                outputs.append({suffix:Path(str(prefix)+suffix).read_bytes() for suffix in
                                ('.u32','.moved.u32','.source-items.u32','.source-users.u32')})
                np.testing.assert_array_equal(np.fromfile(str(prefix)+'.u32',dtype='<u4'),[4,1,3,2])
                self.assertEqual(len(np.fromfile(str(prefix)+'.moved.u32',dtype='<u4')),1)
            self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__': unittest.main()
