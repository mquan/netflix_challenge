"""Protect training/selection isolation for the new residual bets."""
import array
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
from moonshot_boost import fit_fold
from gbdt_blend import blend_fold
from test_joint import run, fold


def fixture(path, changed=False):
    users, items=100,5
    train=[1+(u+i)%5 for u in range(users) for i in range(3)]
    with path.open('wb') as f:
        f.write(struct.pack('<8sIIQQQHH4xd',b'NFPDAT2\0',users,items,300,200,500,0,4,sum(train)/len(train)))
        def put(code,values):
            values=list(values); f.write(struct.pack('<'+str(len(values))+code,*values))
        put('Q',(u*3 for u in range(users+1)))
        put('H',(1+(u+i)%5 for u in range(users) for i in range(3)))
        put('H',[0,1,2]*users); put('B',train)
        for u in range(users):
            for i in (3,4):
                r=1+(u+i)%5
                if changed and fold(u)==0: r=6-r
                f.write(struct.pack('<IHHB3x',u,1+(u+i)%5,i,r))
        put('Q',(u*5 for u in range(users+1)))
        put('H',(1+(u+i)%5 for u in range(users) for i in range(5)))
        put('H',[0,1,2,3,4]*users)


class BetTests(unittest.TestCase):
    def test_tree_held_labels_do_not_affect_selection_or_refit(self):
        rng=np.random.default_rng(6)
        users=np.repeat(np.arange(1000,dtype=np.uint32),3)
        x=rng.normal(size=(len(users),4)).astype(np.float32)
        y=3+.4*x[:,0]+.1*x[:,1]**2
        changed=y.copy(); changed[blend_fold(users)==0]=6-changed[blend_fold(users)==0]
        design=np.column_stack([np.ones(len(users)),x[:,0]])
        for rx in [None,design]:
            first=fit_fold(x,y,users,0,rounds=8,residual_design=rx)
            second=fit_fold(x,changed,users,0,rounds=8,residual_design=rx)
            np.testing.assert_array_equal(first[0],second[0])
            np.testing.assert_array_equal(first[1],second[1])
            self.assertEqual(first[2],second[2])

    def test_user_kernel_solve_and_label_isolation(self):
        binary=str(pathlib.Path('build/moonshot_user_kernel').resolve())
        subprocess.run([binary,'--self-test'],check=True)
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)
            for name in ['a','b']:
                fixture(root/(name+'.bin'),name=='b')
                run('joint','--data',root/(name+'.bin'),'--model',root/(name+'.model'),'--iterations',1,'--factors',4)
                subprocess.run([binary,'--data',str(root/(name+'.bin')),'--model',str(root/(name+'.model')),
                                '--output',str(root/name),'--features','8'],check=True,capture_output=True)
            for suffix in ['.'+str(i)+'.f32' for i in range(6)]+['.base.f32','.mask.f32']:
                self.assertEqual((root/('a'+suffix)).read_bytes(),(root/('b'+suffix)).read_bytes())
            self.assertNotEqual((root/'a.3.f32').read_bytes(),(root/'a.base.f32').read_bytes())
            self.assertEqual(list(array.array('f',(root/'a.mask.f32').read_bytes())),[1.]*200)


if __name__=='__main__': unittest.main()
