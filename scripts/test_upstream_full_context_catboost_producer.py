"""Producer checks run in isolated children; parent remains Torch-free."""
from pathlib import Path
import subprocess
import sys
import unittest


class ProducerTests(unittest.TestCase):
    def child(self, body):
        directory = str(Path(__file__).resolve().parent)
        code = "import sys\nsys.path.insert(0, sys.argv[1])\nimport numpy as np\nimport upstream_full_context_catboost_producer as m\n" + body
        result = subprocess.run([sys.executable, '-c', code, directory], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ten_full_width_solves_exclude_outer_and_inner_outcomes_exactly(self):
        self.child('''
for d in (562,563):
    gram=np.stack([np.eye(d)*(i+1.) for i in range(10)])
    rhs=np.stack([np.linspace(1.,2.,d)*(i+1.) for i in range(10)])
    result=m.eligible_weights(gram,rhs,lambda:None)
    assert result.shape==(5,2,d) and result.dtype==np.float64
    for fold in range(5):
        outer=np.arange(10)//2!=fold
        for pop,eligible in enumerate((outer&(np.arange(10)%2==0),outer)):
            reg=np.eye(d)*1000.;reg[-1,-1]=0
            lhs=gram[eligible].sum(0)+reg;target=rhs[eligible].sum(0)
            assert np.max(np.abs(lhs@result[fold,pop]-target))<1e-12
    changed=rhs.copy();changed[4:6]=1e30
    other=m.eligible_weights(gram,changed,lambda:None)
    assert result[2].tobytes()==other[2].tobytes()
    assert result[0].tobytes()!=other[0].tobytes()
    changed=rhs.copy();changed[1::2]=1e30
    inner=m.eligible_weights(gram,changed,lambda:None)
    assert result[:,0].tobytes()==inner[:,0].tobytes()
assert 'torch' in sys.modules
''')

    def test_incomplete_CPU_gate_prevents_source_scan_or_stream(self):
        self.child('''
from unittest import mock
with mock.patch.object(m.inputs.cpu,'read',return_value={'state':'pending','qualifying_labels_read':False}), mock.patch.object(m.inputs,'sources',side_effect=AssertionError('source scan started')):
    try:m.prepare('scalar',lambda:None)
    except ValueError:pass
    else:raise AssertionError('incomplete CPU proof admitted')
for arm in ('partial','qualifying',None):
    try:m.prepare(arm,lambda:None)
    except ValueError:pass
    else:raise AssertionError('wrong arm admitted')
''')

    def test_invalid_sufficient_statistics_rejected_and_guard_preserved(self):
        self.child('''
gram=np.zeros((10,562,562));rhs=np.zeros((10,562))
for g,r in [(gram[:9],rhs),(gram,rhs.astype('f4')),(gram[:,:,:561],rhs)]:
    try:m.eligible_weights(g,r,lambda:None)
    except ValueError:pass
    else:raise AssertionError('partial statistics admitted')
rhs[0,0]=np.nan
try:m.eligible_weights(gram,rhs,lambda:None)
except ValueError:pass
else:raise AssertionError('nonfinite statistic admitted')
rhs[0,0]=0
try:m.eligible_weights(gram,rhs,lambda:(_ for _ in ()).throw(RuntimeError('same-frame guard')))
except RuntimeError as e:assert 'same-frame' in str(e)
else:raise AssertionError('guard bypassed')
''')


if __name__ == '__main__': unittest.main()
