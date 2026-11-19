"""Exercise the matching-tree path with actual learned models and held labels."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_cf12_tree as m


class MatchingTreeTest(unittest.TestCase):
    def test_actual_tree_fits_exclude_scoring_labels(self):
        rng=np.random.default_rng(42);rows=200
        features=rng.normal(size=(rows,222)).astype(np.float32)
        features[:,221]=np.linspace(-1,1,rows)
        labels=3.6+.4*features[:,221]
        changed=labels.copy();groups=np.tile(np.arange(10),rows//10);changed[groups//2==0]=100
        z=rng.normal(size=(rows,162));z[:,-1]=1
        cfg=dict(m.CONFIG,seeds=[2026],tree_rounds=5,tree_patience=2,tree_min_leaf=5)
        records=[]
        with tempfile.TemporaryDirectory() as tmp:
            for i,y in enumerate([labels,changed]):
                grams,rhs=m.n.grouped_stats(z,y,groups)
                records.append(m.n.fit_fold(features,z,y,groups,grams,rhs,0,cfg,'tree',Path(tmp)/str(i)))
            self.assertGreater(records[0]['chosen_iterations'][0],0)
            self.assertEqual(records[0]['chosen_iterations'],records[1]['chosen_iterations'])
            for name in ['ridge.npz','prediction.f32','tree-2026.txt']:
                self.assertEqual((Path(tmp)/'0'/name).read_bytes(),(Path(tmp)/'1'/name).read_bytes())


if __name__=='__main__':unittest.main()
