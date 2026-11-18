from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_production as p


class ProductionChecks(unittest.TestCase):
    def test_budget_rule_is_per_seed_median_and_rejects_invalid_values(self):
        self.assertEqual(p.budgets([[0,9,4],[1,7,2],[2,6,6],[3,8,8],[4,5,0]],10),[2,7,4])
        for rows in [np.zeros((4,3),dtype=int),np.ones((5,3))*1.5,np.full((5,3),11),np.full((5,3),-1)]:
            with self.assertRaises(ValueError):p.budgets(rows,10)

    def test_zero_correction_is_ridge_and_replay_rejects_wrong_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'model';report=root/'report';data=root/'bank'
            for path in [run/'tree',report,data]:path.mkdir(parents=True)
            w=np.zeros(161);w[-1]=3.1;np.save(run/'ridge.npy',w)
            z=np.zeros((4,161));z[:,-1]=1;np.save(data/'qual-design.npy',z)
            np.save(data/'qual-features.npy',np.zeros((4,221),dtype=np.float32))
            plan=dict(ridge_sha256=p.u.sha(run/'ridge.npy'),seeds=[2026,7,13],iterations={'tree':[0,0,0]})
            p.u.write(report/'plan.json',plan)
            with patch.object(p,'RUN',run),patch.object(p,'REPORT',report),patch.object(p.bank,'RUN',data),patch.object(p,'verify',return_value=plan):
                actual=p.infer('tree',plan,run/'tree')
                self.assertTrue(np.array_equal(actual,np.full(4,3.1,dtype=np.float32)))
                np.full(4,1.5,dtype='<f4').tofile(run/'tree/qualifying.f32')
                p.u.write(report/'tree.json',dict(plan_sha256=p.u.sha(report/'plan.json'),
                    outputs={'qualifying.f32':p.u.sha(run/'tree/qualifying.f32')}))
                with self.assertRaisesRegex(ValueError,'replay differs'):p.worker('tree',True)


if __name__=='__main__':unittest.main()
