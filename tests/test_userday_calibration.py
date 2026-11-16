import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from userday_calibration import statistics,leave_one_out,corrections


class UserDayTests(unittest.TestCase):
    def test_held_rating_cannot_change_its_own_correction(self):
        users=np.array([0,0,0,1],dtype='u4');days=np.array([5,5,6,8],dtype='u2')
        r=np.array([1.,2.,-1.,3.]);s,g=statistics(users,days,r,3)
        before=leave_one_out(users,r,g,s,20,30)
        r[0]=1000;s,g=statistics(users,days,r,3)
        after=leave_one_out(users,r,g,s,20,30)
        self.assertAlmostEqual(before[0],after[0],places=12)
        self.assertNotEqual(before[1],after[1])
        self.assertEqual(before[3],0)

    def test_new_day_uses_user_prior_and_unseen_user_stays_unchanged(self):
        u=np.array([0,0],dtype='u4');day=np.array([5,5],dtype='u2');s,_=statistics(u,day,np.array([1.,3.]),2)
        c=corrections(np.array([0,1],dtype='u4'),np.array([6,5],dtype='u2'),s,20,30)
        self.assertAlmostEqual(c[0],4/22)
        self.assertEqual(c[1],0)


if __name__=='__main__':unittest.main()
