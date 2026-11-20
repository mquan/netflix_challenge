import unittest
from unittest.mock import patch
import upstream_original1200_exact_build_recovery as r


class ExactCompilerAdmission(unittest.TestCase):
    def test_original_group_reservation_and_caps(self):
        g=2**30;baseline=dict(swap_used_bytes=2666004480)
        current=dict(baseline,compressor_bytes=2*g)
        workers={'94817':dict(bytes=.1*g,rss_bytes=.1*g),'2':dict(bytes=1.9*g,rss_bytes=1.9*g),
            '3':dict(bytes=20*g,rss_bytes=20*g)}
        group={'94817','2'}
        with patch.object(r.original.shutil,'disk_usage',return_value=type('D',(),{'free':20.2*g})()):
            self.assertTrue(r.build_admitted(workers,group,current,baseline))
            self.assertEqual(workers['2']['bytes'],1.9*g)
            self.assertFalse(r.build_admitted(dict(workers,**{'2':dict(bytes=3*g,rss_bytes=1.9*g)}),group,current,baseline))
            self.assertFalse(r.build_admitted(dict(workers,**{'2':dict(bytes=1.9*g,rss_bytes=3*g)}),group,current,baseline))
            self.assertFalse(r.build_admitted(dict(workers,**{'3':dict(bytes=22*g,rss_bytes=20*g)}),group,current,baseline))
            self.assertFalse(r.build_admitted(dict(workers,**{'3':dict(bytes=20*g,rss_bytes=22*g)}),group,current,baseline))
            self.assertFalse(r.build_admitted(workers,group,dict(current,compressor_bytes=5*g),baseline))
            self.assertFalse(r.build_admitted(workers,group,dict(current,swap_used_bytes=baseline['swap_used_bytes']+129*2**20),baseline))
            self.assertFalse(r.build_admitted(workers,set(),current,baseline))
        with patch.object(r.original.shutil,'disk_usage',return_value=type('D',(),{'free':19.9*g})()):
            self.assertFalse(r.build_admitted(workers,group,current,baseline))


if __name__=='__main__':unittest.main()
