"""Historical compression may be diagnosed but cannot admit scientific work."""
import unittest
from unittest.mock import patch
import upstream_own391_h512_readonly_recovery as r


class ReadonlyRecoveryGuards(unittest.TestCase):
    def test_diagnostic_budget_does_not_relax_original_admission(self):
        g=2**30;baseline=dict(swap_used_bytes=2666004480)
        current=dict(baseline,compressor_bytes=8*g)
        workers={'94311':dict(bytes=11*g,rss_bytes=g),'1':dict(bytes=.1*g,rss_bytes=.1*g)}
        with patch.object(r.shutil,'disk_usage',return_value=type('D',(),{'free':20*g})()):
            self.assertTrue(r.diagnostic_safe(workers,current,baseline,94311,1))
            self.assertFalse(r.admission(workers,current,baseline,94311))
            self.assertTrue(r.admission(workers,dict(current,compressor_bytes=4.9*g),baseline,94311))
            self.assertFalse(r.diagnostic_safe(dict(workers,**{'1':dict(bytes=1.1*g,rss_bytes=.1*g)}),current,baseline,94311,1))
            self.assertFalse(r.diagnostic_safe(dict(workers,**{'94311':dict(bytes=17*g,rss_bytes=g)}),current,baseline,94311,1))
            self.assertFalse(r.diagnostic_safe(dict(workers,extra=dict(bytes=13*g,rss_bytes=g)),current,baseline,94311,1))
            self.assertFalse(r.diagnostic_safe(dict(workers,extra=dict(bytes=g,rss_bytes=23*g)),current,baseline,94311,1))
            self.assertFalse(r.diagnostic_safe(workers,dict(current,swap_used_bytes=baseline['swap_used_bytes']+129*2**20),baseline,94311,1))
        with patch.object(r.shutil,'disk_usage',return_value=type('D',(),{'free':18*g})()):
            self.assertFalse(r.diagnostic_safe(workers,current,baseline,94311,1))


if __name__=='__main__':unittest.main()
