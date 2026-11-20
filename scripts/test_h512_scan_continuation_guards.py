import unittest
from unittest.mock import patch
import upstream_own391_h512_scan_continuation as r


class ScanContinuationGuards(unittest.TestCase):
    def test_credit_existing_scanner_with_original_total_caps(self):
        g=2**30;baseline=dict(swap_used_bytes=2666004480)
        current=dict(baseline,compressor_bytes=8*g)
        workers={'94311':dict(bytes=10*g,rss_bytes=g),'230':dict(bytes=5*g,rss_bytes=5*g),
            '1':dict(bytes=g,rss_bytes=g)}
        with patch.object(r.shutil,'disk_usage',return_value=type('D',(),{'free':20*g})()):
            self.assertTrue(r.diagnostic_safe(workers,current,baseline,94311,230,reserve_helper=True))
            self.assertEqual(workers['230'],dict(bytes=5*g,rss_bytes=5*g))
            self.assertFalse(r.admission(workers,current,baseline,94311))
            self.assertFalse(r.diagnostic_safe(dict(workers,**{'230':dict(bytes=10.1*g,rss_bytes=5*g)}),current,baseline,94311,230))
            self.assertFalse(r.diagnostic_safe(dict(workers,**{'94311':dict(bytes=17*g,rss_bytes=g)}),current,baseline,94311,230))
            self.assertFalse(r.diagnostic_safe(dict(workers,extra=dict(bytes=4*g,rss_bytes=g)),current,baseline,94311,230,reserve_helper=True))
            self.assertFalse(r.diagnostic_safe(dict(workers,extra=dict(bytes=g,rss_bytes=13*g)),current,baseline,94311,230,reserve_helper=True))
            self.assertFalse(r.diagnostic_safe(workers,dict(current,swap_used_bytes=baseline['swap_used_bytes']+129*2**20),baseline,94311,230))
        with patch.object(r.shutil,'disk_usage',return_value=type('D',(),{'free':18*g})()):
            self.assertFalse(r.diagnostic_safe(workers,current,baseline,94311,230))


if __name__=='__main__':unittest.main()
