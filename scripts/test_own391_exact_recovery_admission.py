"""Admission credits preserved RAM without weakening any original guard."""
import unittest
from unittest.mock import patch
import upstream_own391_neural_exact_recovery as r
import upstream_own391_neural_original_admission as revised


class Admission(unittest.TestCase):
    def test_revised_admission_keeps_original_five_GiB_cap(self):
        g=2**30;baseline=dict(swap_used_bytes=2666004480)
        workers={'71312':dict(bytes=11*g,rss_bytes=7*g)}
        with patch.object(revised.original.shutil,'disk_usage',return_value=type('D',(),{'free':21*g})()):
            self.assertTrue(revised.admission(workers,dict(baseline,compressor_bytes=4.99*g),baseline,71312))
            self.assertFalse(revised.admission(workers,dict(baseline,compressor_bytes=5*g),baseline,71312))
            self.assertFalse(revised.admission({'71312':dict(bytes=17*g,rss_bytes=7*g)},dict(baseline,compressor_bytes=g),baseline,71312))

    def test_original_caps_and_preserved_reservation(self):
        g=2**30;baseline=dict(swap_used_bytes=2666004480)
        current=dict(baseline,compressor_bytes=2*g)
        workers={'71312':dict(bytes=6*g,rss_bytes=7*g),'10':dict(bytes=g,rss_bytes=g)}
        with patch.object(r.original.shutil,'disk_usage',return_value=type('D',(),{'free':21*g})()):
            self.assertTrue(r.admission(workers,current,baseline,71312))
            self.assertEqual(workers['71312'],dict(bytes=6*g,rss_bytes=7*g))
            for change in [dict(compressor_bytes=5*g),dict(swap_used_bytes=baseline['swap_used_bytes']+129*2**20)]:
                self.assertFalse(r.admission(workers,dict(current,**change),baseline,71312))
            for extra in [dict(bytes=7*g,rss_bytes=g),dict(bytes=g,rss_bytes=7*g)]:
                self.assertFalse(r.admission(dict(workers,extra=extra),current,baseline,71312))
            for own in [dict(bytes=17*g,rss_bytes=7*g),dict(bytes=6*g,rss_bytes=17*g)]:
                self.assertFalse(r.admission(dict(workers,**{'71312':own}),current,baseline,71312))
            self.assertFalse(r.admission(workers,current,baseline,123))
        with patch.object(r.original.shutil,'disk_usage',return_value=type('D',(),{'free':18*g})()):
            self.assertFalse(r.admission(workers,current,baseline,71312))


if __name__=='__main__':unittest.main()
