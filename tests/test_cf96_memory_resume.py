import unittest
import upstream_cf96_memory_resume as r


class MemoryAdmissionTest(unittest.TestCase):
    def test_compression_and_swap_growth_override_low_resident_memory(self):
        vm='Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages occupied by compressor: 1048576.\n'
        observed=r.parse_memory(vm,'vm.swapusage: total = 7168.00M used = 6040.56M free = 1127.44M')
        self.assertEqual(observed['compressor_bytes'],16*2**30)
        self.assertFalse(r.admitted(observed,observed,1*2**30))
        good=dict(compressor_bytes=5*2**30,swap_used_bytes=4*2**30)
        self.assertTrue(r.admitted(good,good,10*2**30))
        growing=dict(good,swap_used_bytes=good['swap_used_bytes']+100*2**20)
        self.assertFalse(r.admitted(good,growing,1*2**30))
        self.assertFalse(r.admitted(good,good,16*2**30))


if __name__=='__main__':unittest.main()
