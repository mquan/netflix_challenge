import unittest
from unittest.mock import patch
import upstream_qualifying_backfill as b


class BackfillChecks(unittest.TestCase):
    def test_other_memory_counts_each_owned_training_group_once(self):
        listing='''10 1 10 100 python scripts/upstream_repro.py --run-job mf-60
11 10 10 200 ./target/release/mf-new mf-60
20 1 20 300 ./target/release/tsvdx5-new tsvdx5-150o
30 1 30 400 python upstream_voting_blend.py --worker tree
40 1 40 999999 unrelated app
'''
        with patch.object(b.subprocess,'check_output',return_value=listing),patch.object(b.os,'getpgrp',return_value=99):
            self.assertEqual(b.other_training_rss(),1000*1024)


if __name__=='__main__':unittest.main()
