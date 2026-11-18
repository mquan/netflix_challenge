import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import upstream_mapped_prepare as m
import upstream_mapped_job as j


class MappedChecks(unittest.TestCase):
    def test_patch_anchors_reject_ambiguous_or_missing_source(self):
        self.assertEqual(m.replace_once('abc','b','X'),'aXc')
        for text in ['ac','abbc']:
            with self.assertRaises(ValueError):m.replace_once(text,'b','X')

    def test_layout_does_not_accept_another_data_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'candidate';fresh=root/'fresh'
            (tree/'data').mkdir(parents=True);(fresh/'data').mkdir(parents=True)
            (tree/'data/trainx').symlink_to(root/'unrelated')
            with patch.object(m,'TREE',tree),patch.object(j.u,'FRESH',fresh):
                with self.assertRaisesRegex(ValueError,'Wrong candidate native data'):j.layout()

    def test_finished_regular_worker_is_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'experiments/full-upstream-reproduction/status.json'
            j.u.write(path,dict(state='training',child_pid=123))
            with patch.object(j.u,'ROOT',root),patch.object(j.subprocess,'run') as run,patch.object(j,'group_rss') as rss:
                run.return_value.stdout='unrelated program'
                self.assertEqual(j.regular_rss(),0);rss.assert_not_called()

    def test_layout_satisfies_native_ingest_setup_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'candidate';fresh=root/'fresh'
            (tree/'data').mkdir(parents=True)
            for name in ['train','probe','trainx','probex','fulltrain','qual']:
                (fresh/'data'/name).mkdir(parents=True)
            for name in ['README','movie_titles.csv','probe.txt','qualifying.txt']:
                path=fresh/'data/raw'/name;path.parent.mkdir(exist_ok=True);path.write_text('input')
            with patch.object(m,'TREE',tree),patch.object(m,'BASE',root/'base'),patch.object(j.u,'FRESH',fresh):
                j.layout()
                for name in ['train','probe','fulltrain','qual','raw/README','raw/movie_titles.csv','raw/probe.txt','raw/qualifying.txt']:
                    self.assertTrue((tree/'data'/name).exists(),name)


if __name__=='__main__':unittest.main()
