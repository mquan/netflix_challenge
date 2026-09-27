"""Timing provenance, score parsing and overlap-safe active-time accounting."""
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import experiment
import score_timeline as timeline


class TimelineTests(unittest.TestCase):
    def test_capture_has_timezone_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                code = experiment.capture_output(
                    [sys.executable, '-c', 'print("epoch=1 probe_rmse=0.9")'], p/'run.log', p/'run.events.jsonl')
            self.assertEqual(code, 0)
            self.assertEqual((p/'run.log').read_text(), 'epoch=1 probe_rmse=0.9\n')
            event = json.loads((p/'run.events.jsonl').read_text())
            self.assertIsNotNone(timeline.parse_time(event['observed_utc']).tzinfo)
            self.assertEqual(timeline.scores(event['line']), [('single_model_probe', .9, '1')])

    def test_legacy_proxy_frozen_and_only_best_epoch_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder)
            content = b'epoch=1 probe_rmse=0.92\nepoch=2 probe_rmse=0.90\nepoch=3 probe_rmse=0.91\n'
            record = dict(name='J-test', command=['test'], started_utc='2020-01-01T00:00:00Z',
                          exit_code=0, strategy='test', log_sha256=hashlib.sha256(content).hexdigest())
            (p/'J-test.json').write_text(json.dumps(record))
            (p/'J-test.log').write_bytes(content)
            proxies = {}
            rows = timeline.experiment_rows(p, proxies)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['epoch'], '2')
            self.assertEqual(rows[0]['score'], .90)
            self.assertEqual(rows[0]['timestamp_basis'], 'log_mtime_proxy_upper_bound')
            (p/'J-test.log').touch()
            self.assertEqual(timeline.experiment_rows(p, proxies), rows)
            (p/'J-test.log').write_text('epoch=4 probe_rmse=0.89\n')
            with self.assertRaises(ValueError):
                timeline.experiment_rows(p, proxies)

    def test_event_improvements_and_running_legacy_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder)
            record = dict(name='J-test', command=['test'], started_utc='2020-01-01T00:00:00Z', strategy='test')
            (p/'J-test.json').write_text(json.dumps(record))
            (p/'J-test.log').write_text('epoch=1 probe_rmse=0.92\n')
            self.assertEqual(timeline.experiment_rows(p, {}), [])
            events = [dict(line='epoch=%d probe_rmse=%s' % (i, score), observed_utc='2020-01-01T00:00:0%dZ' % i)
                      for i, score in enumerate((.92, .93, .90), 1)]
            (p/'J-test.events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events)+'{"partial":')
            rows = timeline.experiment_rows(p, {})
            self.assertEqual([r['epoch'] for r in rows], ['1', '3'])
            self.assertTrue(all(r['timestamp_basis'] == 'stdout_observed' for r in rows))

    def test_metric_separation(self):
        self.assertEqual(timeline.scores('column=a probe_rmse=0.8'), [])
        self.assertEqual(timeline.scores('blend_cv_rmse=0.89 blend_fit_rmse=0.88'),
                         [('blend_probe_user_cv', .89, ''), ('blend_probe_fit', .88, '')])
        rows = [dict(timestamp_utc='2020-01-01T00:00:00Z', score=.88, metric='blend_probe_fit'),
                dict(timestamp_utc='2020-01-01T01:00:00Z', score=.89, metric='blend_probe_user_cv')]
        timeline.decorate(rows, dt.timezone.utc)
        self.assertEqual(rows[1]['best_so_far_in_metric'], '0.8900000000')
        self.assertEqual(rows[1]['activity_coverage'], 'partial')
        self.assertEqual(float(rows[1]['unclassified_seconds']), 3600)

    def test_overlapping_pauses_work_precedence_and_clipping(self):
        start = timeline.parse_time('2020-01-01T00:00:00Z')
        def t(seconds):
            return start+dt.timedelta(seconds=seconds)
        intervals = [(t(-10), t(20), 'idle'), (t(10), t(30), 'rate_limit'),
                     (t(25), t(40), 'working'), (t(35), t(45), 'working'),
                     (t(55), t(100), 'timeout')]
        # Pause union 0..30 and 55..60, minus useful work 25..45.
        self.assertEqual(timeline.activity_totals(start, t(60), intervals), [20, 30, 10])

    def test_invalid_activity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder)/'activity.csv'
            path.write_text('start_utc,end_utc,state,source\n2020-01-02T00:00:00Z,2020-01-01T00:00:00Z,idle,test\n')
            with self.assertRaises(ValueError):
                timeline.load_activity(path)


if __name__ == '__main__':
    unittest.main()
