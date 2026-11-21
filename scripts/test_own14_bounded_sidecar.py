"""Fail-closed phase/resource/identity gates; no real process is signalled."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock
import upstream_own14_matched391_bounded_sidecar as run


class SidecarTests(unittest.TestCase):
    def test_native_window_requires_same_early_sequential_first_phase(self):
        native=dict(pid=16010,pgid=16010,started='original start',command='original native',state='Rs')
        header='trainx => probex\n'
        for count in [1,7,21]:
            text=header+''.join('Epoch %02d | RMSE .9\n'%i for i in range(1,count+1))
            self.assertTrue(run.native_window(native,native,text))
        for invalid in ['',header,header+'Epoch 02 | RMSE .9\n',
            header+''.join('Epoch %02d | RMSE .9\n'%i for i in range(1,23)),
            header+'Epoch 01 | RMSE .9\nfulltrain => qual\n']:
            self.assertFalse(run.native_window(native,native,invalid))
        for actual in [None,dict(native,started='reused'),dict(native,state='Ts'),dict(native,pgid=4)]:
            self.assertFalse(run.native_window(native,actual,header+'Epoch 01 | RMSE .9\n'))

    def test_added20_limit_and_original_compressor_swap_disk_limits(self):
        baseline=dict(compressor_bytes=0,swap_used_bytes=1000);workers={'1':dict(bytes=15*2**30,rss_bytes=15*2**30)}
        with mock.patch.object(run.shutil,'disk_usage',return_value=type('D',(),{'free':23*2**30})()):
            self.assertTrue(run.safe(workers,baseline,baseline,2**30))
            self.assertFalse(run.safe(workers,baseline,baseline,5*2**30))
            self.assertFalse(run.safe(workers,dict(baseline,compressor_bytes=5*2**30),baseline))
            self.assertFalse(run.safe(workers,dict(baseline,swap_used_bytes=1001+128*2**20),baseline))

    def test_owned_bootstrap_survives_only_declared_exec(self):
        capture=dict(pid=20,pgid=20,started='same',command='bootstrap',worker_command='worker canonical')
        self.assertTrue(run.same_child(capture,dict(capture,command='worker canonical')))
        self.assertFalse(run.same_child(capture,dict(capture,command='unrelated')))
        self.assertFalse(run.same_child(capture,dict(capture,started='reused')))

    def test_stop_function_never_signals_native_or_unrelated_process(self):
        for captured in [dict(pid=16010,pgid=16010,started='same',command='native'),
                         dict(pid=20,pgid=20,started='same',command='unrelated')]:
            with mock.patch.object(run.identity,'process',return_value=captured),mock.patch.object(run.os,'killpg') as signal:
                with self.assertRaises(ValueError):run.stop_child(captured)
                signal.assert_not_called()

    def test_log_inode_and_original_prefix_fail_closed_on_replacement_or_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/'report';report.mkdir();log=root/'experiments/upstream-original1200/native.log';log.parent.mkdir(parents=True)
            prefix=b'trainx => probex\nEpoch 01 | RMSE .9\n';log.write_bytes(prefix);s=log.stat()
            native=dict(pid=16010,pgid=16010,started='original',command='native',state='Rs')
            (report/'native-ownership.json').write_text(json.dumps(native))
            (report/'native-log-origin.json').write_text(json.dumps(dict(device=s.st_dev,inode=s.st_ino,prefix_bytes=len(prefix),prefix_sha256=hashlib.sha256(prefix).hexdigest())))
            with mock.patch.object(run.u,'ROOT',root),mock.patch.object(run,'REPORT',report),mock.patch.object(run.identity,'process',return_value=native):
                self.assertTrue(run.current_native_window())
                log.write_bytes(prefix+b'Epoch 02 | RMSE .9\n');self.assertTrue(run.current_native_window())
                log.write_bytes(prefix.replace(b'.9',b'.8'));self.assertFalse(run.current_native_window())
                replacement=log.with_name('new.log');replacement.write_bytes(prefix);replacement.replace(log)
                self.assertFalse(run.current_native_window())


if __name__=='__main__':unittest.main()
