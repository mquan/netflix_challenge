#!/usr/bin/env python3
"""Self-stop the exact supervised return worker before losing its state."""
import datetime as dt
import os
import shutil
import signal
import time
import upstream_repro as u
import upstream_cf96_primary_lane as identity
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory


class Guard:
    def __init__(self, job):
        self.pid = os.getpid(); self.report = u.ROOT/'experiments/upstream-remote-weekend-return-queue'/job
        owner, child = [u.read(self.report/name) for name in ['monitor.json', 'child-ownership.json']]
        actual = identity.process(self.pid)
        if (os.getpgid(self.pid) != self.pid or child['pid'] != self.pid or actual is None
            or child['started'] != actual['started'] or child.get('worker_command') != actual['command']
            or not identity.same_owner(owner, identity.process(owner['pid']))):
            raise ValueError('Exact supervised dedicated return worker required before allocation')
        self.baseline = u.read(self.report/'first-scientific-baseline.json'); self.last = 0; self.check()

    def check(self):
        # Hashing/streaming calls frequently; the outer supervisor also samples.
        # Cheap disk checks remain per block, full process accounting every3s.
        if time.monotonic()-self.last < 3 and shutil.disk_usage(u.ROOT).free >= 18.75*2**30:
            return
        while True:
            sample = dict(pid=self.pid, original_baseline=self.baseline, observed=dt.datetime.now(dt.timezone.utc).isoformat())
            try:
                workers, current = resources.footprints(self.pid), memory.memory(); own = workers[str(self.pid)]
                valid = (max(own['bytes'], own['rss_bytes']) <= 2*2**30
                    and sum(w['bytes'] for w in workers.values()) < 24*2**30
                    and sum(w['rss_bytes'] for w in workers.values()) < 24*2**30
                    and current['compressor_bytes'] < 5*2**30
                    and current['swap_used_bytes']-self.baseline['swap_used_bytes'] <= 128*2**20
                    and shutil.disk_usage(u.ROOT).free >= 18.75*2**30)
                if valid:
                    self.last = time.monotonic(); return
                sample.update(workers=workers, memory=current)
            except Exception as error: sample['measurement_error'] = repr(error)
            try: u.write(self.report/('self-guard-%s.json'%self.pid), sample)
            except Exception: pass
            os.killpg(self.pid, signal.SIGSTOP)
            # Explicit same-process continuation rechecks the SAME baseline.
