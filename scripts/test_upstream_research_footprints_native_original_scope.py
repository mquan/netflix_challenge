"""Original-scope candidate; no changes or signals to production workers."""
from pathlib import Path
import unittest
from unittest.mock import patch
import upstream_research_footprints_native_original_scope as candidate


class OriginalScopeTests(unittest.TestCase):
    def test_exact_original_scope_excludes_plain_instrumentation_not_science(self):
        commands={1:'unrelated',2:'ps -axo pid=,ppid=,stat=,rss=,lstart=,command=',
            3:'python upstream_a.py',4:'python verify_upstream_a.py',
            5:'/a/target/release/model',6:str(candidate.ROOT)+'/scripts/model.py',
            7:'python /elsewhere/upstream_model.py',8:'other worker'}
        rows={pid:dict(command=command,ppid=3) for pid,command in commands.items()}
        expected=set()
        # Same predicate as the original vmmap observer, before descendants
        # were added by the separately versioned first native candidate.
        for pid,row in rows.items():
            cmd=row['command'];tokens=cmd.split()
            native='/target/release/' in tokens[0]
            script=any(Path(t).name.startswith('upstream_') or Path(t).name.startswith('verify_upstream_') for t in tokens[1:])
            if pid==1 or native or str(candidate.ROOT) in cmd or script:expected.add(pid)
        self.assertEqual(candidate.selected(rows,1),expected)
        self.assertEqual(expected,{1,3,4,5,6,7})

    def test_transient_observation_race_resamples_but_permission_never_becomes_zero(self):
        with patch.object(candidate,'sample',side_effect=[ProcessLookupError('gone'),{'7':dict(bytes=10,rss_bytes=20)}]) as sample,patch.object(candidate.time,'sleep'):
            self.assertEqual(candidate.footprints(7),{'7':dict(bytes=10,rss_bytes=20)})
            self.assertEqual(sample.call_count,2)
        with patch.object(candidate,'sample',side_effect=PermissionError('denied')) as sample,patch.object(candidate.time,'sleep'):
            with self.assertRaises(PermissionError):candidate.footprints(7)
            self.assertEqual(sample.call_count,3)

    def test_stopped_protected_os_observer_is_never_queried(self):
        rows={1:dict(command='python upstream_work.py',ppid=99,started='same',rss_bytes=100),
              2:dict(command='ps -axo pid=,ppid=,stat=,rss=,lstart=,command=',ppid=1,started='same',rss_bytes=500)}
        with patch.object(candidate.native,'processes',side_effect=[rows,rows]),patch.object(candidate.native,'usage',return_value=dict(bytes=300,rss_bytes=200,started=7)) as usage:
            sample=candidate.sample(1)
            self.assertEqual(set(sample),{'1'})
            self.assertEqual([call.args for call in usage.call_args_list],[(1,),(1,)])

    def test_reused_scientific_process_fails(self):
        rows={1:dict(command='python upstream_work.py',ppid=99,started='same',rss_bytes=100)}
        with patch.object(candidate.native,'processes',side_effect=[rows,rows]),patch.object(candidate.native,'usage',side_effect=[dict(bytes=300,rss_bytes=200,started=7),dict(bytes=300,rss_bytes=200,started=8)]):
            with self.assertRaises(RuntimeError):candidate.sample(1)


if __name__=='__main__':unittest.main()
