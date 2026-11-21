"""Full producer proof and exact terminal-handle gate before same-audit continuation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import upstream_own14_matched391_post_native_audit as run


class DeferredAuditTests(unittest.TestCase):
    def test_full23_23_all58_original_proof_and_terminal_handle_required(self):
        original=run.u.read(run.u.ROOT/'experiments/upstream-original1200/execution-plan.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir()
            native=dict(pid=16010,pgid=16010,started='original',command='native',state='Rs')
            (old/'native-ownership.json').write_text(json.dumps(native))
            report=root/'experiments/upstream-original1200-exact-build-recovery';report.mkdir(parents=True)
            producer=root/'experiments/upstream-original1200';producer.mkdir()
            (producer/'execution-plan.json').write_text(json.dumps(original))
            proof=producer/'independent-verification.json';receipt=report/'completion.json'
            prefix='artifacts/upstream-original1200/source/preds_new/'
            outputs={prefix+row['name']:'a'*64 for row in original['exact_expected_outputs']}
            outputs[prefix+'tsvdx5-1200lm.out']='b'*64
            self.assertEqual(len(outputs),59)
            value=dict(state='complete_independent_original1200_both23_phases_and_all58_numeric_outputs',epochs_each=23,
                factors=1200,neighbors=100,outputs=outputs,qualifying_labels_read=False)
            with mock.patch.object(run.u,'ROOT',root),mock.patch.object(run,'OLD',old),mock.patch.object(run.identity,'process',return_value=None) as process:
                self.assertFalse(run.native_prerequisite())
                proof.write_text(json.dumps(value));receipt.write_text(json.dumps(dict(state='complete_original1200_full23_23_and_independent58_output_verification',proof_sha256=run.u.sha(proof))))
                self.assertTrue(run.native_prerequisite())
                process.return_value=native;self.assertFalse(run.native_prerequisite());process.return_value=None
                missing_log={k:v for k,v in outputs.items() if not k.endswith('.out')}
                missing_numeric=dict(outputs);missing_numeric.pop(prefix+original['exact_expected_outputs'][0]['name'])
                for changed in [dict(value,epochs_each=22),dict(value,outputs={}),dict(value,outputs=missing_log),
                    dict(value,outputs=missing_numeric),dict(value,neighbors=0),dict(value,qualifying_labels_read=True)]:
                    proof.write_text(json.dumps(changed));receipt.write_text(json.dumps(dict(state='complete_original1200_full23_23_and_independent58_output_verification',proof_sha256=run.u.sha(proof))))
                    self.assertFalse(run.native_prerequisite())

    def test_actual_audit_exec_identity_survives_bootstrap_transition_only(self):
        captured=dict(pid=26049,pgid=26049,started='original',command='bootstrap',worker_command='original audit')
        self.assertTrue(run.same_child(captured,dict(captured,command='original audit')))
        self.assertFalse(run.same_child(captured,dict(captured,started='reused')))


if __name__=='__main__':unittest.main()
