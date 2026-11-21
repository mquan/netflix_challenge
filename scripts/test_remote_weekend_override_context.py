"""Private metadata fixtures: no arrays imported and no native process launched."""
import copy
import json
import unittest
import upstream_remote_weekend_return as original
import verify_upstream_remote_weekend_return as old_verifier
from upstream_remote_weekend_override_context import Context, POLICY


def encoded(value):return (json.dumps(value,indent=2)+'\n').encode()


def fixture(ctx,*,prep_sha=None,binary_sha=None,deadline=None,started=None,omit_side=False,bad_epoch=False,job=None):
    t=ctx.transport;job=t.job_spec('s3-tx300r') if job is None else job;name=job['name'];plan=t.read(t.PLAN)
    prep=json.loads(ctx.preparations[name]);prep_sha=prep_sha or t.digest(ctx.preparations[name])
    trace=list(range(1,job['epochs_per_phase']+1))*2
    if bad_epoch:trace[8]=8
    log=('\n'.join('Epoch %02d | RMSE .9'%i for i in trace)+'\n').encode()
    launch=encoded(dict(job=name,plan_sha256=t.ops.sha(t.PLAN),command=job['command'],environment=job['environment'],
        preparation_sha256=prep_sha,started=started or '2026-11-21T17:55:00Z',deadline=deadline or plan['training_deadline_utc']))
    outputs={p:'a'*64 for p in t.native_shapes(job)}
    if omit_side:del outputs['preds_3/'+name+'.fulltrain.npy']
    outputs['preds_3/'+name+'.out']=t.digest(log)
    receipt=encoded(dict(state='native_complete_pending_export',observed='2026-11-21T20:00:00Z',job=name,
        outputs=outputs,log_sha256=t.digest(log),launch_sha256=t.digest(launch)))
    files={column+'.npz':'b'*64 for column in job['published_columns']}
    files.update({'plan.json':t.ops.sha(t.PLAN),name+'-receipt.json':t.digest(receipt),name+'-launch.json':t.digest(launch),
        name+'.log':t.digest(log),'preparation.json':prep_sha,'build.log':'c'*64,'split3.log':'c'*64,
        'upstream_remote_weekend.py':plan['controller_sha256']['scripts/upstream_remote_weekend.py']})
    source={'source/'+p:s for p,s in plan['source_files_sha256'].items()}
    source.update({'source/'+p:s for p,s in prep['binary_sha256'].items()})
    if binary_sha:source['source/target/release/fit']=binary_sha
    proof=dict(state='verified_complete_original_remote_weekend_producer',job=name,source_pin=plan['source_pin'],
        source_data_binary_rehashed=True,both_full_original_epoch_traces=True,independent_known_projection=True,
        qualifying_labels_read=False,public_prediction_values_read=False,
        all_native_output_sha256={'native/'+p:s for p,s in outputs.items()},original_source_and_binary_sha256=source,
        files_sha256=files,records=[dict(column=c,file=c+'.npz',sha256=files[c+'.npz'],
            native_probe_sha256=outputs['preds_3/'+c+'.probe_3.npy'],native_qual_sha256=outputs['preds_3/'+c+'.qual.npy'])
            for c in job['published_columns']])
    export=encoded(proof);expected={**files,**proof['all_native_output_sha256'],**source,'verification.json':t.digest(export)}
    index=encoded(dict(state='complete_all_original_native_outputs_and_all_epoch_pairs_lossless_return',job=name,
        plan_sha256=t.ops.sha(t.PLAN),qualifying_labels_read=False,raw_dataset_uploaded=False,all_original_files_retained=True,
        source_verification_sha256=t.digest(export),parts=[dict(file=name+'-part01.tar.gz',bytes=100,
            sha256='f'*64,members_sha256=expected)]))
    transfer=encoded(dict(job=name,release_tag=t.TAG,release_draft=True,numeric_arrays_in_git=False,
        assets=[dict(name=name+'-part01.tar.gz',bytes=100,sha256='f'*64),
                dict(name=name+'-index.json',bytes=len(index),sha256=t.digest(index))]))
    return job,{'archive-index.json':index,'export-verification.json':export,name+'-receipt.json':receipt,
                name+'-launch.json':launch,name+'.log':log,'transfer.json':transfer}


class VariantContext(unittest.TestCase):
    def setUp(self):self.ctx=Context()

    def test_original_modules_and_scientific_bytecode_unchanged(self):
        self.assertNotEqual(self.ctx.transport.PLAN,original.PLAN)
        self.assertEqual(original.PLAN,original.u.ROOT/'experiments/upstream-remote-weekend/plan.json')
        self.assertIs(old_verifier.transport,original)
        self.assertIs(self.ctx.verifier.transport,self.ctx.transport)
        self.assertEqual(self.ctx.verifier.verify.__code__.co_code,old_verifier.verify.__code__.co_code)
        self.assertEqual(sum(len(j['published_columns']) for j in self.ctx.transport.read(self.ctx.transport.PLAN)['jobs']),67)

    def test_complete_new_plan_metadata(self):
        job,blobs=fixture(self.ctx)
        self.ctx.transport.validate_metadata(job,blobs)

    def test_coherently_rehashed_wrong_preparation_binary_or_deadline(self):
        for kwargs in ({'prep_sha':'c'*64},{'binary_sha':'d'*64},{'deadline':'2026-11-24T14:00:00Z'},
                       {'started':'2026-11-20T17:00:00Z'}):
            with self.subTest(kwargs=kwargs):
                job,blobs=fixture(self.ctx,**kwargs)
                self.ctx.base_validate_metadata(job,blobs)  # Coherent old metadata alone is insufficient.
                with self.assertRaises(ValueError):self.ctx.transport.validate_metadata(job,blobs)

    def test_full_traces_and_native_side_outputs_remain_required(self):
        for kwargs in ({'omit_side':True},{'bad_epoch':True}):
            with self.subTest(kwargs=kwargs):
                job,blobs=fixture(self.ctx,**kwargs)
                with self.assertRaises(ValueError):self.ctx.transport.validate_metadata(job,blobs)

    def test_caller_cannot_trim_the_declared_job(self):
        job=copy.deepcopy(self.ctx.transport.job_spec('s3-tx300r'));job['published_columns'].pop()
        job,blobs=fixture(self.ctx,job=job)
        self.ctx.base_validate_metadata(job,blobs)
        with self.assertRaises(ValueError):self.ctx.transport.validate_metadata(job,blobs)

    def test_pinned_lineage_and_source_identity(self):
        self.ctx.require_revision('444fa334fede478f1724162e81d6080d4bb1bf21')
        for commit in ('3d9a2a0da8692f01479927e716c32976938f8695','not-a-commit'):
            with self.subTest(commit=commit),self.assertRaises(ValueError):self.ctx.require_revision(commit)
        self.ctx.bindings['scripts/upstream_remote_weekend_return.py']='0'*64
        with self.assertRaises(ValueError):self.ctx.check_sources()


if __name__=='__main__':unittest.main()
