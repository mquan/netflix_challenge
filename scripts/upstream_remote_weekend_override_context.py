#!/usr/bin/env python3
"""Isolate the original full scientific validators for the pinned policy variant.

Library only: this does not launch a worker or admit any numerical return.
The supervised adapter must bind this configuration in its own declaration.
"""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import upstream_remote_weekend_override_policy as policy

ROOT=policy.ROOT
POLICY=ROOT/'experiments/upstream-remote-weekend-owner-override-policy'
BASE_SOURCES={
    'scripts/upstream_remote_weekend_return.py':'5d04c8c093832b0421adf4c13c12740e7025f9b6a085b54ef5681706a9351f6e',
    'scripts/verify_upstream_remote_weekend_return.py':'44609cb7563e76ba57a423e4b887bed5b1004f8491515a6c5a111e8f4ea51ace',
}


def timestamp(value):
    result=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.utcoffset() is None:raise ValueError('Timezone-qualified original timestamps required')
    return result


def isolated_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Context:
    """Fresh module namespaces; original imported modules/globals stay unchanged."""
    def __init__(self):
        evidence=json.loads((POLICY/'verification.json').read_text())
        if evidence['state']!='verified_pinned_remote_resource_policy_variant_metadata_only':
            raise ValueError('Previously checked owner-override policy required')
        self.bindings=dict(evidence['source_sha256'])
        self.bindings.update({str((POLICY/name).relative_to(ROOT)):digest
                              for name,digest in evidence['snapshots_sha256'].items()})
        self.bindings.update(BASE_SOURCES)
        self.bindings[str((POLICY/'verification.json').relative_to(ROOT))]=policy.sha((POLICY/'verification.json').read_bytes())
        self.bindings[str(Path(__file__).relative_to(ROOT))]=policy.sha(Path(__file__).read_bytes())
        self.check_sources()
        self.preparations={job:(POLICY/(job+'-preparation.json')).read_bytes() for job in policy.JOBS}
        policy.validate((ROOT/policy.PLAN_PATH).read_bytes(),(ROOT/policy.CODE_PATH).read_bytes(),
                        (POLICY/'remote-plan.json').read_bytes(),(POLICY/'remote-controller.py').read_bytes(),self.preparations)
        self.transport=isolated_module('_weekend_override_transport',ROOT/'scripts/upstream_remote_weekend_return.py')
        self.transport.PLAN=POLICY/'remote-plan.json'
        self.base_validate_metadata=self.transport.validate_metadata
        self.base_metadata=self.transport.metadata
        self.transport.validate_metadata=self.validate_metadata
        self.transport.metadata=self.metadata
        self.verifier=isolated_module('_weekend_override_verifier',ROOT/'scripts/verify_upstream_remote_weekend_return.py')
        self.verifier.transport=self.transport

    def check_sources(self):
        for name,expected in self.bindings.items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or policy.sha((ROOT/p).read_bytes())!=expected:
                raise ValueError('Pinned override context source changed: '+name)

    def require_revision(self,commit):
        self.check_sources()
        if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full remote commit required')
        if subprocess.run(['git','merge-base','--is-ancestor',policy.COMMIT,commit],cwd=ROOT,
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            raise ValueError('Execution revision must descend from the pinned policy checkpoint')
        for name,expected in ((policy.PLAN_PATH,policy.REMOTE_PLAN_SHA),(policy.CODE_PATH,policy.REMOTE_CODE_SHA)):
            if policy.sha(policy.git_blob(commit,name))!=expected:
                raise ValueError('Returned execution changed the admitted policy/blueprint')
        for job,tree in policy.JOBS.items():
            actual=policy.git_blob(commit,policy.EXECUTION+'bootstrap-'+tree+'/preparation.json')
            if actual!=self.preparations[job]:raise ValueError('Pinned fresh preparation changed')

    def metadata(self,commit,job):
        self.require_revision(commit)
        return self.base_metadata(commit,job)

    def validate_metadata(self,job,blobs):
        self.check_sources()
        if job!=self.transport.job_spec(job['name']):
            raise ValueError('Full original job recipe and every named pair required')
        result=self.base_validate_metadata(job,blobs)
        _,proof,receipt,launch,_=[json.loads(blobs[name]) for name in
            ['archive-index.json','export-verification.json',job['name']+'-receipt.json',job['name']+'-launch.json','transfer.json']]
        prep_bytes=self.preparations[job['name']];prep=json.loads(prep_bytes);expected=policy.sha(prep_bytes)
        plan=json.loads((POLICY/'remote-plan.json').read_text())
        if (proof['files_sha256']['preparation.json']!=expected or launch['preparation_sha256']!=expected or
            timestamp(launch['deadline'])!=timestamp(plan['training_deadline_utc']) or
            timestamp(launch['started'])<timestamp(prep['observed']) or
            timestamp(receipt['observed'])<timestamp(launch['started'])):
            raise ValueError('Exact fresh preparation, original deadline and chronological launch required')
        for name,digest in prep['binary_sha256'].items():
            if proof['original_source_and_binary_sha256']['source/'+name]!=digest:
                raise ValueError('Fresh prepared binary expectation differs')
        return result
