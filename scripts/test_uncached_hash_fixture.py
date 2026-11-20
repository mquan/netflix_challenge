#!/usr/bin/env python3
"""Exercise reversible Darwin read-cache policy on a private disposable fixture."""
import fcntl
import hashlib
from pathlib import Path
import re
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-uncached-hash-fixture'
RUN=u.ROOT/'artifacts/upstream-uncached-hash-fixture'
HEADER=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/fcntl.h')


def identity(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mode=s.st_mode,uid=s.st_uid,gid=s.st_gid,
        mtime_ns=s.st_mtime_ns,flags=s.st_flags,nlink=s.st_nlink)


def main():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing fixture evidence retained')
    source=HEADER.read_text()
    if not re.search(r'#define\s+F_GLOBAL_NOCACHE\s+55\b',source) or not re.search(r'#define\s+F_NOCACHE\s+48\b',source):
        raise ValueError('Actual local Darwin SDK cache controls required')
    REPORT.mkdir();RUN.mkdir();path=RUN/'fixture.bin';path.write_bytes(bytes(range(256))*4096)
    before=identity(path);expected=u.sha(path);prior=None
    with path.open('rb') as reader,path.open('rb') as controller:
        try:
            prior=fcntl.fcntl(controller.fileno(),55,1)
            already_enabled=fcntl.fcntl(controller.fileno(),55,1)
            inherited=hashlib.sha256(reader.read()).hexdigest()
            new_descriptor=u.sha(path)
            restoring_previous=fcntl.fcntl(controller.fileno(),55,int(bool(prior)))
            restored=fcntl.fcntl(controller.fileno(),55,int(bool(prior)))
        finally:
            if prior is not None:fcntl.fcntl(controller.fileno(),55,int(bool(prior)))
    with path.open('rb') as reader:
        fcntl.fcntl(reader.fileno(),48,1)
        per_descriptor=hashlib.sha256(reader.read()).hexdigest()
    if (not already_enabled or not restoring_previous or bool(restored)!=bool(prior) or
        not expected==inherited==new_descriptor==per_descriptor==u.sha(path) or before!=identity(path)):
        raise ValueError('Complete bytes, metadata or prior cache state did not reproduce')
    u.write(REPORT/'verification.json',dict(state='complete_private_fixture_uncached_hash_bytes_metadata_and_prior_policy_roundtrip',
        script_sha256=u.sha(Path(__file__)),header_sha256=u.sha(HEADER),fixture_sha256=expected,identity=before,
        prior_global_state=prior,already_enabled_return=already_enabled,restore_return=restoring_previous,restored_return=restored,
        preopened_descriptor_sha256=inherited,new_descriptor_sha256=new_descriptor,per_descriptor_sha256=per_descriptor,
        original_bytes_and_metadata_unchanged=True,prior_global_cache_state_restored=True,
        real_research_inputs_modified=False,real_worker_policy_changed=False,memory_pressure_reduction_proven=False,
        primary_reference='https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_descrip.c',
        qualifying_outcomes_read=False))
    print('Private fixture bytes/metadata and prior cache-policy roundtrip verified; no real worker policy changed',flush=True)


if __name__=='__main__':main()
