#!/usr/bin/env python3
"""Journaled reversible Darwin cache policy for explicit immutable matrix inputs."""
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import signal
import upstream_repro as u

LOCK=u.ROOT/'experiments/upstream-read-cache-policy.lock'
OWNER=u.ROOT/'experiments/upstream-read-cache-policy-owner.json'


def write(path,value):
    stage=path.with_name(path.name+'.new')
    with stage.open('w') as out:
        json.dump(value,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
    stage.replace(path)


def identity(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
        mode=s.st_mode,uid=s.st_uid,gid=s.st_gid,flags=s.st_flags,nlink=s.st_nlink)


def candidates():
    paths={}
    for name in ['upstream-own391-paired-data/manifest.json','upstream-own352-bounded-replay/independent-canonical-verification.json']:
        proof=u.read(u.ROOT/'experiments'/name)
        for relative,digest in proof['source_sha256'].items():
            path=u.ROOT/relative
            if (path.is_relative_to(u.ROOT/'artifacts') and path.suffix=='.npy' and
                ('features' in path.name or 'design' in path.name) and path.is_file() and path.stat().st_size>=128*2**20):
                if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Explicit regular cache matrix required')
                s=identity(path);key=(s['device'],s['inode'])
                if key in paths and paths[key]['expected_sha256']!=digest:raise ValueError('Conflicting immutable matrix hash')
                paths.setdefault(key,dict(path=str(path),expected_sha256=digest,identity=s))
    if not paths:raise ValueError('Explicit bound cache matrix inventory required')
    return list(paths.values())


class Policy:
    def __init__(self,report,owner):
        self.report=report;self.owner=owner;self.lock=None;self.handles=[];self.records=[];self.restored=False

    def apply(self):
        fixture_path=u.ROOT/'experiments/upstream-uncached-hash-fixture/verification.json';fixture=u.read(fixture_path)
        if (fixture['state']!='complete_private_fixture_uncached_hash_bytes_metadata_and_prior_policy_roundtrip' or
            not fixture['prior_global_cache_state_restored'] or not fixture['original_bytes_and_metadata_unchanged'] or
            fixture['script_sha256']!=u.sha(u.ROOT/'scripts/test_uncached_hash_fixture.py') or
            fixture['header_sha256']!=u.sha('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/fcntl.h')):
            raise ValueError('Actual SDK/cache-policy roundtrip proof required')
        self.lock=LOCK.open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if OWNER.exists() and u.read(OWNER)['state']!='restored':raise ValueError('Previous cache policy must be explicitly restored')
        rows=candidates()
        observation=subprocess.run(['/usr/sbin/lsof','-Fpan','--',*[r['path'] for r in rows]],text=True,capture_output=True)
        if observation.returncode not in (0,1) or observation.stderr.strip() or any(line in ('aw','au') for line in observation.stdout.splitlines()):
            raise ValueError('Immutable cache matrices must have no active writable descriptor')
        write(self.report/'cache-policy-candidates.json',dict(rows=rows,fixture_sha256=u.sha(fixture_path)))
        write(OWNER,dict(state='active',report=str(self.report),owner=self.owner))
        try:
            for row in rows:
                path=Path(row['path']);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                before=identity(path)
                if before!=row['identity']:os.close(fd);raise ValueError('Cache matrix identity changed before policy')
                blocked=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
                try:
                    try:prior=fcntl.fcntl(fd,55,1)
                    except BaseException:os.close(fd);raise
                    self.handles.append(fd);self.records.append(dict(row,prior_global_state=prior))
                    write(self.report/'cache-policy.json',dict(state='active',owner=self.owner,records=self.records,
                        real_numeric_bytes_modified=False,source_code_modified=False))
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,blocked)
            return self.records
        except BaseException:
            self.restore();raise

    def restore(self):
        if self.restored:return
        if len(self.handles)!=len(self.records):
            proof=self.report/'cache-policy-owner-loss-restoration.json'
            if (OWNER.exists() and u.read(OWNER).get('state')=='restored' and u.read(OWNER).get('report')==str(self.report) and
                proof.exists() and len(u.read(proof)['records'])==len(self.records)):
                self.restored=True;return
            raise ValueError('Lost policy descriptors require explicit journal restoration')
        results=[]
        try:
            for fd,row in zip(self.handles,self.records):
                path=Path(row['path'])
                if identity(path)!=row['identity']:raise ValueError('Cache inode/metadata changed; explicit recovery required')
                desired=int(bool(row['prior_global_state']))
                fcntl.fcntl(fd,55,desired);observed=fcntl.fcntl(fd,55,desired)
                if bool(observed)!=bool(desired):raise ValueError('Prior file-cache policy did not restore')
                results.append(dict(path=row['path'],prior_global_state=row['prior_global_state'],restored_return=observed))
            if self.records:
                write(self.report/'cache-policy-restoration.json',dict(state='complete_prior_cache_policy_and_metadata_restoration',records=results))
            if OWNER.exists() and u.read(OWNER).get('report')==str(self.report):
                write(OWNER,dict(state='restored',report=str(self.report),owner=self.owner))
            self.restored=True
        finally:
            for fd in self.handles:os.close(fd)
            self.handles=[]
            if self.lock is not None:self.lock.close();self.lock=None


def restore_after_owner_loss(report):
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=report/'cache-policy.json'
        if not path.exists():return
        record=u.read(path);results=[]
        if OWNER.exists() and u.read(OWNER).get('report')!=str(report):raise ValueError('Another cache lease must not be changed')
        for row in record['records']:
            p=Path(row['path'])
            if identity(p)!=row['identity']:raise ValueError('Exact prior cache inode required for restoration')
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                desired=int(bool(row['prior_global_state']));fcntl.fcntl(fd,55,desired);observed=fcntl.fcntl(fd,55,desired)
                if bool(observed)!=bool(desired):raise ValueError('Prior cache policy restoration failed')
                results.append(dict(path=str(p),prior_global_state=row['prior_global_state'],restored_return=observed))
            finally:os.close(fd)
        write(report/'cache-policy-owner-loss-restoration.json',dict(state='complete_all_journaled_prior_cache_policies_restored',records=results,
            abrupt_failure_during_unjournaled_syscall_cannot_be_excluded=True))
        write(OWNER,dict(state='restored',report=str(report),owner=record['owner']))
