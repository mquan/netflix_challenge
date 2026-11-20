#!/usr/bin/env python3
"""Release cached pages of explicit immutable matrices; never touch model RAM."""
import argparse
import ctypes as c
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import upstream_read_cache_policy as policy
import upstream_repro as u

HEADER=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/mman.h')
REPORT=u.ROOT/'experiments/upstream-readonly-cache-trim'


def trim(fd, length):
    lib=c.CDLL(None,use_errno=True)
    lib.mmap.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_int,c.c_int,c.c_longlong]
    lib.mmap.restype=c.c_void_p
    lib.mincore.argtypes=[c.c_void_p,c.c_size_t,c.c_void_p]
    lib.msync.argtypes=[c.c_void_p,c.c_size_t,c.c_int]
    lib.munmap.argtypes=[c.c_void_p,c.c_size_t]
    p=lib.mmap(None,length,1,1,fd,0)  # Darwin PROT_READ, MAP_SHARED
    if p==c.c_void_p(-1).value:raise OSError(c.get_errno(),'readonly mmap')
    try:
        def residency():
            pages=(length+os.sysconf('SC_PAGE_SIZE')-1)//os.sysconf('SC_PAGE_SIZE')
            vector=(c.c_ubyte*pages)()
            if lib.mincore(p,length,vector):raise OSError(c.get_errno(),'mincore')
            return sum(bool(value&1) for value in vector)
        before=residency()
        if lib.msync(p,length,0x12):raise OSError(c.get_errno(),'MS_SYNC|MS_INVALIDATE')
        return dict(resident_pages_before=before,resident_pages_after=residency(),page_bytes=os.sysconf('SC_PAGE_SIZE'))
    finally:
        if lib.munmap(p,length):raise OSError(c.get_errno(),'munmap')


def fixture():
    header=HEADER.read_text()
    for name,value in [('PROT_READ','0x01'),('MAP_SHARED','0x0001'),('MS_SYNC','0x0010'),('MS_INVALIDATE','0x0002')]:
        import re
        if not re.search(r'#define\s+'+name+r'\s+'+value+r'\b',header):raise ValueError('Darwin SDK constant changed')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'private-fixture.bin'
        path.write_bytes(b'fixture!'*(2**20))
        before=policy.identity(path);digest=u.sha(path)
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:result=trim(fd,before['bytes'])
        finally:os.close(fd)
        if before!=policy.identity(path) or digest!=u.sha(path):raise ValueError('Fixture bytes/metadata changed')
        if result['resident_pages_before']<=0 or result['resident_pages_after']!=0:raise ValueError('Fixture did not evict cached pages')
    return dict(state='complete_private_readonly_fixture_cache_eviction_bytes_metadata_equal',**result,
        script_sha256=u.sha(Path(__file__)),header_sha256=u.sha(HEADER),bytes_and_metadata_equal=True)


def main():
    if REPORT.exists():raise ValueError('Existing cache-trim observation retained')
    REPORT.mkdir()
    proof=fixture();u.write(REPORT/'fixture.json',proof)
    with policy.LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if u.read(policy.OWNER)['state']!='restored':raise ValueError('Prior policy must be restored')
        rows=policy.candidates()
        check=subprocess.run(['/usr/sbin/lsof','-Fpan','--',*[r['path'] for r in rows]],capture_output=True,text=True)
        if check.returncode not in (0,1) or check.stderr.strip() or any(s in ('aw','au') for s in check.stdout.splitlines()):
            raise ValueError('Observed writable matrix descriptor or incomplete lsof')
        u.write(REPORT/'plan.json',dict(rows=rows,fixture_sha256=u.sha(REPORT/'fixture.json'),
            rule='Only explicit existing readonly immutable feature/design inodes. Readonly mmap; Darwin msync sync+invalidate; mincore before/after. No data writes, no cache-policy changes, no model/checkpoint/tensor/RNG access, no worker signals. Metadata identity checked before/after. Does not claim compressor recovery. Full existing numerical source hashes remain frozen.'))
        records=[]
        for row in rows:
            path=Path(row['path']);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                s=os.fstat(fd)
                if policy.identity(path)!=row['identity'] or (s.st_dev,s.st_ino,s.st_size)!=(row['identity']['device'],row['identity']['inode'],row['identity']['bytes']):
                    raise ValueError('Matrix inode changed')
                result=trim(fd,s.st_size)
                if policy.identity(path)!=row['identity']:raise ValueError('Matrix metadata changed')
            finally:os.close(fd)
            records.append(dict(path=str(path),**result,metadata_equal=True))
            u.write(REPORT/'observation.json',dict(state='observing_readonly_cache_eviction',records=records))
        u.write(REPORT/'verification.json',dict(state='complete_explicit_readonly_matrix_cache_eviction_observation',
            records=records,source_sha256=u.sha(Path(__file__)),no_model_state_access=True,no_policy_changes=True,
            numeric_bytes_rehashed=False,all_original_proof_hashes_retained=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    print(fixture()) if args.fixture else main()
