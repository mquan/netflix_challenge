#!/usr/bin/env python3
"""Time bitwise-equal traversal alternatives without touching live training."""
from pathlib import Path
import os
import statistics
import subprocess
import upstream_grouped_storage as candidate

u=candidate.u


def run():
    proof=candidate.REPORT/'verification.json';record=u.read(proof)
    if not record['pass_all']:raise ValueError('Candidate parity is not complete')
    prepared=u.read(candidate.BASE/'parallel-preparation.json')
    if record['preparation']!=prepared:raise ValueError('Candidate preparation changed')
    for name,digest in prepared['generated'].items():
        if u.sha(candidate.TREE/name)!=digest:raise ValueError('Candidate source changed')
    fixture=candidate.TREE/'src/bin/tsvdx4p-storage-parity.rs'
    support=u.ROOT/'research/parallel_mapped/grouped_benchmark.rs'
    driver=candidate.TREE/'src/bin/grouped-storage-benchmark.rs'
    driver.write_text(fixture.read_text().split('fn check(',1)[0]+support.read_text())
    env=u.environment();env.update(CARGO_TARGET_DIR=str(candidate.BASE/'target'),
        CARGO_BUILD_JOBS='2',CARGO_NET_OFFLINE='true',RAYON_NUM_THREADS='2')
    with (candidate.REPORT/'benchmark-build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','grouped-storage-benchmark'],cwd=candidate.TREE,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=candidate.BASE/'target/release/grouped-storage-benchmark';arms=[]
    for mapped in [False,True]:
        name='mapped' if mapped else 'heap';scratch=candidate.BASE/('benchmark-'+name)
        arm={k:v for k,v in env.items() if k not in ['TX_MMAP_DIR','TX_LAZY_DAY']}
        if mapped:arm['TX_MMAP_DIR']=str(scratch)
        child=subprocess.run([str(executable)],cwd=candidate.TREE,env=arm,capture_output=True,text=True,timeout=300)
        path=candidate.REPORT/('benchmark-'+name+'.log');path.write_text(child.stdout+child.stderr)
        if child.returncode:raise RuntimeError('Timed traversal failed; preserved log '+str(path))
        timings={'row':[],'grouped':[]};digests=[]
        for line in child.stdout.splitlines():
            if line.startswith('BENCH '):
                _,mode,repeat,seconds,digest=line.split();timings[mode].append(float(seconds));digests.append(digest)
        if any(len(a)!=3 for a in timings.values()) or len(set(digests))!=1:
            raise ValueError('Timed traversal samples incomplete or not bitwise equal')
        if list(scratch.glob('tx-*.f32')):raise ValueError('Timed mapping files survived successful exit')
        medians={mode:statistics.median(values) for mode,values in timings.items()}
        arms.append(dict(arm=name,timings_seconds=timings,medians_seconds=medians,
            speedup=medians['row']/medians['grouped'],all_prediction_md5_equal=digests[0],log_sha256=u.sha(path)))
        print(name,medians,'speedup',arms[-1]['speedup'],flush=True)
    if any(u.sha(candidate.TREE/name)!=digest for name,digest in prepared['generated'].items()):
        raise ValueError('Candidate changed during timing')
    u.write(candidate.REPORT/'benchmark.json',dict(parity_proof_sha256=u.sha(proof),
        driver_sha256=u.sha(driver),support_sha256=u.sha(support),script_sha256=u.sha(Path(__file__)),
        executable_sha256=u.sha(executable),arms=arms,
        scope='Synthetic inference timing on this concurrently busy host,200000 query rows,1000 factors,20000 padded users;80MB user matrices. Three alternating-order repeats; each timed output is bitwise identical. Not full-scale training speed, memory admission, or competition quality.'))


if __name__=='__main__':run()
