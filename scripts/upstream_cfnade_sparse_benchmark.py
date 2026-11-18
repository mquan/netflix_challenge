#!/usr/bin/env python3
"""Measure sparse Adam traversal against the current checkpoint-capable trainer."""
from pathlib import Path
import statistics
import subprocess
import upstream_cfnade_sparse_rows as candidate

u=candidate.u


def run():
    proof=candidate.REPORT/'verification.json';record=u.read(proof)
    if not record['pass_all']:raise ValueError('Sparse candidate parity required')
    prepared=u.read(candidate.REPORT/'preparation.json')
    if record['preparation']!=prepared:raise ValueError('Sparse candidate preparation changed')
    for name,digest in prepared['files'].items():
        if u.sha(candidate.TREE/name)!=digest:raise ValueError('Sparse candidate source changed')
    source=u.ROOT/'research/cfnade_fast/sparse_rows_benchmark.rs';driver=candidate.TREE/'src/bin/cfnade-sparse-benchmark.rs'
    driver.write_bytes(source.read_bytes())
    env=u.environment();env.update(CARGO_TARGET_DIR=str(candidate.BASE/'target'),CARGO_BUILD_JOBS='2')
    for key in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT']:env.pop(key,None)
    with (candidate.REPORT/'benchmark-build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','cfnade-sparse-benchmark'],cwd=candidate.TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=candidate.BASE/'target/release/cfnade-sparse-benchmark'
    result=subprocess.run([str(binary)],cwd=candidate.TREE,env=env,capture_output=True,text=True,timeout=600)
    log=candidate.REPORT/'benchmark.log';log.write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Sparse timing/parity failed; preserve log')
    timings={'baseline':[],'candidate':[]}
    for line in result.stdout.splitlines():
        if line.startswith('BENCH '):
            _,mode,repeat,epoch,seconds=line.split();timings[mode].append(float(seconds))
    if any(len(rows)!=9 for rows in timings.values()):raise ValueError('Missing sparse timing samples')
    medians={name:statistics.median(rows) for name,rows in timings.items()}
    for name,digest in prepared['files'].items():
        if u.sha(candidate.TREE/name)!=digest:raise ValueError('Sparse candidate changed during timing')
    output=dict(parity_proof_sha256=u.sha(proof),source_sha256=u.sha(source),driver_sha256=u.sha(driver),
        script_sha256=u.sha(Path(__file__)),executable_sha256=u.sha(binary),log_sha256=u.sha(log),
        timings_seconds=timings,medians_seconds=medians,speedup=medians['baseline']/medians['candidate'],
        all_state_and_prediction_bits_equal=True,
        scope='Canonical320/96,batch8,512 synthetic users with200 ratings each/2048 items. Nine timed epochs per implementation with alternating order; full-state/query equality after every epoch. Concurrently busy host; not full-scale throughput or goal quality.')
    u.write(candidate.REPORT/'benchmark.json',output);print(output,flush=True)


if __name__=='__main__':run()
