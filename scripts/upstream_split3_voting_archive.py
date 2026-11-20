#!/usr/bin/env python3
"""Losslessly preserve all original probe3 voting bytes before releasing RAM."""
import argparse
import datetime as dt
import hashlib
import os
from multiprocessing import shared_memory, resource_tracker
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-split3-voting-archive'
RUN=u.ROOT/'artifacts/upstream-split3-voting-archive'
ORIGINAL=u.ROOT/'experiments/upstream-split3-voting-stream'
SHAPE=(130,1760494)
FEATURE_BYTES=SHAPE[1]*4


def consumers():
    result=[]
    for line in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines():
        fields=line.strip().split(None,2)
        if len(fields)!=3 or fields[1].startswith('Z'):continue
        tokens=fields[2].split()
        if not Path(tokens[0]).name.startswith('python'):continue
        scripts=[Path(token).name for token in tokens[1:]]
        if any('split3_matched' in name for name in scripts):result.append(line)
    return result


def decoded(archive,aggregate=None):
    child=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    h=hashlib.sha256();count=0
    for block in iter(lambda:child.stdout.read(1<<20),b''):
        h.update(block);count+=len(block)
        if aggregate is not None:aggregate.update(block)
    error=child.stderr.read();code=child.wait()
    if code:raise RuntimeError('Exact archive decode failed: '+error.decode())
    return h.hexdigest(),count


def archive():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing full/partial exact archive preserved')
    REPORT.mkdir();RUN.mkdir()
    proof_path=ORIGINAL/'verification.json';proof=u.read(proof_path)
    matched=u.ROOT/'experiments/upstream-split3-matched-resume/independent-verification.json';done=u.read(matched)
    if (proof['state']!='complete_original_full_two_phase_voting_features_in_ram' or
        proof['shared_memory']['shape']!=list(SHAPE) or proof['shared_memory']['dtype']!='<f4' or
        done['state']!='complete_both_matched_arms_original_statistics_scalers_and_actual_saved_model_replays' or
        len(done['records'])!=10 or sum(r['actual_models_reopened'] for r in done['records'])!=30 or
        not all(r['saved_pipeline_replay_bitwise_equal'] for r in done['records']) or consumers()):
        raise ValueError('Complete original native table and ten actual completed model pipelines required')
    owner=owners.process(proof['shared_memory']['owner_pid'])
    if not owner or 'upstream_split3_voting_stream.py' not in owner['command']:
        raise ValueError('Exact original RAM holder must still be live')
    plan_path=ORIGINAL/'plan.json';plan=u.read(plan_path)
    if u.sha(plan_path)!=proof['plan_sha256']:raise ValueError('Original generation plan changed')
    for rel,digest in plan['source_sha256'].items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('Original feature-generation source changed')
    u.write(REPORT/'plan.json',dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),
            source_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),proof_path,plan_path,matched,u.ROOT/'SPLIT3_VOTING_ARCHIVE_PLAN.md']},
            original_owner=owner,shape=list(SHAPE),dtype='<f4',original_payload_sha256=proof['probe3_feature_major_payload_sha256'],
            rule='All original bytes, no recomputation or precision/feature/row reduction. Ten13-feature lossless parts. Full original/per-feature hashes, every independently decoded byte and completed matched pipelines required before explicit release. Existing fitted source/proofs unchanged.'))
    segment=shared_memory.SharedMemory(name=proof['shared_memory']['name']);resource_tracker.unregister(segment._name,'shared_memory')
    table=np.ndarray(SHAPE,dtype='<f4',buffer=segment.buf);records=[]
    try:
        if hashlib.sha256(segment.buf[:130*FEATURE_BYTES]).hexdigest()!=proof['probe3_feature_major_payload_sha256']:
            raise ValueError('Every original source table bit must match')
        for j in range(130):
            if hashlib.sha256(memoryview(table[j])).hexdigest()!=proof['records'][j]['payload_sha256']:
                raise ValueError('Every complete original feature must match')
        for first in range(0,130,13):
            count=min(13,130-first);length=count*FEATURE_BYTES
            # Preserve18.75GiB floor, two complete CF checkpoint writes and
            # a100MiB margin even if this part compresses no smaller than raw.
            minimum=18.75*2**30+2*352276417+length+100*2**20
            if shutil.disk_usage(u.ROOT).free<minimum:raise RuntimeError('Exact archive checkpoint/disk reserve insufficient; RAM remains held')
            target=RUN/f'features-{first:03}-{first+count-1:03}.f32.zst'
            h=hashlib.sha256()
            with target.open('xb') as output:
                child=subprocess.Popen(['zstd','-q','-T1','-c'],stdin=subprocess.PIPE,stdout=output,stderr=subprocess.PIPE)
                for offset in range(first*FEATURE_BYTES,(first+count)*FEATURE_BYTES,1<<20):
                    block=segment.buf[offset:min(offset+(1<<20),(first+count)*FEATURE_BYTES)]
                    h.update(block);child.stdin.write(block);del block
                child.stdin.close();error=child.stderr.read();code=child.wait()
                if code:raise RuntimeError('Lossless original table compression failed: '+error.decode())
                output.flush();os.fsync(output.fileno())
            digest,decoded_bytes=decoded(target)
            if digest!=h.hexdigest() or decoded_bytes!=length:raise ValueError('Every original part must independently decode exactly')
            records.append(dict(first_feature=first,feature_count=count,path=str(target.relative_to(u.ROOT)),
                                raw_bytes=length,raw_sha256=digest,archive_bytes=target.stat().st_size,archive_sha256=u.sha(target)))
            u.write(REPORT/'partial.json',dict(records=records))
        aggregate=hashlib.sha256()
        for row in records:
            digest,count=decoded(u.ROOT/row['path'],aggregate)
            if digest!=row['raw_sha256'] or count!=row['raw_bytes']:raise ValueError('Final full archive pass differs')
        if aggregate.hexdigest()!=proof['probe3_feature_major_payload_sha256'] or sum(r['raw_bytes'] for r in records)!=130*FEATURE_BYTES:
            raise ValueError('Full original feature-major restoration bytes differ')
    finally:
        del table;segment.close()
    if consumers() or not owners.same_owner(owner,owners.process(owner['pid'])):
        raise ValueError('Original table owner/consumers changed before release')
    result=dict(state='complete_lossless_original_probe3_voting_archive_all_bytes_verified',
                plan_sha256=u.sha(REPORT/'plan.json'),original_generation_proof_sha256=u.sha(proof_path),
                matched_complete_proof_sha256=u.sha(matched),shape=list(SHAPE),dtype='<f4',
                original_payload_sha256=aggregate.hexdigest(),raw_bytes=130*FEATURE_BYTES,parts=records,
                source_features_recomputed=False,original_bytes_discarded_without_archive=0,
                qualifying_labels_read=False,models_fitted=0)
    u.write(REPORT/'verification.json',result)
    # Persist the sole replacement bytes and their complete catalog before
    # the original owner may unlink its shared-memory segment.
    for path in [REPORT/'plan.json',REPORT/'verification.json']:
        with path.open('rb') as source:os.fsync(source.fileno())
    for directory in [RUN,REPORT,RUN.parent,REPORT.parent]:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    u.write(ORIGINAL/'release.json',dict(reason='Completed matched study and full independently decoded exact archive',
            archive_verification_sha256=u.sha(REPORT/'verification.json'),original_owner=owner))
    # Frozen owner polls its explicit release contract every15seconds.
    for _ in range(300):
        if not owners.same_owner(owner,owners.process(owner['pid'])):break
        time.sleep(.1)
    if owners.same_owner(owner,owners.process(owner['pid'])):raise RuntimeError('Owner release pending; do not signal or discard archive')
    u.write(REPORT/'release-verification.json',dict(state='complete_original_RAM_owner_released_after_full_lossless_archive',
            original_owner=owner,archive_verification_sha256=u.sha(REPORT/'verification.json'),
            freed_original_table_bytes=130*FEATURE_BYTES,source_and_completed_model_files_modified=False))
    print('Full original voting bytes archived and verified; idle RAM owner released',flush=True)


def restore():
    proof=u.read(REPORT/'verification.json');path=RUN/'probe3-feature-major.f32'
    if path.exists():
        if path.stat().st_size!=proof['raw_bytes'] or u.sha(path)!=proof['original_payload_sha256']:
            raise ValueError('Existing restoration differs; preserve and investigate')
        return path
    if shutil.disk_usage(u.ROOT).free<18.75*2**30+proof['raw_bytes']+2*352276417+100*2**20:
        raise RuntimeError('Complete restoration requires original disk/checkpoint headroom')
    temp=path.with_name(path.name+'.restoring')
    if temp.exists():raise ValueError('Existing partial restoration preserved')
    with temp.open('xb') as output:
        for row in proof['parts']:
            archive=u.ROOT/row['path']
            if u.sha(archive)!=row['archive_sha256']:raise ValueError('Protected exact archive changed')
            subprocess.run(['zstd','-q','-d','-c',str(archive)],stdout=output,check=True)
    if temp.stat().st_size!=proof['raw_bytes'] or u.sha(temp)!=proof['original_payload_sha256']:
        raise ValueError('Every exact original restored byte must match')
    temp.replace(path);return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    print(restore()) if args.restore else archive()
