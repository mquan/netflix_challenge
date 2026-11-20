#!/usr/bin/env python3
"""Original full two-phase voting generator, exact FIFO bytes, RAM probe features."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime as dt
import hashlib
from multiprocessing import shared_memory
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import numpy as np
import upstream_repro as u
import upstream_remote15h as original
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

RUN=u.ROOT/'artifacts/upstream-split3-voting-stream'
REPORT=u.ROOT/'experiments/upstream-split3-voting-stream'
TREE=u.ROOT/'artifacts/upstream-split3-next3/source'

class HashedReader:
    def __init__(self,stream):self.stream=stream;self.sha=hashlib.sha256();self.bytes=0
    def read(self,count):
        value=self.stream.read(count);self.sha.update(value);self.bytes+=len(value);return value

def frame(path,count,target=None,reference=None):
    with path.open('rb',buffering=0) as stream:
        reader=HashedReader(stream);version=np.lib.format.read_magic(reader)
        shape,order,dtype=np.lib.format._read_array_header(reader,version)
        if shape!=(count,) or order or dtype!=np.dtype('<f4'):raise ValueError('Original full native float32 NPY frame differs')
        header_bytes=reader.bytes;payload=hashlib.sha256();offset=0;equal=True
        destination=memoryview(target).cast('B') if target is not None else None
        expected=memoryview(reference).cast('B') if reference is not None else None
        while offset<count*4:
            value=reader.read(min(4*2**20,count*4-offset))
            if not value:raise ValueError('Incomplete original native feature stream')
            payload.update(value)
            if destination is not None:destination[offset:offset+len(value)]=value
            if expected is not None and value!=expected[offset:offset+len(value)].tobytes():equal=False
            offset+=len(value)
    if target is not None and not np.isfinite(target).all():raise ValueError('Nonfinite complete native feature')
    return dict(rows=count,serialized_sha256=reader.sha.hexdigest(),payload_sha256=payload.hexdigest(),
        header_bytes=header_bytes,reference_payload_bitwise_equal=equal if reference is not None else None)

def fixture():
    import tempfile
    source=next(iter(sorted((u.FRESH/'preds_new/vf').glob('*.qual.npy'))));a=np.load(source,mmap_mode='r')
    with __import__('tempfile').TemporaryDirectory() as directory:
        pipe=Path(directory)/'native.npy';os.mkfifo(pipe)
        def writer():
            with source.open('rb') as src,pipe.open('wb',buffering=0) as dst:
                for block in iter(lambda:src.read(262144),b''):dst.write(block)
        thread=threading.Thread(target=writer);thread.start()
        target=np.empty_like(a);r=frame(pipe,len(a),target=target,reference=a);thread.join()
        if r['serialized_sha256']!=u.sha(source) or not r['reference_payload_bitwise_equal'] or not np.array_equal(a.view('u4'),target.view('u4')):
            raise ValueError('Complete original NPY FIFO byte/shape transport fixture differs')
    print('Original complete2817131-row NPY stream: every byte, float32 bit and SHA256 match',flush=True)

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing original feature-stream evidence preserved')
    RUN.mkdir();REPORT.mkdir();destination=RUN/'preds/vf';destination.mkdir(parents=True)
    bank=u.ROOT/'experiments/upstream-split3-full-raw-bank/manifest.json';b=u.read(bank)
    if b['state']!='complete_full_probe3_own_raw_bank_all_native_pairs_verified':raise ValueError('Full original raw/query alignment proof required')
    receipt_path=u.BASE/'fresh-receipts/vfeat1.json';receipt=u.read(receipt_path)
    if receipt['source_commit']!=u.PIN:raise ValueError('Original own fulltrain feature source pin required')
    references=sorted((u.FRESH/'preds_new/vf').glob('*.qual.npy'))
    if len(references)!=130:raise ValueError('All130 original own qualifying features required')
    names=[p.name.removesuffix('.qual.npy') for p in references]
    sources=dict(b['source_sha256'])
    for p in [bank,receipt_path,Path(__file__),Path(resources.__file__),TREE/'target/release/fit']:
        key=str(p.relative_to(u.ROOT));h=u.sha(p)
        if key in sources and sources[key]!=h:raise ValueError('Original native implementation changed')
        sources[key]=h
    for p in references:
        key=str(p.relative_to(u.FRESH));expected=receipt['outputs'][key]
        if u.sha(p)!=expected:raise ValueError('Existing own fulltrain voting feature changed')
        sources[str(p.relative_to(u.ROOT))]=expected
    for key,h in sources.items():
        if u.sha(u.ROOT/key)!=h:raise ValueError('Original full source changed before streaming')
    pipes=[]
    for stage in ['probe_3','qual']:
        for name in names:
            p=destination/(name+'.'+stage+'.npy');os.mkfifo(p,mode=0o600);pipes.append(p)
    previous=memory.memory();good=0
    while good<2:
        workers=resources.footprints();current=memory.memory()
        good=good+1 if sum(v['bytes'] for v in workers.values())+15*2**30<24*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20 else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_complete_native_voting_headroom',workers=workers,memory=current,good_samples=good))
        previous=current
        if good<2:time.sleep(10)
    segment=shared_memory.SharedMemory(create=True,size=130*1760494*4)
    table=np.ndarray((130,1760494),dtype='<f4',buffer=segment.buf)
    command=['./target/release/fit','vfeat1','vfeat1','-p','pipeline-3.toml','--preds',str(destination.parent)]
    environment=original.environment({'RAYON_NUM_THREADS':'4'})
    plan=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_pin=u.PIN,source_sha256=sources,
        command=command,environment=environment,probe3_rows=1760494,qualifying_rows=2817131,feature_names=names,
        rule='Unchanged original native binary executes both complete original phases and all130 features. Ordinary FIFO output transports exact native NPY frames to RAM; no large disk cache. Probe3 table lives until explicit release; qualifying payloads compared bitwise to existing verified own fulltrain features. No qualifying outcomes, fitting or data/feature/budget reduction.')
    u.write(REPORT/'plan.json',plan);records=[];errors=[];child=None
    def consume():
        try:
            for index,name in enumerate(names):
                row=frame(destination/(name+'.probe_3.npy'),1760494,target=table[index])
                records.append(dict(name=name,query='probe_3',**row))
            for index,name in enumerate(names):
                ref=np.load(references[index],mmap_mode='r')
                if ref.shape!=(2817131,) or ref.dtype!=np.float32 or not np.isfinite(ref).all():raise ValueError('Complete own qualifying reference shape/finite domain differs')
                row=frame(destination/(name+'.qual.npy'),2817131,reference=ref)
                records.append(dict(name=name,query='qual',**row))
        except BaseException as error:errors.append(repr(error))
    thread=threading.Thread(target=consume,daemon=True);thread.start()
    def interrupted(signum,frame):raise KeyboardInterrupt('Original feature transport signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        baseline=memory.memory();high=0
        with (REPORT/'native.log').open('x') as log:
            child=subprocess.Popen(command,cwd=TREE,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if errors:raise RuntimeError('Original NPY stream failed: '+errors[0])
                workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                high=high+1 if (sum(v['bytes'] for v in workers.values())>24*2**30 or sum(v['rss_bytes'] for v in workers.values())>24*2**30 or
                    own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                u.write(REPORT/'status.json',dict(state='original_complete_native_two_phase_features',pid=child.pid,
                    frames_received=len(records),workers=workers,memory=current,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Original voting resource guard exceeded; no truncated result admitted')
                time.sleep(5)
            if child.returncode:raise RuntimeError('Original full native feature generator failed')
        thread.join(timeout=10)
        if thread.is_alive() or errors or len(records)!=260:raise RuntimeError('All260 complete original feature streams required: '+repr(errors))
        for key,h in sources.items():
            if u.sha(u.ROOT/key)!=h:raise ValueError('Original source changed during native generation')
        result=dict(state='complete_original_full_two_phase_voting_features_in_ram',source_pin=u.PIN,
            plan_sha256=u.sha(REPORT/'plan.json'),records=records,feature_names=names,
            shared_memory=dict(name=segment.name,shape=[130,1760494],dtype='<f4',owner_pid=os.getpid()),
            probe3_feature_major_payload_sha256=hashlib.sha256(memoryview(table).cast('B')).hexdigest(),
            all_existing_own_qualifying_payloads_bitwise_equal=all(r['reference_payload_bitwise_equal'] for r in records if r['query']=='qual'),
            qualifying_labels_read=False,models_fitted=0,native_source_or_binary_modified=False,
            scope='Complete actual original native computation, exact full NPY stream hashes and RAM inputs. Recompute original feature transport after RAM release to rematerialize inputs; no approximate/sample reconstruction.')
        u.write(REPORT/'verification.json',result)
        print('All260 full native features received; RAM owner',os.getpid(),'segment',segment.name,flush=True)
        while not (REPORT/'release.json').exists():
            u.write(REPORT/'owner-status.json',dict(state='holding_complete_original_probe3_features_for_matched_study',pid=os.getpid(),shared_memory=segment.name,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
            time.sleep(15)
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
        thread.join(timeout=.2)
        if thread.is_alive():
            index=min(len(records),259);stage='probe_3' if index<130 else 'qual'
            pending=destination/(names[index%130]+'.'+stage+'.npy')
            wake=os.open(pending,os.O_RDWR|os.O_NONBLOCK)
            try:os.write(wake,b'FAILED!!');thread.join(timeout=3)
            finally:os.close(wake)
        u.write(REPORT/'failure.json',dict(error=repr(error),received_frames=len(records),original_native_partial_evidence_preserved=True));raise
    finally:
        del table;segment.close();segment.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    fixture() if args.fixture else main()
