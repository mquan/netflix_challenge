#!/usr/bin/env python3
"""Freeze exact original1200 output accounting without touching active jobs."""
import re
import shutil
import subprocess
from pathlib import Path
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-original1200-preflight'
DECL=u.ROOT/'ORIGINAL1200_EXECUTION_PLAN.md'
NAME='tsvdx5-1200lm'


def main():
    if REPORT.exists():raise ValueError('Existing original1200 preflight evidence retained')
    files=['src/bin/tsvdx5-new.rs','src/tx.rs','src/lib.rs']
    contents={name:subprocess.check_output(['git','show',u.PIN+':'+name],cwd=u.SOURCE) for name in files}
    dispatcher=contents[files[0]].decode()
    args=re.search(r'"tsvdx5-1200lm"\s*=>\s*run_base\(([^)]*)\)',dispatcher)
    expected=['1200','23','100','0.0015','false','1024','true','true','true','job_name']
    if args is None or [v.strip() for v in args.group(1).split(',')]!=expected:
        raise ValueError('Original1200 full preset differs')
    if ('["bias", "mf", "nsvd1", "nbr"]' not in contents['src/tx.rs'].decode() or
        'save_train: true, save_subscores: true, save_probe_each_epoch: true' not in dispatcher):
        raise ValueError('Original full component/training/epoch output semantics differ')
    registry_path=u.BASE/'registry.json';job=u.read(registry_path)['jobs'][NAME]
    if job['command']!='cargo build --release --bin tsvdx5-new && ./target/release/tsvdx5-new tsvdx5-1200lm':
        raise ValueError('Original registry command differs')
    rows=[]
    for query,count in [('probex',1128164),('qual',2817131)]:
        for prefix in [NAME]+[f'{NAME}_ep{epoch:02}' for epoch in range(1,24)]+[NAME+'-'+c for c in ['bias','mf','nsvd1','nbr']]:
            rows.append(dict(name=prefix+'.'+query+'.npy',rows=count,payload_bytes=count*4,dtype='float32'))
    for split,count in [('trainx',99352343),('fulltrain',100480507)]:
        rows.append(dict(name=NAME+'.'+split+'.npy',rows=count,payload_bytes=count*4,dtype='float32'))
    if len(rows)!=58 or len({r['name'] for r in rows})!=58:raise ValueError('Complete unique58 file inventory required')
    payload=sum(r['payload_bytes'] for r in rows)
    if payload+58*256>=1.25*2**30:raise ValueError('Declared output reserve is inadequate')
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [DECL,Path(__file__),registry_path]}
    source_sha256={}
    import hashlib
    for name,raw in contents.items():source_sha256[name]=hashlib.sha256(raw).hexdigest()
    free=shutil.disk_usage(u.ROOT).free
    REPORT.mkdir()
    u.write(REPORT/'preflight.json',dict(state='complete_original1200_exact_preset_and58_output_preflight_no_training',
        source_pin=u.PIN,source_sha256=bindings,original_git_blob_sha256=source_sha256,
        original_registry_job=job,factors=1200,neighbors=100,epochs_each_phase=23,
        expected_outputs=rows,total_numeric_payload_bytes=payload,output_reserve_bytes=int(1.25*2**30),
        continuous_disk_floor_bytes=int(18.75*2**30),observed_free_bytes=free,
        free_after_reserving_outputs_bytes=free-int(1.25*2**30),
        output_reserve_alone_admitted=free>20*2**30,compiler_reserve_not_included=True,
        training_launched=False,training_queued=False,qualifying_labels_read=False,
        full_data_binary_and_memory_admission_still_required=True))
    print('Original1200:',len(rows),'outputs;',round(payload/2**30,6),'GiB payload; freeGiB',round(free/2**30,3),flush=True)


if __name__=='__main__':main()
