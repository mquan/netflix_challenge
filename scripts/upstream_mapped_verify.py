#!/usr/bin/env python3
"""Build and verify all mapped-storage execution modes against pinned arithmetic."""
import os
import subprocess
import upstream_repro as u
import upstream_mapped_prepare as m


def verify():
    m.prepare()
    env=u.environment();env['CARGO_TARGET_DIR']=str(m.BASE/'target')
    report=u.ROOT/'experiments/full-upstream-reproduction/mapped-parity'
    report.mkdir(parents=True,exist_ok=True)
    with (report/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
                        '--bin','tx-storage-parity'],cwd=m.TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=m.BASE/'target/release/tx-storage-parity'
    initial=u.read(m.BASE/'preparation.json')
    rows=[]
    for name,extra in [('heap-cached',{}),('heap-lazy',{'TX_LAZY_DAY':'1'}),
        ('mapped-cached',{'TX_MMAP_DIR':str(m.BASE/'scratch')}),
        ('mapped-lazy',{'TX_MMAP_DIR':str(m.BASE/'scratch'),'TX_LAZY_DAY':'1'})]:
        arm_env={k:v for k,v in env.items() if k not in ['TX_MMAP_DIR','TX_LAZY_DAY']}
        p=subprocess.run([str(executable)],env=dict(arm_env,**extra),capture_output=True,text=True)
        path=report/(name+'.log');path.write_text(p.stdout+p.stderr)
        rows.append(dict(arm=name,exit_code=p.returncode,log_sha256=u.sha(path),tail=(p.stdout+p.stderr)[-1000:]))
        print(name,p.returncode,flush=True)
    if list((m.BASE/'scratch').glob('tx-*.f32')):raise ValueError('Owned mapping files survived successful exit')
    final=u.read(m.BASE/'preparation.json')
    if final!=initial:raise ValueError('Source preparation changed while verifying')
    result=dict(executable_sha256=u.sha(executable),preparation=initial,arms=rows,
                pass_all=all(r['exit_code']==0 for r in rows),
                scope='Synthetic bitwise arithmetic and 72 MB backing-file persistence; full-scale training resource validation remains required.')
    u.write(report/'verification.json',result)
    if not result['pass_all']:raise ValueError('Mapped temporal parity failed')


if __name__=='__main__':verify()
