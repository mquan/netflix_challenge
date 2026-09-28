#!/usr/bin/env python3
"""Matched contextual-ridge additions of linear and nonlinear user kernels."""
import argparse
import array
import hashlib
import json
import pathlib
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',required=True)
p.add_argument('--kernel-prefix',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
bank=json.loads(pathlib.Path('experiments/UNION-round-manifest.json').read_text())['reports']['union']['inputs']
mask=array.array('f',pathlib.Path(a.kernel_prefix+'.mask.f32').read_bytes())
if not mask or any(v!=1 for v in mask): raise ValueError('full-probe predictions required')
for name,sha in bank.items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==sha,name
results={}
for family,indices in [('linear',range(3)),('rbf',range(3,6)),('both',range(6))]:
    names=list(bank)+[a.kernel_prefix+'.'+str(i)+'.f32' for i in indices]
    command=['build/netflix','ensemble','--data',a.data,'--contexts','1']
    for name in names: command+=['--prediction',name]
    output=a.output+'-'+family
    command+=['--predictions','artifacts/'+output+'.f32']
    with open('experiments/'+output+'.log','w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    results[family]=dict(command=command,
        inputs={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in names},
        output_sha256=hashlib.sha256(pathlib.Path('artifacts/'+output+'.f32').read_bytes()).hexdigest())
    print(f'{family} finished',flush=True)
pathlib.Path('experiments/'+a.output+'.json').write_text(json.dumps(results,indent=2)+'\n')
