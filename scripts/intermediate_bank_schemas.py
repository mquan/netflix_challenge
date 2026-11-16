#!/usr/bin/env python3
"""Describe intermediate local-column insertions without duplicating matrices."""
import argparse
import copy
import json
from pathlib import Path
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-bank',required=True);ap.add_argument('--bank',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();base=json.loads(Path(a.base_bank+'.inputs.json').read_text())
    final=json.loads(Path(a.bank+'.inputs.json').read_text())
    n=base['columns'];end=final['columns'];delta=end-n
    if delta<1 or base['config'].get('extra') or len(final['config']['extra'])!=delta:
        raise ValueError('expected only appended local base columns')
    if final['names'][:n]!=base['names'][:n] or final['names'][end:]!=base['names'][n:]:
        raise ValueError('base or support columns differ')
    for key in ('data_sha256','meta_sha256','extra_meta_names'):
        if final[key]!=base[key]:raise ValueError('different development schemas')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);records=[]
    for count in range(1,delta):
        b=copy.deepcopy(final);names=final['names'][:n+count]+final['names'][end:]
        prefix=str(root/f'stage{count}')
        b.update(names=names,columns=n+count,inputs={name:final['inputs'][name] for name in names},
                 is_schema_only=True,parent_bank_sha256=digest(a.bank+'.inputs.json'))
        b['config'].update(output=prefix,extra=final['config']['extra'][:count])
        Path(prefix+'.inputs.json').write_text(json.dumps(b,indent=2)+'\n')
        records.append(dict(prefix=prefix,columns=b['columns'],sha256=digest(prefix+'.inputs.json')))
    (root/'manifest.json').write_text(json.dumps(dict(state='complete',schemas=records),indent=2)+'\n')
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
