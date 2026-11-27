"""Re-evaluate each newly completed, development-eligible package against the current frozen parent."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as retained
import own_full_context as full

ROOT=blend.OUT
FAMILIES={
    'own-anchor-covariance-20261126':ne.ROOT/'own-anchor-covariance-20261126',
    'own-user-gp-20261126':ne.ROOT/'own-user-gp-20261126',
    'attention':ne.ROOT/'own-masked-attention-20261126',
    'shallow':full.OUT/'shallow',
    'deep':full.OUT/'deep',
    'own-kernel-family-20261126':ne.ROOT/'own-kernel-family-20261126',
    'bayes':retained.OUT/'bayes',
    'attention:own-temporal-attention-20261126':ne.ROOT/'own-temporal-attention-20261126',
    'own-retrieval-net-20261126':ne.ROOT/'own-retrieval-net-20261126',
}
CUTOFF=datetime.datetime(2026,11,26,4,58).timestamp()


def run():
    statepath=ROOT/'queue.json'
    if statepath.exists():state=json.loads(statepath.read_text())
    else:state=dict(parent='first',tested=[],keys=['retained','anchor','pairs'],stages=[])
    while time.time()<CUTOFF:
        new=[]
        all_roots=dict(FAMILIES)
        extra=ROOT/'extra-families.json'
        if extra.exists():all_roots.update({k:Path(v) for k,v in json.loads(extra.read_text()).items()})
        for key,folder in all_roots.items():
            if key in state['tested'] or not (folder/'freeze.json').exists():continue
            report=json.loads((folder/'freeze.json').read_text())
            if report['qualifying_hashes']:new.append(key)
            else:state['tested'].append(key)
        if new:
            stage='stage-%02d'%(len(state['stages'])+1)
            keys=state['keys']+new
            argv=[sys.executable,'-u','scripts/own_overnight_blend.py',stage,*keys,'--parent',state['parent']]
            subprocess.run(argv,check=True)
            report=json.loads((ROOT/stage/'freeze.json').read_text())
            state['stages'].append(dict(name=stage,added=new,selected=report['selected'],
                known=report['candidate']['pooled'],wins=report['improved_folds']))
            state['tested']+=new
            state['keys']=keys
            state['parent']=stage
        tmp=statepath.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(statepath)
        if len(state['tested'])==len(all_roots) and (ROOT/'families-closed').exists():return
        time.sleep(15)


if __name__=='__main__':run()
