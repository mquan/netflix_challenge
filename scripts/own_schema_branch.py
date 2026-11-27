"""Replay the fixed development gate with rating history and forecast inputs kept distinct."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_overnight_queue as queue

OUT=ne.ROOT/'own-schema-branch-20261126'
ANCHOR='stage-24'
EXCLUDE='own-distribution-stack-20261126'
NATIVE_ATTENTION={
    'attention:own-cf-temporal-attention-20261126',
    'attention:own-temporal-continuation-20261126',
    'attention:own-slate-attention-20261126',
}


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def run():
    OUT.mkdir(exist_ok=True);statepath=OUT/'queue.json'
    original=json.loads((blend.OUT/ANCHOR/'freeze.json').read_text())
    if statepath.exists():
        state=json.loads(statepath.read_text())
    else:
        state=dict(parent=ANCHOR,keys=original['columns'],stages=[],excluded={EXCLUDE:'Forecast matrix included 14 virtual statistics in rating units.'})
    finish=datetime(2026,11,26,4,58).timestamp()
    while time.time()<finish:
        root=json.loads((blend.OUT/'queue.json').read_text())
        roots=dict(queue.FAMILIES)
        roots.update({k:Path(v) for k,v in json.loads((blend.OUT/'extra-families.json').read_text()).items()})
        pending=[]
        for stage in root['stages']:
            if stage['name']<=ANCHOR:continue
            for key in stage['added']:
                if key in state['keys'] or key in state['excluded']:continue
                record=json.loads((roots[key]/'freeze.json').read_text())
                parent=Path(record.get('parent','')).name
                if key in NATIVE_ATTENTION:
                    # The blend adapter reads the direct trainx/fulltrain neural forecasts.
                    assert all((roots[key]/(arm+'.npy')).is_file() for arm in ('probex','qual'))
                    pending.append(key)
                elif parent.startswith('stage-') and parent>ANCHOR:
                    state['excluded'][key]='Derived from forecast-matrix branch '+parent
                elif parent.startswith('schema-') or parent in ('first','retained') or (parent.startswith('stage-') and parent<=ANCHOR):
                    pending.append(key)
                else:
                    state['excluded'][key]='Parent provenance requires review: '+parent
        if pending:
            stage='schema-%02d'%(len(state['stages'])+1)
            keys=state['keys']+pending
            dest=blend.OUT/stage
            if not (dest/'freeze.json').is_file():
                subprocess.run([sys.executable,'-u','scripts/own_overnight_blend.py',stage,*keys,'--parent',state['parent']],check=True)
            report=json.loads((dest/'freeze.json').read_text())
            assert report['columns']==keys and report['parent']==state['parent']
            state['stages'].append(dict(stage=stage,added=pending,selected=report['selected'],improved_folds=report['improved_folds']))
            state.update(parent=stage,keys=keys)
        write(statepath,state)
        time.sleep(20)


if __name__=='__main__':run()
