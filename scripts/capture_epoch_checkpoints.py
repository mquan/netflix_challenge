#!/usr/bin/env python3
"""Retain atomically published epoch checkpoints without altering a live fit."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--producer',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--epochs',type=int,nargs='+',required=True)
    a=ap.parse_args();producer=Path(a.producer);root=Path(a.output)
    deadline=time.monotonic()+12*3600
    # A producer can spend time hashing inputs before publishing its first
    # report. Do not create its output directory ahead of its exclusive mkdir.
    while not (producer/'report.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('producer did not start')
        time.sleep(5)
    root.mkdir(parents=True,exist_ok=False)
    plan=dict(epochs=a.epochs,producer=str(producer),snapshots={},qualifying_labels_read=False)
    progress=producer/'model.progress.json';checkpoint=producer/'model.pt'
    while time.monotonic()<deadline:
        r=json.loads((producer/'report.json').read_text())
        if progress.exists() and checkpoint.exists():
            encoded=progress.read_bytes();p=json.loads(encoded);epoch=p['selected_epoch']
            before=checkpoint.stat()
            if epoch in a.epochs and str(epoch) not in plan['snapshots'] and before.st_mtime_ns<=progress.stat().st_mtime_ns:
                temp=root/f'e{epoch}.tmp';shutil.copyfile(checkpoint,temp)
                after=checkpoint.stat()
                if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size) or progress.read_bytes()!=encoded:
                    temp.unlink()
                else:
                    path=root/f'e{epoch}.pt';temp.replace(path)
                    plan['snapshots'][str(epoch)]=dict(path=str(path),sha256=digest(path),
                        source_report_sha256=digest(producer/'report.json'),
                        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    (root/'report.json').write_text(json.dumps(plan,indent=2)+'\n')
                    print(json.dumps(dict(state='captured',epoch=epoch,sha256=plan['snapshots'][str(epoch)]['sha256'])),flush=True)
        if len(plan['snapshots'])==len(set(a.epochs)):
            plan['state']='complete';break
        if r['state'] not in ('prepared','training'):
            plan['state']='producer_ended_before_all_snapshots';break
        time.sleep(10)
    else:plan['state']='watch_timeout'
    (root/'report.json').write_text(json.dumps(plan,indent=2)+'\n')


if __name__=='__main__':main()
