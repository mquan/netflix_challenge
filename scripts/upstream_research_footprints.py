#!/usr/bin/env python3
"""Account for native and Python research peers, including compressed idle ones."""
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def live(pid):
    row=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],text=True,capture_output=True).stdout.strip()
    return bool(row and not row.split(None,1)[0].startswith('Z'))

def footprints(extra=None):
    result={}
    for line in subprocess.check_output(['ps','-axo','pid=,stat=,rss=,command='],text=True).splitlines():
        parts=line.strip().split(None,3)
        if len(parts)!=4:continue
        pid,state,rss,cmd=parts;pid=int(pid)
        if state.startswith('Z'):continue
        native='/target/release/' in cmd.split()[0]
        python_script=any(Path(token).name.startswith('upstream_') or Path(token).name.startswith('verify_upstream_') for token in cmd.split()[1:])
        if not (pid==extra or native or str(ROOT) in cmd or python_script):continue
        match=None
        for attempt in range(3):
            if not live(pid):break
            summary=subprocess.run(['vmmap','-summary',str(pid)],text=True,capture_output=True)
            match=re.search(r'^Physical footprint:\s+([\d.]+)([KMG])',summary.stdout,re.M)
            if match or not live(pid):break
            time.sleep(.2)
        if match:
            result[str(pid)]=dict(bytes=float(match.group(1))*{'K':2**10,'M':2**20,'G':2**30}[match.group(2)],rss_bytes=int(rss)*1024,command=cmd)
        elif live(pid):raise RuntimeError(f'Cannot account for live research process {pid}: {cmd}')
    return result
