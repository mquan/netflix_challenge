#!/usr/bin/env python3
"""Check deployed commit and every whitelisted byte, including the root URL."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
BASE='https://mquan.github.io/netflix_challenge/'


def main(commit):
    output=ROOT/'experiments/website-complete13-verification.json'
    if output.exists():raise ValueError('Existing exact deployment proof retained')
    whitelist=ROOT/'experiments/website-complete13-publish-files.json'
    files=json.loads(whitelist.read_text())['files']
    if len(files)!=100 or len(set(files))!=100 or 'index.html' not in files:raise ValueError('Original explicit100-file whitelist required')
    forbidden={'BREAKTHROUGH.md','PUBLIC_BANK.md','QUALIFYING_PROGRESS.md','PROGRESS.md','qualifying_progress.svg'}
    for name in files:
        path=Path(name)
        if '..' in path.parts or path.is_absolute() or path.suffix not in {'.md','.html','.json','.svg','.png'} or path.name in forbidden:
            raise ValueError('Only declared own static files may be checked')
    local=(ROOT/'index.html').read_bytes()
    if local!=(ROOT/'research_journey.html').read_bytes() or b'"strategy":"Public bank"' in local or b'FRESH complete13 qualifying' not in local:
        raise ValueError('Current root and journey must agree and retain only own milestones')
    payload,_=json.JSONDecoder().raw_decode(local.decode().split('const DATA = ',1)[1])
    milestones=payload['milestones']
    if len(milestones)!=87 or any(row['strategy']=='Public bank' or row['metric']=='qualifying' for row in milestones):
        raise ValueError('Exactly the declared87 own milestones and zero public-bank milestones required')
    build=json.loads(subprocess.check_output(['gh','api','repos/mquan/netflix_challenge/pages/builds/latest']))
    if build['status']!='built' or build['commit']!=commit:raise ValueError('Exact declared Pages commit must finish building first')
    def fetch(item):
        name,relative=item;expected=(ROOT/relative).read_bytes()
        url=BASE+quote(name,safe='/')
        # Root must also be current at the ordinary, unversioned user URL.
        if name:url+='?verified='+commit
        result=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','30',
            '-H','Cache-Control: no-cache',url],capture_output=True,check=True)
        actual=result.stdout
        return dict(path=name or '/',local_path=relative,bytes=len(actual),
            sha256=hashlib.sha256(actual).hexdigest(),exact=actual==expected)
    items=[(name,name) for name in files]+[('', 'index.html')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,items))
    if not all(row['exact'] for row in records):
        observed=dt.datetime.now(dt.timezone.utc)
        attempt=ROOT/('experiments/website-complete13-attempt-'+observed.strftime('%Y%m%dT%H%M%SZ')+'.json')
        attempt.write_text(json.dumps(dict(state='deployment_bytes_not_all_current',pages_commit=commit,records=records),indent=2)+'\n')
        raise ValueError('Declared deployment has stale bytes; attempt preserved, no success claimed')
    output.write_text(json.dumps(dict(state='complete_all_one_hundred_live_files_and_default_root_exact',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),pages_commit=commit,records=records,
        whitelist_sha256=hashlib.sha256(whitelist.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        default_root_matches_current_journey=True,own_milestones=len(milestones),public_bank_milestones=0),indent=2)+'\n')
    print('All100 whitelisted files and ordinary root URL match exact current bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pages-commit',required=True);args=parser.parse_args();main(args.pages_commit)
