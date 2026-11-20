#!/usr/bin/env python3
"""Verify an explicit own-only Pages publication and its ordinary root URL."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
import build_research_journey as journey

ROOT=Path(__file__).resolve().parents[1]
BASE='https://mquan.github.io/netflix_challenge/'


def main(args):
    whitelist=ROOT/args.whitelist;output=ROOT/args.output
    if output.exists():raise ValueError('Existing exact publication proof retained')
    if output.parent!=ROOT/'experiments' or whitelist.parent!=ROOT/'experiments':raise ValueError('Versioned repository evidence paths required')
    files=json.loads(whitelist.read_text())['files']
    if len(files)!=len(set(files)) or 'index.html' not in files or 'research_journey.html' not in files:raise ValueError('Explicit unique root/journey publication manifest required')
    for name in files:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix not in {'.md','.html','.json','.svg','.png'} or path.name in {'BREAKTHROUGH.md','PUBLIC_BANK.md','QUALIFYING_PROGRESS.md','PROGRESS.md','qualifying_progress.svg'}:
            raise ValueError('Only explicit own static pages, plots, docs and metadata permitted')
    root=(ROOT/'index.html').read_bytes()
    if root!=(ROOT/'research_journey.html').read_bytes():raise ValueError('Ordinary root and detailed journey must be identical')
    payload,_=json.JSONDecoder().raw_decode(root.decode().split('const DATA = ',1)[1]);rows=payload['milestones']
    if len(rows)!=args.milestones or any(r['strategy']=='Public bank' or r['metric']=='qualifying' for r in rows):raise ValueError('Exact declared own milestones and zero public-bank results required')
    record=journey.validate_fresh_status()
    build=json.loads(subprocess.check_output(['gh','api','repos/mquan/netflix_challenge/pages/builds/latest']))
    if build['status']!='built' or build['commit']!=args.pages_commit:raise ValueError('Exact declared commit must complete its Pages build')
    def fetch(pair):
        name,relative=pair;expected=(ROOT/relative).read_bytes();url=BASE+quote(name,safe='/')
        if name:url+='?verified='+args.pages_commit
        actual=subprocess.check_output(['curl','--fail','--silent','--show-error','--location','--max-time','30','-H','Cache-Control: no-cache',url])
        return dict(path=name or '/',local_path=relative,bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest(),exact=actual==expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,[(name,name) for name in files]+[('', 'index.html')]))
    now=dt.datetime.now(dt.timezone.utc)
    if not all(r['exact'] for r in results):
        attempt=output.with_name(output.stem+'-attempt-'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
        attempt.write_text(json.dumps(dict(state='live_bytes_not_all_current',pages_commit=args.pages_commit,records=results),indent=2)+'\n')
        raise ValueError('Stale deployment bytes retained as failed attempt; no success claimed')
    output.write_text(json.dumps(dict(state='complete_all_declared_live_files_and_default_root_exact',observed=now.isoformat(),
        pages_commit=args.pages_commit,static_files=len(files),own_milestones=len(rows),public_bank_milestones=0,
        default_root_matches_current_journey=True,latest_qualifying_verification_sha256=record['latest_qualifying_verification_sha256'],
        whitelist_sha256=hashlib.sha256(whitelist.read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=results),indent=2)+'\n')
    print('All',len(files),'static files and the ordinary root match; own milestones',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--whitelist',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pages-commit',required=True);parser.add_argument('--milestones',type=int,required=True);main(parser.parse_args())
