#!/usr/bin/env python3
"""Assemble the unchanged 85-column recipe with disclosed independent temporal phases."""
import argparse
from pathlib import Path
import time
import upstream_production_bank as core
import upstream_temporal_phase as phase

u=core.u
RUN=u.ROOT/'artifacts/upstream-phase-bank'
REPORT=u.ROOT/'experiments/upstream-phase-bank'
PRIVATE=REPORT/'assembly'
CORE_FILE=Path(core.__file__)
native_pair=core.choose_pair


def choose_pair(row,jobs,verify=False):
    receipt_path=phase.REPORT/'paired-receipt.json'
    if row['name'] not in phase.NAMES or not receipt_path.exists():
        return native_pair(row,jobs,verify)
    receipt=phase.verify_receipt(verify_files=verify)
    files=receipt['pairs'][row['name']]
    if not all((u.ROOT/path).is_file() for path in files.values()):return None
    return dict(name=row['name'],clip=row['clip'],producer=phase.JOB,
        kind='our-independent-source-phases',receipt=str(receipt_path.relative_to(u.ROOT)),
        receipt_sha256=u.sha(receipt_path),files=files,
        sha256={key:receipt['outputs'][path] for key,path in files.items()})


def configure():
    core.RUN,core.REPORT,core.choose_pair=RUN,PRIVATE,choose_pair


def inventory(verify=False):
    configure();result=core.inventory(verify)
    u.write(REPORT/'inventory.json',result)
    return result


def build():
    configure()
    if (REPORT/'manifest.json').exists():
        result=u.read(REPORT/'manifest.json')
        if result['adapter_sha256']!=u.sha(Path(__file__)) or result['core_sha256']!=u.sha(CORE_FILE):
            raise ValueError('Frozen independent-phase bank implementation changed')
        for path,digest in {**result['source_sha256'],**{str((RUN/name).relative_to(u.ROOT)):v for name,v in result['files'].items()}}.items():
            if u.sha(u.ROOT/path)!=digest:raise ValueError('Frozen independent-phase input changed: '+path)
        return result
    result=core.build()
    result.update(adapter_sha256=u.sha(Path(__file__)),core_sha256=u.sha(CORE_FILE),
        temporal_provenance='Four temporal ProbeX vectors from OUR completed fresh remote run, paired with the independent original fulltrain/qual phase. Same capacity, epochs and source arithmetic; no public pretrained prediction features.')
    result['source_sha256'].update({str(path.relative_to(u.ROOT)):u.sha(path) for path in [Path(__file__),CORE_FILE,Path(phase.__file__)]})
    u.write(REPORT/'manifest.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.watch:
        while inventory()['missing']:time.sleep(60)
        build()
    elif args.build:build()
    else:print({k:v for k,v in inventory().items() if k!='columns'})
