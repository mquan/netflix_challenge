#!/usr/bin/env python3
"""Fixed85 assembly using our hash-reconstructed, completed remote temporal job."""
import argparse
from pathlib import Path
import upstream_production_bank as core
import upstream_remote_temporal_pair as temporal

u=core.u
RUN=u.ROOT/'artifacts/upstream-recovered-temporal-bank'
REPORT=u.ROOT/'experiments/upstream-recovered-temporal-bank'
PRIVATE=REPORT/'assembly'
CORE_FILE=Path(core.__file__)
NATIVE_PAIR=core.choose_pair
PROVENANCE='Our completed original remote two-phase temporal source job; all eight selected native prediction hashes independently reconstructed exactly. Original25 epochs/150 factors; no downloaded pretrained prediction features. The currently running independent local temporal phase is not represented as completed.'


def choose_pair(row,jobs,verify=False):
    if row['name'] not in temporal.NAMES:return NATIVE_PAIR(row,jobs,verify)
    receipt=temporal.verify_receipt(verify_files=verify)
    files=receipt['pairs'][row['name']]
    return dict(name=row['name'],clip=row['clip'],producer=temporal.JOB,
        kind='our-completed-remote-native-job',receipt=str((temporal.REPORT/'paired-receipt.json').relative_to(u.ROOT)),
        receipt_sha256=u.sha(temporal.REPORT/'paired-receipt.json'),files=files,
        sha256={key:receipt['outputs'][path] for key,path in files.items()})


def configure():
    core.RUN,core.REPORT,core.choose_pair=RUN,PRIVATE,choose_pair


def inventory(verify=False):
    configure();result=core.inventory(verify);u.write(REPORT/'inventory.json',result);return result


def code_hashes():
    paths=[Path(__file__),CORE_FILE,Path(temporal.__file__),Path(temporal.phase.__file__)]
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}


def build():
    configure()
    if (REPORT/'manifest.json').exists():
        result=u.read(REPORT/'manifest.json')
        if result['adapter_code_sha256']!=code_hashes():raise ValueError('Frozen recovered-temporal bank code changed')
        for path,digest in {**result['source_sha256'],**{str((RUN/k).relative_to(u.ROOT)):v for k,v in result['files'].items()}}.items():
            if u.sha(u.ROOT/path)!=digest:raise ValueError('Frozen recovered-temporal bank source changed: '+path)
        return result
    receipt=temporal.verify_receipt()
    result=core.build()
    result.update(adapter_code_sha256=code_hashes(),temporal_provenance=PROVENANCE,
        original_remote_transfer_commit=receipt['transfer_commit'])
    result['source_sha256'].update(code_hashes())
    result['source_sha256'].update(receipt['source_sha256'])
    u.write(REPORT/'manifest.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.build:build()
    else:print({k:v for k,v in inventory(True).items() if k!='columns'})
