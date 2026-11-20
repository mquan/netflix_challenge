#!/usr/bin/env python3
"""Original complete MF60 split3 producer in its own frozen execution namespace."""
import argparse
from pathlib import Path
import upstream_split3_next3 as batch
import upstream_research_footprints as resources

u=batch.u
batch.RUN=u.ROOT/'artifacts/upstream-split3-mf60-original'
batch.TREE=batch.RUN/'source'
batch.REPORT=u.ROOT/'experiments/upstream-split3-mf60-original'
batch.JOBS={'s3-mf60':('misc',dict(EPOCHS='20',F='60'))}
batch.EPOCHS={'s3-mf60':20}
batch.PUBLISHED=['s3-mf60']
ORIGINAL_PREPARE=batch.prepare
# Preserve original admission thresholds while accounting for compressed peers
# and transiently exited processes through the separately verified scanner.
batch.latent.footprints=resources.footprints

def prepare():
    existed=(batch.REPORT/'plan.json').exists();plan=ORIGINAL_PREPARE()
    additions=[Path(__file__),Path(resources.__file__),u.ROOT/'scripts/upstream_split3_mf60_original_queue.py',
        u.ROOT/'scripts/verify_upstream_split3_mf60_original.py',u.ROOT/'SPLIT3_MF60_ORIGINAL_PLAN.md']
    if not existed:
        plan['bindings'].update({str(p.relative_to(u.ROOT)):u.sha(p) for p in additions})
        plan['admission']='Original full20/20 MF60, RAYON4/BLAS4, no feature/row/epoch cuts. Two stable samples of other research physical footprints below16GiB, compressor below5GiB, swap growth below32MiB/30s,free disk above20GiB;8GiB projected native memory reservation. Full train-prediction outputs retained (~0.82GB), above original16GiB disk floor+0.75GiB checkpoint writes+2GiB margin. Independent full source/native query/epoch/projection verification required. No qualifying outcomes or downloaded numeric features.'
        plan['selection']='First plain MF family missing from local complete full-probe3 bank; original registry recipe selected by structural source coverage only, not public/qualifying performance.'
        u.write(batch.REPORT/'plan.json',plan)
    for p in additions:
        if plan['bindings'].get(str(p.relative_to(u.ROOT)))!=u.sha(p):raise ValueError('Frozen complete MF60 execution changed')
    return plan

batch.prepare=prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else batch.run()
