#!/usr/bin/env python3
"""Final admission gate for original paired-bank code/source expectations."""
import datetime as dt
from pathlib import Path
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-nn-pool6-original-bank-admission'
QUEUE=u.ROOT/'experiments/upstream-nn-pool6-full-refit-queue'
FULL=u.ROOT/'experiments/upstream-nn-pool6-full-refit'
DECL=u.ROOT/'experiments/upstream-native-nn-pool6-production/declaration.json'
BANK=u.ROOT/'experiments/upstream-item154-stream-bank'

def main():
    REPORT.mkdir(exist_ok=False)
    while not (QUEUE/'completion.json').exists():time.sleep(10)
    bm=u.read(BANK/'manifest.json');bp=u.read(BANK/'column-verification.json')
    if bp['state']!='complete' or bp['manifest_sha256']!=u.sha(BANK/'manifest.json'):raise ValueError('Original bank proof differs')
    expected={}
    for collection in [bm['source_sha256'],bm['code_sha256']]:
        for key,digest in collection.items():
            if key in expected and expected[key]!=digest:raise ValueError('Original bank provenance conflict')
            expected[key]=digest
    for key,digest in bm['files'].items():expected['artifacts/upstream-item154-stream-bank/'+key]=digest
    for key,digest in expected.items():
        if u.sha(u.ROOT/key)!=digest:raise ValueError('Original bank code/data/source changed: '+key)
    records=[]
    for entry in u.read(DECL)['families']:
        if entry['reuse_existing_full_base']:continue
        path=FULL/entry['development']/'plan.json';p=u.read(path)
        for key,digest in expected.items():
            if p['source_sha256'].get(key)!=digest:raise ValueError('Full fit did not preserve original bank expectation: '+key)
        records.append(dict(development=entry['development'],plan_sha256=u.sha(path)))
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),BANK/'manifest.json',BANK/'column-verification.json',DECL]}
    u.write(REPORT/'verification.json',dict(state='complete_original_bank_expectations_for_all_five_full_refits',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,original_bank_expected_sha256=expected,
        bindings=bindings,qualifying_labels_read=False,
        rule='Mandatory alongside each independent serialized-pipeline proof before ensemble admission. No replacement of original expected hashes by current hashes.'))
    print('All five full refits match original paired-bank code/source/data expectations',flush=True)

if __name__=='__main__':main()
