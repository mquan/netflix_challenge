#!/usr/bin/env python3
"""Inventory every unused complete native pair without rating-based selection."""
import hashlib
from pathlib import Path
import re
import subprocess
import numpy as np
import upstream_repro as u

REPORT = u.ROOT / 'experiments/upstream-unused-native-pairs'
BASE = u.ROOT / 'experiments/upstream-own181-recent15-data'


def payload(path, rows, clip=False):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape != (rows,) or a.dtype != np.float32:
        raise ValueError('Complete original float32 query population required')
    h = hashlib.sha256()
    for lo in range(0, rows, 65536):
        part = a[lo:lo+65536]
        if not np.isfinite(part).all():
            raise ValueError('Nonfinite original prediction')
        h.update((np.clip(part, 0, 6) if clip else part).tobytes(order='C'))
    return h.hexdigest()


def main():
    if REPORT.exists():
        raise ValueError('Existing complete native coverage evidence preserved')
    u.validate_fresh()
    novelty_path = BASE / 'canonical-novelty-verification.json'
    novelty = u.read(novelty_path)
    manifest = u.read(BASE / 'manifest.json')
    if novelty['manifest_sha256'] != u.sha(BASE / 'manifest.json'):
        raise ValueError('Complete original181 canonical fingerprint proof required')
    replay = u.read(BASE / 'independent-verification.json')
    if (replay['state'] != 'complete_all_own181_retained_added_operator_bits_original_statistics_and_ridge_replay'
            or replay['manifest_sha256'] != u.sha(BASE / 'manifest.json')):
        raise ValueError('Complete own181 control proof required')
    old = set(novelty['original_raw_payload_sha256'] + novelty['recent_raw_payload_sha256'])
    if len(novelty['original_raw_payload_sha256']) != 166 or len(novelty['recent_raw_payload_sha256']) != 15:
        raise ValueError('Full original181 raw-column fingerprints required')
    registry_path = u.BASE / 'registry.json'
    jobs = u.read(registry_path)['jobs']
    receipts = []
    for p in sorted((u.BASE / 'fresh-receipts').glob('*.json')):
        r = u.read(p);name = p.stem
        if name not in jobs:
            continue
        if r.get('source_commit') != u.PIN or r.get('command') != jobs[name]['command'] or r.get('job') != name:
            raise ValueError('Original native receipt provenance differs: ' + name)
        if name.endswith('__epochs'):
            raise ValueError('Published precomputed epoch weights are ineligible')
        pairs = []
        for rel in sorted(r['outputs']):
            if not rel.endswith('.probex.npy'):
                continue
            stem = Path(rel).name.removesuffix('.probex.npy')
            if stem != name and not re.fullmatch(re.escape(name) + r'_ep\d+', stem):
                continue
            qual = rel.removesuffix('.probex.npy') + '.qual.npy'
            if qual in r['outputs']:
                pairs.append((stem, rel, qual))
        if pairs:
            receipts.append((p, r, pairs))
    REPORT.mkdir()
    bindings = {str(p.relative_to(u.ROOT)):u.sha(p) for p in
                [Path(__file__), novelty_path, BASE / 'manifest.json',
                 BASE / 'independent-verification.json', registry_path,
                 u.BASE / 'fresh-training-plan.json', u.BASE / 'fresh-data-receipt.json']}
    u.write(REPORT / 'plan.json', dict(state='fixed_all_complete_local_native_pairs_before_payload_or_score_checks',
            producers=[r['job'] for _,r,_ in receipts],
            candidates=[stem for _,_,pairs in receipts for stem,_,_ in pairs],
            source_sha256=bindings,
            rule='All completed local original terminal/epoch pairs in fixed producer/name order. No score threshold, epoch search, learned public coefficients or qualifying outcomes. Omit exact canonical clipped duplicates of original181; deduplicate new candidates only when both full raw payloads match. This is input admission for a known-only diagnostic, not qualifying production admission.'))
    rows=[];already=[];duplicates=[];seen={};producer_checks=[]
    for rp,r,pairs in receipts:
        bindings[str(rp.relative_to(u.ROOT))] = u.sha(rp)
        log = u.FRESH / ('preds_new/' + r['job'] + '.out')
        if str(log.relative_to(u.FRESH)) not in r['outputs'] or u.sha(log) != r['outputs'][str(log.relative_to(u.FRESH))]:
            raise ValueError('Bound original native log required: ' + r['job'])
        text=log.read_text()
        if 'trainx => probex' not in text or 'fulltrain => qual' not in text:
            raise ValueError('Both complete original native phases required')
        epochs=re.findall(r'\bn_epochs:\s*(\d+)',text)
        trace=[int(v) for v in re.findall(r'^Epoch\s+(\d+)\b',text,re.M)]
        if epochs:
            expected=list(range(1,int(epochs[0])+1))*2
            if len(set(epochs)) != 1 or trace != expected:
                raise ValueError('Original complete epoch budget differs: '+r['job'])
        elif trace:
            raise ValueError('Unexpected iterative trace without original budget')
        bindings[str(log.relative_to(u.ROOT))] = u.sha(log)
        producer_checks.append(dict(job=r['job'],epochs_per_phase=int(epochs[0]) if epochs else None,
                                    both_original_phases=True,complete_iterative_trace=bool(epochs),
                                    receipt_sha256=u.sha(rp)))
        for name,probe,qual in pairs:
            paths=dict(probex=u.FRESH/probe,qual=u.FRESH/qual)
            for split,p in paths.items():
                rel=probe if split=='probex' else qual
                if u.sha(p)!=r['outputs'][rel]:
                    raise ValueError('Own original native vector changed: '+name)
                bindings[str(p.relative_to(u.ROOT))]=r['outputs'][rel]
            canonical=payload(paths['probex'],1128164,True)
            raw_key=(payload(paths['probex'],1128164),payload(paths['qual'],2817131))
            row=dict(name=name,producer=r['job'],receipt=str(rp.relative_to(u.ROOT)),
                     paired_files={k:str(v.relative_to(u.ROOT)) for k,v in paths.items()},
                     canonical_clipped_payload_sha256=canonical,
                     raw_payload_sha256=dict(zip(['probex','qual'],raw_key)))
            if canonical in old:
                already.append(dict(name=name,reason='Exact original181 canonical clipped payload; no full qualifying-equivalence claim'))
            elif raw_key in seen:
                duplicates.append(dict(name=name,duplicate_of=seen[raw_key]))
            else:
                seen[raw_key]=name;rows.append(row)
    u.validate_fresh()
    for rel,digest in bindings.items():
        if u.sha(u.ROOT/rel)!=digest:
            raise ValueError('Bound original native source changed during admission')
    u.write(REPORT/'verification.json',dict(state='complete_all_unused_local_original_native_pairs_and_query_payloads',
            rows=rows,new_pairs=len(rows),original_candidates=sum(len(pairs) for _,_,pairs in receipts),
            producer_checks=producer_checks,original181_canonical_matches=already,exact_full_pair_duplicates=duplicates,
            source_sha256=bindings,plan_sha256=u.sha(REPORT/'plan.json'),
            qualifying_labels_read=False,public_prediction_values_read=False,
            scope='Complete own receipts, native phase traces and original paired query hashes/shapes checked. Fixed all-available input admission only; no fitting, performance selection, model replay or qualifying production equivalence claim.'))
    print('Complete unused native inventory:',len(rows),'new pairs;',len(already),'original181 canonical matches;',len(duplicates),'exact new pair aliases',flush=True)


if __name__=='__main__':main()
