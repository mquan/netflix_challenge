#!/usr/bin/env python3
"""Independently replay exact retirement scope, absence and preserved hashes."""
import argparse
import datetime as dt
import json
from pathlib import Path
import upstream_retire_remaining_public_stats as run


def main(attempt):
    report=run.report(attempt);output=report/'independent-verification.json'
    if output.exists():raise ValueError('Original independent evidence retained')
    plan=json.loads((report/'plan.json').read_text());proof=json.loads((report/'verification.json').read_text())
    expected={str(run.OLD/(s[0]+'.stats.npz.zst')) for s in run.SPECS}
    if (plan['state']!='declared_three_authorized_obsolete_public_dependent_stat_archive_discards' or
        proof['state']!='complete_three_authorized_public_stat_archives_retired' or
        proof['plan_sha256']!=run.ops.sha(report/'plan.json') or len(plan['records'])!=3 or
        {r['path'] for r in plan['records']}!=expected or set(proof['removed'])!=expected or len(proof['removed'])!=3 or
        proof['independent_model_checkpoint_prediction_or_dataset_files_removed'] is not False or
        proof['original_metadata_preserved'] is not True or proof['qualifying_outcomes_read'] is not False or
        proof['processes_changed'] is not False or
        proof['logical_bytes_discarded']!=sum(r['identity']['bytes'] for r in plan['records'])):
        raise ValueError('Exact three-cache public retirement contract required')
    specs={str(run.OLD/(s[0]+'.stats.npz.zst')):s for s in run.SPECS}
    for row in plan['records']:
        path=Path(row['path']);spec=specs[row['path']]
        if path.exists() or row['identity']['nlink']!=1 or row['sha256']!=spec[4]:
            raise ValueError('Exact declared public archives must be absent')
        if run.classify(spec,json.loads((run.OLD/(spec[0]+'.inputs.json')).read_text()))!=row['matched_downloaded_public_inputs']:
            raise ValueError('Public provenance differs')
    for name,digest in {**plan['protected_scientific_proof_sha256'],**plan['preserved_metadata_sha256']}.items():
        if run.ops.sha(name)!=digest:raise ValueError('Preserved own proof or metadata changed')
    run.prior.write(output,dict(state='complete_independent_three_public_stat_archive_retirement',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=run.ops.sha(report/'plan.json'),
        retirement_sha256=run.ops.sha(report/'verification.json'),files=3,
        logical_bytes_discarded=proof['logical_bytes_discarded'],measured_free_space_gain_bytes=proof['free_after_bytes']-proof['free_before_bytes'],
        protected_scientific_proofs_rehashed=len(plan['protected_scientific_proof_sha256']),
        preserved_metadata_rehashed=len(plan['preserved_metadata_sha256']),
        verifier_sha256=run.ops.sha(Path(__file__)),qualifying_outcomes_read=False,
        limitation='Original discarded bytes are pinned by the fsynced pre-deletion SHA/identity journal; this later replay proves scope, absence and preservation.'))
    print('Independent three-cache retirement verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);main(p.parse_args().attempt)
