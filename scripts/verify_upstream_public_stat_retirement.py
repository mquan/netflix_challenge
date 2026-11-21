#!/usr/bin/env python3
"""Read-only replay of retirement scope, durable journal and protected evidence."""
from pathlib import Path
import datetime as dt
import upstream_repro as u
import upstream_complete_trajectory_ops as ops

REPORT=u.ROOT/'experiments/upstream-retire-public-stat-caches'
PUBLIC=Path('/Users/quan/workspace/netflix_prize/artifacts/public-new')
NAMES=['late539-bank','early539-bank','MT8-537-bank','A2h537-bank','ensemble-tree-s1']


def main():
    output=REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Original independent retirement proof retained')
    plan=u.read(REPORT/'plan.json');proof=u.read(REPORT/'verification.json')
    expected={str(PUBLIC/(name+'.stats.npz.zst')) for name in NAMES}
    if (plan['state']!='declared_five_authorized_obsolete_public_bank_statistic_cache_discards'
        or proof['state']!='complete_five_authorized_public_stat_caches_retired_and_all_protected_proofs_equal'
        or proof['plan_sha256']!=ops.sha(REPORT/'plan.json')
        or len(plan['records'])!=5 or {r['path'] for r in plan['records']}!=expected
        or len(proof['removed'])!=5 or set(proof['removed'])!=expected
        or proof['numerical_own_files_removed'] or proof['models_or_checkpoints_removed']
        or not proof['original_metadata_preserved'] or proof['qualifying_outcomes_read'] or proof['processes_changed']
        or proof['logical_bytes_discarded']!=sum(r['identity']['bytes'] for r in plan['records'])):
        raise ValueError('Exact authorized five-cache retirement scope/proof required')
    for row in plan['records']:
        if Path(row['path']).exists() or row['identity']['nlink']!=1:
            raise ValueError('All five explicitly inventoried single-link cache files must be absent')
    for name,digest in {**plan['protected_proof_sha256'],**plan['preserved_metadata_sha256']}.items():
        if ops.sha(name)!=digest:raise ValueError('Protected scientific proof or original metadata differs')
    u.write(output,dict(state='complete_independent_public_stat_retirement_scope_absence_and_protected_evidence',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=ops.sha(REPORT/'plan.json'),
        retirement_verification_sha256=ops.sha(REPORT/'verification.json'),
        files=5,logical_bytes_discarded=proof['logical_bytes_discarded'],
        measured_free_space_gain_bytes=proof['free_after_bytes']-proof['free_before_bytes'],
        protected_scientific_proofs_rehashed=len(plan['protected_proof_sha256']),
        preserved_metadata_rehashed=len(plan['preserved_metadata_sha256']),
        verifier_sha256=ops.sha(Path(__file__)),qualifying_outcomes_read=False,
        limitation='Discarded cache bytes cannot be rehashed by this later verifier. Their original hashes/identities and classification remain in the fsynced pre-deletion journal; this replay verifies exact scope, absence and preserved expected hashes.'))


if __name__=='__main__':main()
