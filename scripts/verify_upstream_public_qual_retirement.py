"""Independent absence/provenance audit; deleted numerical bytes are not rehashed."""
from pathlib import Path
import datetime as dt
import upstream_retire_public_qual_arrays as cleanup


def verify():
    root=cleanup.REPORT;plan=cleanup.read(root/'plan.json');done=cleanup.read(root/'verification.json')
    partial=cleanup.read(root/'partial.json');anchors,index=cleanup.manifests()
    if (plan['state']!='prepared_exact_downloaded_public_qualifying_prediction_retirement'
        or done['state']!='complete_exact_downloaded_public_qualifying_prediction_retirement'
        or done['plan_sha256']!=cleanup.sha(root/'plan.json')
        or partial['plan_sha256']!=done['plan_sha256']
        or done['source_sha256']!=plan['source_sha256']
        or plan['allowed_roots']!=[str(p) for p in cleanup.ALLOWED]):
        raise ValueError('Complete exact prepared public retirement and original journals required')
    cleanup.check_bindings(plan['source_sha256'])
    removed=[];inodes=set();logical=0
    for row in plan['records']:
        s=row['metadata'];key=(s['device'],s['inode']);paths=[Path(p) for p in row['paths']]
        if key in inodes or len(paths)!=s['nlink'] or len(set(paths))!=len(paths):
            raise ValueError('Distinct complete original inode/alias accounting required')
        inodes.add(key);logical+=s['bytes']
        if any(not cleanup.allowed(p) or not p.name.endswith('.qual.npy') or p.exists() or p.is_symlink()
               or p.parent.resolve()!=p.parent for p in paths):
            raise ValueError('Every prepared public prediction alias must be absent')
        expected=[anchors[str(p)] for p in paths if str(p) in anchors]
        if expected!=row['download_manifest_entries'] or not expected:
            raise ValueError('Exact original manifest provenance alias required')
        if row['index_entries']!=[index[r['source']] for r in expected]:raise ValueError('Original public index differs')
        if any(r['sha256']!=row['sha256'] or r['md5']!=row['md5'] for r in expected):
            raise ValueError('Pre-deletion SHA and original index MD5 must agree')
        if any(index[r['source']]['size']!=s['bytes'] or index[r['source']]['md5']!=row['md5'] for r in expected):
            raise ValueError('Pre-deletion size/index must agree')
        removed.extend(row['paths'])
    if (removed!=partial['removed_paths'] or len(removed)!=done['removed_aliases']
        or logical!=plan['logical_candidate_bytes'] or logical!=done['logical_bytes_discarded']
        or len(inodes)!=done['inode_groups']
        or any(done[k] is not False for k in ('own_models_outputs_checkpoints_data_modified',
                'scientific_processes_signalled','qualifying_outcomes_read','earlier_archives_and_provenance_modified'))):
        raise ValueError('Complete removal accounting and retained scientific scope required')
    cleanup.write(root/'independent-verification.json',dict(
        state='complete_independent_public_qual_prediction_absence_provenance_and_protected_proof_check',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=cleanup.sha(root/'plan.json'),
        verification_sha256=cleanup.sha(root/'verification.json'),partial_sha256=cleanup.sha(root/'partial.json'),
        source_sha256=plan['source_sha256'],inode_groups=len(inodes),removed_aliases=len(removed),
        logical_bytes_discarded=logical,measured_free_gain=done['free_after']-done['free_before'],
        qualifying_outcomes_read=False,own_scientific_proof_bindings_unchanged=True,
        deleted_numerical_bytes_rehashed=False,
        scope='Post-deletion absence, original catalog/manifest and durable pre-deletion hashes replay; protected proof files remain exact. Does not independently rehash absent public bytes.'),exclusive=True)
    print('Independent public prediction retirement verification:',len(inodes),'inode groups',flush=True)


if __name__=='__main__':verify()
