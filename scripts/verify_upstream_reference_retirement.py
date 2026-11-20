#!/usr/bin/env python3
"""Check retired public aliases are absent and all protected evidence remains."""
from pathlib import Path
import re
import tomli
import upstream_repro as u
import upstream_retire_reference_hardlinks as retire


def main():
    output=retire.REPORT/'independent-verification.json'
    if output.exists():raise ValueError('Existing retirement absence proof retained')
    plan=u.read(retire.REPORT/'plan.json');receipt=u.read(retire.REPORT/'verification.json')
    inventory=u.read(retire.INVENTORY)
    if receipt['state']!='complete_authorized_index_exact_public_reference_hardlink_retirement':
        raise ValueError('Complete authorized retirement receipt required')
    for relative,digest in plan['source_sha256'].items():
        if u.sha(u.ROOT/relative)!=digest:raise ValueError('Protected own proof, inventory, script or index changed')
    entries={r['path']:r for r in tomli.loads(retire.INDEX.read_text())['file']}
    allowed={str(p) for p in retire.ALLOWED}
    if set(plan['allowed_directories'])!=allowed:raise ValueError('Exact original public allowlist differs')
    aliases={frozenset(r['aliases']):r for r in inventory['records']}
    seen=set();total=0
    for row in receipt['records']:
        paths=row['paths'];key=frozenset(paths)
        if key not in aliases or key in seen or set(row['removed_paths'])!=set(paths):
            raise ValueError('Every complete original public alias group required')
        seen.add(key);before=aliases[key]
        if len(paths)!=row['original_nlink'] or len(paths)!=before['nlink']:
            raise ValueError('Original complete hardlink count differs')
        for name in paths:
            path=Path(name)
            if str(path.parent) not in allowed or path.exists() or path.is_symlink():
                raise ValueError('Retired path outside allowlist or still present')
        if len({Path(p).name for p in paths})!=1:raise ValueError('Same public-index filename required')
        entry=entries['preds_new/'+Path(paths[0]).name]
        if (row['bytes']!=entry['size'] or row['bytes']!=before['bytes'] or
            row['download_index_md5']!=entry['md5'] or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):
            raise ValueError('Original downloaded-byte catalog differs')
        total+=row['bytes']
    if total!=receipt['logical_bytes_discarded']:raise ValueError('Discarded logical size differs')
    u.write(output,dict(state='complete_independent_retired_public_alias_absence_and_protected_evidence_hashes',
        receipt_sha256=u.sha(retire.REPORT/'verification.json'),verifier_sha256=u.sha(Path(__file__)),
        source_sha256=plan['source_sha256'],groups=len(seen),logical_bytes_discarded=total,
        every_declared_retired_alias_absent=True,all_protected_proof_hashes_unchanged=True,
        original_deleted_bytes_rehashed_by_this_independent_verifier=False,
        qualifying_outcomes_read=False))
    print('Independent public-retirement absence/proof check:',len(seen),'groups',flush=True)


if __name__=='__main__':main()
