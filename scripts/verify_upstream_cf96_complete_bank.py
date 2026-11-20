#!/usr/bin/env python3
"""Independently check each complete own CF96 input and paired alias."""
import hashlib
import numpy as np
import upstream_cf96_complete_bank as bank

u = bank.u


def main():
    output = bank.REPORT/'independent-verification.json'
    if output.exists():
        raise ValueError('Existing independent evidence preserved')
    manifest = u.read(bank.REPORT/'manifest.json')
    if (manifest['state'] != 'complete_all98_cf96_pairs_pending_independent_replay' or
        manifest['plan_sha256'] != u.sha(bank.REPORT/'plan.json') or
        manifest['qualifying_labels_read'] is not False or manifest['models_fitted'] != 0):
        raise ValueError('Unchanged label-free admission required')
    receipt, original, phases = bank.lineage()
    for relative, digest in manifest['source_sha256'].items():
        if u.sha(u.ROOT/relative) != digest:
            raise ValueError('Bound input changed')
    if any(manifest['source_sha256'].get(k) != v for k,v in original.items()):
        raise ValueError('Original full source closure missing')
    expected_names = [f'cfnade-96_ep{i:02}' for i in range(1,96)] + ['cfnade-96','cfnade-96r','cfnade-96r_ep95']
    if [r['name'] for r in manifest['all_rows']] != expected_names:
        raise ValueError('All95 epochs plus both terminal predictions required')
    distinct, aliases, seen = [], [], {}
    for row in manifest['all_rows']:
        digests = []
        for query, count in [('probex',1128164),('qual',2817131)]:
            path = bank.native.TREE/'preds_new'/(row['name']+'.'+query+'.npy')
            if row['paired_files'][query] != str(path.relative_to(u.ROOT)):
                raise ValueError('Unexpected original native query path')
            if u.sha(path) != receipt['outputs'][str(path.relative_to(bank.native.TREE))]:
                raise ValueError('Original finalizer hash differs')
            values = np.load(path, allow_pickle=False)
            if values.shape != (count,) or values.dtype != np.dtype('<f4') or not np.isfinite(values).all():
                raise ValueError('Wrong original prediction values')
            # Independent whole-vector min/max clipping, unlike chunked np.clip.
            clipped = np.maximum(np.minimum(values, np.float32(6)), np.float32(0))
            digests.append(hashlib.sha256(clipped.astype('<f4').tobytes()).hexdigest())
        if digests != row['clipped_pair_sha256']:
            raise ValueError('Independent clipped pair differs')
        key = tuple(digests)
        if key in seen:
            aliases.append(dict(name=row['name'],alias_of=seen[key],clipped_pair_sha256=digests))
        else:
            seen[key] = row['name']
            distinct.append(row)
    if distinct != manifest['rows'] or aliases != manifest['aliases']:
        raise ValueError('Complete paired alias partition differs')
    for relative,digest in manifest['source_sha256'].items():
        if u.sha(u.ROOT/relative) != digest:
            raise ValueError('Source changed during independent replay')
    u.write(output,dict(state='complete_independent_all98_own_cf96_pairs_and_exact_aliases',
        manifest_sha256=u.sha(bank.REPORT/'manifest.json'), source_sha256=manifest['source_sha256'],
        original_pairs=98, distinct_pairs=len(distinct), aliases=aliases, phases=phases,
        qualifying_labels_read=False, models_fitted=0, prediction_inference_replayed=False,
        public_prediction_values_read=False))
    print('Independent admission complete:',len(distinct),'distinct pairs',flush=True)


if __name__ == '__main__':
    main()
