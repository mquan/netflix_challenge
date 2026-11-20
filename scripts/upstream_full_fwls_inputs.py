#!/usr/bin/env python3
"""Bounded full original scalar bank plus every declared native/nonlinear pair."""
import hashlib
from pathlib import Path
import numpy as np
import upstream_own13_fwls as context
import upstream_own352_fold_bag as original
import upstream_remote15h_all_snapshots as remote

u,checks=context.u,context.own6
REPORT=u.ROOT/'experiments/upstream-full-fwls-inputs'
DECL=u.ROOT/'OWN_FULL_FEATURE_WEIGHTED_STACKING_PLAN.md'
RAW_INDICES=list(range(62))+list(range(137,427))
COUNTS=dict(probex=1128164,qual=2817131)


def sources():
    families,gates,bindings=context.checked_sources()
    mp=remote.REPORT/'manifest.json';vp=remote.REPORT/'independent-verification.json'
    manifest=u.read(mp);proof=u.read(vp)
    if (proof['state']!='complete_independent_all48_native_known_projections_qualifying_counterparts_and_paired_aliases' or
        proof['manifest_sha256']!=u.sha(mp) or proof['plan_sha256']!=u.sha(remote.REPORT/'plan.json') or
        len(proof['records'])!=48 or not proof['all_canonical_identity_label_map_rows_rebuilt'] or
        proof['qualifying_labels_read'] is not False or proof['models_fitted']!=0 or
        len(manifest['rows'])!=45 or proof['additional_pairs']!=36):raise ValueError('All independently admitted remote native trajectories required')
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));value=u.sha(path) if expected is None else expected
        if key in bindings and bindings[key]!=value:raise ValueError('Original complete input source expectation conflicts: '+key)
        bindings[key]=value
    for key,value in proof['source_sha256'].items():bind(u.ROOT/key,value)
    additions=[]
    for row in manifest['rows']:
        bind(u.ROOT/row['paired_files']['probex'],row['known_sha256']);bind(u.ROOT/row['paired_files']['qual'],row['qual_sha256'])
        additions.append(dict(id='native:'+row['name'],paired_files=row['paired_files'],transform='clip0to6'))
    tp=u.ROOT/'experiments/upstream-remote15h-ext-return/s3-tx60o-v1/local-verification.json';tx=u.read(tp)
    if (tx['state']!='complete_local_compact_remote15h_extension_return_check' or tx['job']!='s3-tx60o' or
        tx['source_pin']!=u.PIN or tx['original_full_epochs_each']!=25 or
        tx['qualifying_labels_read'] is not False or tx['public_prediction_values_read'] is not False or
        not tx['original_recipe_trace_and_immutable_export_verified'] or
        not tx['local_full_known_map_and_training_disjointness_recomputed'] or
        not tx['native_qualifying_npy_bytes_reconstructed_exactly'] or len(tx['rows'])!=3):
        raise ValueError('Complete original25/25 TX60o compact own return required')
    for key,value in tx['bindings'].items():bind(u.ROOT/key,value)
    for row in tx['rows']:additions.append(dict(id='extension:'+row['name'],paired_files=row['paired_files'],transform='clip0to6'))
    for row in families:
        additions.append(dict(id='component:'+row['id'],paired_files=dict(probex=row['known_prediction'],qual=row['qualifying_prediction']),transform='retain'))
    entries=[dict(id='original352:%03d'%i,original_design_column=column,transform='original') for i,column in enumerate(RAW_INDICES)]+additions
    if len(entries)!=413 or len({r['id'] for r in entries})!=413:raise ValueError('All413 declared source entries required')
    for path in [Path(__file__),u.ROOT/'scripts/verify_upstream_full_fwls_inputs.py',DECL,mp,vp,remote.REPORT/'plan.json',tp]:bind(path)
    checks.verify_bindings(bindings)
    return entries,gates,bindings


class Inputs:
    def __init__(self,query,entries):
        if query not in COUNTS or len(entries)!=413:raise ValueError('Full original413-entry query required')
        self.query=query;self.entries=entries
        if query=='probex':
            m=u.read(original.study.data.REPORT/'manifest.json')
            self.parent=original.study.data.Inputs(m['original_rows'],m['recent_rows'])
        else:
            old=u.read(original.base.REPORT/'declaration.json')['rows']
            added=u.read(original.study.data.REPORT/'manifest.json')['recent_rows']
            self.parent=original.Inputs(old,added)

    def block(self,lo,hi):
        z,_=self.parent.block(lo,hi)
        result=np.empty((hi-lo,413),dtype=np.float64);result[:,:352]=z[:,RAW_INDICES]
        for i,row in enumerate(self.entries[352:],352):
            path=u.ROOT/row['paired_files'][self.query]
            if path.suffix=='.npy':
                reader=original.base.raw.Rows(path,(COUNTS[self.query],),np.float32)
                try:values=reader.read(lo,hi)
                finally:reader.close()
            else:
                if path.stat().st_size!=COUNTS[self.query]*4:raise ValueError('Complete component raw vector required')
                with path.open('rb') as stream:
                    stream.seek(lo*4);values=np.frombuffer(stream.read((hi-lo)*4),dtype='<f4')
            if values.shape!=(hi-lo,):raise ValueError('Every complete scalar block required')
            result[:,i]=np.clip(values,0,6) if row['transform']=='clip0to6' else values
        if not np.isfinite(result).all() or not np.array_equal(result,result.astype('<f4').astype(np.float64)):
            raise ValueError('Every original signal must be finite and round-trip float32 exactly')
        return result

    def close(self):self.parent.close()


def main():
    if REPORT.exists():raise ValueError('Existing full original FWLS input admission retained')
    entries,gates,bindings=sources();context.check_disk();REPORT.mkdir()
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,entries=entries,gates=gates,
        declaration_sha256=u.sha(DECL),worker_sha256=u.sha(Path(__file__)),
        verifier_sha256=u.sha(u.ROOT/'scripts/verify_upstream_full_fwls_inputs.py'),qualifying_labels_read=False))
    hashes={}
    for query,count in COUNTS.items():
        digests=[hashlib.sha256() for _ in entries];reader=Inputs(query,entries)
        try:
            for lo in range(0,count,8192):
                hi=min(lo+8192,count);values=reader.block(lo,hi).astype('<f4')
                for column,digest in enumerate(digests):digest.update(values[:,column].tobytes())
                context.check_disk()
        finally:reader.close()
        hashes[query]=[d.hexdigest() for d in digests]
        print('Complete original413-entry stream:',query,flush=True)
    selected=[];aliases=[];seen={}
    for i,row in enumerate(entries):
        key=(hashes['probex'][i],hashes['qual'][i])
        if key in seen:aliases.append(dict(index=i,id=row['id'],alias_of_index=seen[key],paired_signal_sha256=list(key)))
        else:seen[key]=i;selected.append(i)
    checks.verify_bindings(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_all413_original_signal_streams_and_full_paired_alias_partition_pending_independent_replay',
        plan_sha256=u.sha(REPORT/'plan.json'),entries=entries,gates=gates,source_sha256=bindings,
        raw_indices=RAW_INDICES,signal_sha256=hashes,selected_indices=selected,aliases=aliases,
        original_entries=413,distinct_signals=len(selected),interaction_dimension=len(selected)*14+1,
        qualifying_labels_read=False,models_fitted=0))
    print('All413 entries:',len(selected),'distinct full paired signals; independent replay remains',flush=True)


if __name__=='__main__':main()
