#!/usr/bin/env python3
"""Independently replay complete original413-entry streams and paired aliases."""
import hashlib
from pathlib import Path
import numpy as np
import upstream_full_fwls_inputs as experiment

u=experiment.u


def parent(query):
    original=experiment.original
    m=u.read(original.study.data.REPORT/'manifest.json')
    if query=='probex':return original.study.data.Inputs(m['original_rows'],m['recent_rows'])
    return original.Inputs(u.read(original.base.REPORT/'declaration.json')['rows'],m['recent_rows'])


def block(reader,entries,query,lo,hi):
    original,_=reader.block(lo,hi);result=np.empty((hi-lo,413),dtype=np.float64)
    for i,column in enumerate(list(range(62))+list(range(137,427))):result[:,i]=original[:,column]
    for i,row in enumerate(entries[352:],352):
        path=u.ROOT/row['paired_files'][query]
        if path.suffix=='.npy':
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if array.shape!=(experiment.COUNTS[query],) or array.dtype!=np.dtype('<f4'):raise ValueError('Complete original native scalar required')
            values=np.array(array[lo:hi],copy=True);del array
        else:
            array=np.memmap(path,mode='r',dtype='<f4')
            if array.shape!=(experiment.COUNTS[query],):raise ValueError('Complete original component scalar required')
            values=np.array(array[lo:hi],copy=True);del array
        result[:,i]=np.clip(values,0,6) if row['transform']=='clip0to6' else values
    if not np.isfinite(result).all() or not np.array_equal(result,result.astype('<f4').astype(np.float64)):
        raise ValueError('Complete original finite float32-representable operators required')
    return result


def main():
    report=experiment.REPORT;output=report/'independent-verification.json'
    if output.exists():raise ValueError('Existing independent full input replay retained')
    plan=u.read(report/'plan.json');m=u.read(report/'manifest.json')
    if (m['state']!='complete_all413_original_signal_streams_and_full_paired_alias_partition_pending_independent_replay' or
        m['plan_sha256']!=u.sha(report/'plan.json') or plan['worker_sha256']!=u.sha(Path(experiment.__file__)) or
        plan['verifier_sha256']!=u.sha(Path(__file__)) or plan['declaration_sha256']!=u.sha(experiment.DECL) or
        m['qualifying_labels_read'] is not False or m['models_fitted']!=0):raise ValueError('Complete original full-bank input freeze required')
    entries,gates,bindings=experiment.sources()
    if (entries!=m['entries'] or entries!=plan['entries'] or gates!=m['gates'] or gates!=plan['gates'] or
        bindings!=m['source_sha256'] or bindings!=plan['source_sha256'] or m['raw_indices']!=list(range(62))+list(range(137,427))):
        raise ValueError('Original complete input list/gates/source expectations differ')
    hashes={}
    for query,count in experiment.COUNTS.items():
        reader=parent(query);digests=[hashlib.sha256() for _ in entries]
        try:
            for lo in range(0,count,3072):
                hi=min(lo+3072,count);values=block(reader,entries,query,lo,hi).astype('<f4')
                for column,digest in enumerate(digests):digest.update(values[:,column].tobytes())
                experiment.context.check_disk()
        finally:reader.close()
        hashes[query]=[d.hexdigest() for d in digests]
        print('Independently reconstructed every original413 signal:',query,flush=True)
    selected=[];aliases=[];seen={}
    for i,row in enumerate(entries):
        key=(hashes['probex'][i],hashes['qual'][i])
        if key in seen:aliases.append(dict(index=i,id=row['id'],alias_of_index=seen[key],paired_signal_sha256=list(key)))
        else:seen[key]=i;selected.append(i)
    if (hashes!=m['signal_sha256'] or selected!=m['selected_indices'] or aliases!=m['aliases'] or
        m['original_entries']!=413 or m['distinct_signals']!=len(selected) or m['interaction_dimension']!=len(selected)*14+1):
        raise ValueError('Every paired scalar identity and complete exact alias partition must replay')
    experiment.checks.verify_bindings(bindings)
    u.write(output,dict(state='complete_independent_all413_original_input_streams_and_exact_complete_paired_aliases',
        observed=experiment.checks.stamp(),manifest_sha256=u.sha(report/'manifest.json'),plan_sha256=u.sha(report/'plan.json'),
        verifier_sha256=u.sha(Path(__file__)),source_sha256=bindings,all_original_entries=413,
        all_known_and_qualifying_operator_bits_equal=True,distinct_signals=len(selected),
        interaction_dimension=len(selected)*14+1,qualifying_labels_read=False,models_fitted=0))
    print('Complete original full-bank scalar/alias replay:',len(selected),'distinct signals',flush=True)


if __name__=='__main__':main()
