"""Declare original raw input hashes; full payload admission is separate.

No command here starts training or imports Torch. Declaration reads existing
metadata/code only. File verification requires a caller-supplied resource guard.
Full row/map/disjointness replay remains a separate required admission step.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import upstream_factored_rbm_training as training

ROOT=Path(__file__).resolve().parents[1]
PIN='49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
REPORT=Path('experiments/upstream-factored-rbm-inputs')
COMMON={'dates.npy','is_test.npy','item_cnts.npy','item_idxs.npy','item_years.npy','ratings.npy','user_cnts.npy','user_idxs.npy'}
MAPS={'iidx_to_iid.npy','uidx_to_uid.npy'}
PROVIDERS={
    'weekend':'experiments/upstream-remote-weekend/plan.json',
    'split3':'experiments/upstream-split3-pilot/data-verification.json',
    'known':'experiments/upstream-split3-pilot/original-holdout-projection.json',
    'original':'experiments/upstream-original1200/execution-plan.json',
}


def digest_ok(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def compile_records(weekend,split3,known):
    """Preserve provider expectations, never derive new expected data hashes."""
    data={s+'/'+n for s in ('train','probe','probex','fulltrain','qual') for n in COMMON|(MAPS if s in ('train','fulltrain') else set())}
    extra={s+'/'+n for s in ('train_3','probe_3') for n in COMMON|(MAPS if s=='train_3' else set())}
    if (weekend['source_pin']!=PIN or split3['source_pin']!=PIN
            or weekend['qualifying_labels_read'] is not False or weekend['public_prediction_values_read'] is not False
            or split3['state']!='complete' or split3['rows']!=1128164 or split3['source_rows']!=1760494
            or split3['known_rows_absent_from_training'] is not True or split3['known_labels_match'] is not True
            or split3['qualifying_labels_read'] is not False
            or known['state']!='complete_original_holdout_projection' or known['known_rows']!=1128164 or known['source_rows']!=1408395
            or known['unique_exact_identity_mapping'] is not True or known['known_labels_equal'] is not True
            or known['qualifying_labels_read'] is not False):
        raise ValueError('Original known-only split/projection metadata required')
    if set(weekend['data_sha256'])!=data or set(weekend['split3_sha256'])!=extra:
        raise ValueError('Exact44 original and18 split3 raw array expectations required')
    expected_split={'source/data/'+n:v for n,v in weekend['split3_sha256'].items()}
    if split3['data_sha256']!=expected_split or split3['row_map_sha256']!=weekend['row_map_sha256']:
        raise ValueError('Original independent split/map expectations disagree')
    files={'artifacts/upstream-repro/fresh/data/'+n:s for n,s in weekend['data_sha256'].items()}
    files.update({'artifacts/upstream-split3-pilot/source/data/'+n:s for n,s in weekend['split3_sha256'].items()})
    known_keys={'artifacts/upstream-repro/fresh/data/'+split+'/'+name+'.npy'
                for split in ('probe','probex') for name in ('user_idxs','item_idxs','dates','ratings')}
    if set(known['input_sha256'])!=known_keys:raise ValueError('Exact original/known projection input set required')
    for key in known_keys:
        if known['input_sha256'][key]!=files[key]:raise ValueError('Original known-row array expectation differs')
    files['artifacts/upstream-split3-pilot/probex-row-map.npy']=split3['row_map_sha256']
    if len(files)!=63 or any(not digest_ok(s) for s in files.values()):
        raise ValueError('Exact63 original file hashes required')
    return files


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return h.hexdigest()


def expected_plan(root=ROOT):
    payload={k:json.loads((root/v).read_text()) for k,v in PROVIDERS.items()}
    files=compile_records(payload['weekend'],payload['split3'],payload['known'])
    if payload['original']['source_pin']!=PIN:raise ValueError('Original source pin required')
    references=['artifacts/upstream-original1200/source/'+n for n in ('src/rx.rs','src/bin/fit.rs','src/lib.rs','pipeline-3.toml')]
    original={n:payload['original']['bindings'][n] for n in references}
    for n,s in original.items():
        if sha(root/n)!=s:raise ValueError('Original mathematical source changed')
    code={}
    proof_dir=Path('experiments/upstream-factored-rbm-kernel')
    proofs=[proof_dir/n for n in ('verification.json','prediction-verification.json','rows-verification.json','training-verification.json')]
    logs=['cpu-checks.log','prediction-cpu-checks.log','rows-cpu-checks.log','training-cpu-checks.log']
    for relative,count,log in zip(proofs,(8,13,19,26),logs):
        record=json.loads((root/relative).read_text())
        if (record['production_data_read'] is not False or record['qualifying_labels_read'] is not False
                or record['test_exit_code']!=0 or record['test_count']!=count or record['actual_MPS_tested'] is not False
                or record.get('production_models_fitted',record.get('models_fitted'))!=0
                or record['test_log_sha256']!=sha(root/proof_dir/log)):
            raise ValueError('Only explicitly scoped synthetic prototypes admitted to declaration')
        for field in ('source_sha256','prediction_source_sha256','original_kernel_source_sha256'):
            for n,s in record.get(field,{}).items():
                if n in code and code[n]!=s:raise ValueError('Conflicting prototype source expectation')
                if sha(root/n)!=s:raise ValueError('Recorded prototype source changed')
                code[n]=s
    for name in ('scripts/upstream_factored_rbm_inputs.py','scripts/test_upstream_factored_rbm_inputs.py'):
        code[name]=sha(root/name)
    config=training.Config();config.validate()
    return dict(state='declared_original_raw_hashes_pending_full_payload_and_independent_row_admission',
        source_pin=PIN,provider_sha256={v:sha(root/v) for v in PROVIDERS.values()},
        prototype_proof_sha256={str(p):sha(root/p) for p in proofs},source_code_sha256=code,
        original_reference_sha256=original,raw_input_sha256=files,exact_config=asdict(config),
        phases={'known':dict(training='artifacts/upstream-split3-pilot/source/data/train_3',query='artifacts/upstream-split3-pilot/source/data/probe_3',training_rows=98720013,query_rows=1760494),
                'qualifying':dict(training='artifacts/upstream-repro/fresh/data/fulltrain',query='artifacts/upstream-repro/fresh/data/qual',training_rows=100480507,query_rows=2817131)},
        known_rows=1128164,users=480189,items=17770,expected_data_hashes_rebased=False,
        production_data_scanned=False,qualifying_labels_read=False,models_fitted=0,
        remaining=['Guarded full63 payload/hash/header validation including zero qualifying targets.',
                   'Separate complete raw-row/conditional/day/map/disjointness replay.',
                   'MPS/resource/ownership/durability admission and complete15+15 fitting.'])


def verify_files(root,bindings,guard):
    """Fully hash bound regular payloads with identity checks and bounded reads.

    This alone is not a complete data-admission proof. The caller must validate
    bindings against the frozen declaration and run the separate row verifier.
    """
    import fcntl
    if not callable(guard):raise ValueError('Explicit original resource/ownership guard required')
    results={}
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    for relative,expected in bindings.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not digest_ok(expected):
            raise ValueError('Safe relative bound path and original SHA256 required')
        guard();actual=root/path;h=hashlib.sha256()
        with actual.open('rb') as stream:
            before=os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):raise ValueError('Bound regular file required')
            if sys.platform=='darwin':fcntl.fcntl(stream.fileno(),48,1)  # F_NOCACHE
            while True:
                guard();block=stream.read(1048576)
                if not block:break
                h.update(block)
            after=os.fstat(stream.fileno())
            if identity(before)!=identity(after) or identity(after)!=identity(actual.stat()):
                raise ValueError('Original bound file identity changed during full hash')
        if h.hexdigest()!=expected:raise ValueError('Original expected hash mismatch: '+relative)
        results[relative]=dict(sha256=expected,bytes=after.st_size,identity=list(identity(after)))
    guard();return results


def declare():
    value=expected_plan();directory=ROOT/REPORT
    directory.mkdir(parents=True,exist_ok=False)
    path=directory/'plan.json'
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    for parent in (directory,directory.parent):
        fd=os.open(parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    print(value['state'],'raw files',len(value['raw_input_sha256']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['declare'])
    parser.parse_args();declare()
