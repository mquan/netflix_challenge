"""Full original-data admission routine; no CLI or unguarded production launch.

Requires the future source-bound owner/controller's original resource guard.
This routine alone establishes neither process ownership nor GPU permission.
No Torch import/operation, model fitting or recovered-label access occurs here.
"""
from contextlib import contextmanager,ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import numpy as np
import upstream_factored_rbm_inputs as inputs
from upstream_factored_rbm_rows import UserRows
from upstream_factored_rbm_data_audit import audit_reader,audit_known_projection,audit_zero_targets,identity_keys

ROW_COUNTS=dict(train=99072112,probe=1408395,probex=1128164,fulltrain=100480507,qual=2817131,train_3=98720013,probe_3=1760494)
PER_ROW=dict(user_idxs='int32',item_idxs='int16',ratings='int8',dates='int16',is_test='int8')
USER_FIELDS=dict(user_cnts='int32',uidx_to_uid='int32')
ITEM_FIELDS=dict(item_cnts='int32',item_years='int32',iidx_to_iid='int16')
TRAIN_FIELDS=('user_idxs','item_idxs','dates','ratings','is_test','user_cnts')
QUERY_FIELDS=('user_idxs','item_idxs','dates','user_cnts')


def load_declared_plan(root):
    plan_path=root/inputs.REPORT/'plan.json';proof_path=root/inputs.REPORT/'declaration-verification.json'
    p=json.loads(plan_path.read_text());v=json.loads(proof_path.read_text())
    if (v['state']!='verified_pending_original_input_declaration_and_observed30_synthetic_CPU_checks_only'
            or v['plan_sha256']!=inputs.sha(plan_path) or v['production_data_admitted'] is not False
            or v['qualifying_labels_read'] is not False or p!=inputs.expected_plan(root)):
        raise ValueError('Exact pending original-input declaration required')
    for name,digest in v['source_sha256'].items():
        if inputs.sha(root/name)!=digest:raise ValueError('Original declaration source changed')
    return p


def file_identity(value):
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


@contextmanager
def mapped(path,expected=None):
    """Decode and map the authenticated FD, never reopen its path for the map."""
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError('Bound regular NPY file required')
        if expected is not None and file_identity(before)!=expected['identity']:
            raise ValueError('NPY file differs from original fully hashed identity')
        if sys.platform=='darwin':fcntl.fcntl(stream.fileno(),48,1)
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError('Original native NPYv1/v2 header required')
        shape,fortran,dtype=readers[version](stream,max_header_size=4096)
        if dtype.hasobject or len(shape)!=1 or shape[0]<1:raise ValueError('Nonempty primitive native vector required')
        array=np.memmap(stream,dtype=dtype,mode='r',offset=stream.tell(),shape=shape,order='F' if fortran else 'C')
        try:yield array
        finally:
            array._mmap.close()
            if file_identity(before)!=file_identity(os.fstat(stream.fileno())) or file_identity(before)!=file_identity(path.stat()):
                raise ValueError('Mapped original file identity changed during array use')


def header_record(path,dtype,shape,expected=None):
    with mapped(path,expected) as array:
        if array.dtype!=np.dtype(dtype) or array.shape!=shape or not array.flags.c_contiguous:
            raise ValueError('Exact native NPY schema required: '+str(path))
        if path.stat().st_size!=array.offset+array.nbytes:
            raise ValueError('Exact header plus complete payload, without trailing bytes, required')
        return dict(dtype=array.dtype.str,shape=list(shape),payload_bytes=array.nbytes,header_bytes=array.offset)


@contextmanager
def arrays(root,paths,guard,records=None):
    """Open only named bound fields and close every map on success/failure."""
    opened={}
    with ExitStack() as stack:
        for key,relative in paths.items():
            guard();p=Path(relative)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Safe original array path required')
            expected=None if records is None else records[relative]
            opened[key]=stack.enter_context(mapped(root/p,expected))
        yield opened


def phase_paths(plan,phase):
    if phase not in ('known','qualifying'):raise ValueError('Declared original phase required')
    row=plan['phases'][phase]
    return ({k:row['training']+'/'+k+'.npy' for k in TRAIN_FIELDS},
            {k:row['query']+'/'+k+'.npy' for k in QUERY_FIELDS})


def statistics(reader,config):
    ni,nu,nd=reader.n_items,reader.n_users,len(reader.day_values)
    # Actual full predictive state, plus separate complete restart-state cost.
    shared=4*(ni*5*config['factors']+config['factors']*config['hidden']+ni*config['hidden']+ni*5+config['hidden'])
    bias=4*5*(nu+nd);model=shared+bias
    metadata=sum(getattr(reader,k).nbytes for k in ('training_offsets','query_offsets','query_order','day_offsets','day_values','positive_counts','visible_counts'))
    return dict(users=nu,items=ni,training_day_slots=nd,nonempty_users=len(reader.nonempty_users),
                predictive_parameter_bytes=model,model_plus_momentum_bytes=2*model,
                retained_reader_metadata_bytes=metadata,
                scope='Logical tensor/metadata bytes only; actual process/GPU/temporary/file serialization peaks still require measurement.')


def admit(root,guard):
    if not callable(guard):raise ValueError('External owned resource guard required')
    guard();plan=load_declared_plan(root);plan_path=root/inputs.REPORT/'plan.json'
    bindings={**plan['raw_input_sha256'],**plan['source_code_sha256'],**plan['original_reference_sha256'],
              **plan['provider_sha256'],**plan['prototype_proof_sha256']}
    records=inputs.verify_files(root,bindings,guard)
    schemas={}
    for relative in plan['raw_input_sha256']:
        guard();path=Path(relative)
        if path.name=='probex-row-map.npy':dtype,shape='int64',(plan['known_rows'],)
        else:
            split=path.parent.name;field=path.stem
            if field in PER_ROW:dtype,shape=PER_ROW[field],(ROW_COUNTS[split],)
            elif field in USER_FIELDS:dtype,shape=USER_FIELDS[field],(plan['users'],)
            elif field in ITEM_FIELDS:dtype,shape=ITEM_FIELDS[field],(plan['items'],)
            else:raise ValueError('Unexpected original array schema')
        schemas[relative]=header_record(root/path,dtype,shape,records[relative])
    results={}
    for phase in ('known','qualifying'):
        training_paths,query_paths=phase_paths(plan,phase);guard()
        with arrays(root,training_paths,guard,records) as tr,arrays(root,query_paths,guard,records) as qr:
            if (len(tr['user_idxs'])!=plan['phases'][phase]['training_rows']
                    or len(qr['user_idxs'])!=plan['phases'][phase]['query_rows']):
                raise ValueError('Complete original phase rows required')
            reader=UserRows(tr,qr,plan['items'])
            if reader.n_users!=plan['users']:raise ValueError('Complete original user universe required')
            semantic=audit_reader(reader,tr,qr,guard,plan['items'])
            result=dict(reader=semantic,statistics=statistics(reader,plan['exact_config']))
            if phase=='known':
                known_paths={k:'artifacts/upstream-repro/fresh/data/probex/'+k+'.npy' for k in ('user_idxs','item_idxs','dates','ratings')}
                extra_paths=dict(holdout_ratings=plan['phases'][phase]['query']+'/ratings.npy',mapping='artifacts/upstream-split3-pilot/probex-row-map.npy')
                with arrays(root,known_paths,guard,records) as known,arrays(root,extra_paths,guard,records) as extra:
                    qlabels=dict(qr,ratings=extra['holdout_ratings'])
                    result['projection']=audit_known_projection(tr,qlabels,known,extra['mapping'],plan['users'],plan['items'],guard)
                    if result['projection']['known_rows']!=plan['known_rows']:raise ValueError('Full known map required')
            else:
                with arrays(root,{'zero_targets':plan['phases'][phase]['query']+'/ratings.npy'},guard,records) as extra:
                    result['zero_targets']=audit_zero_targets(extra['zero_targets'],plan['phases'][phase]['query_rows'],guard)
                guard();keys=identity_keys(qr,plan['users'],plan['items'])
                if len(np.unique(keys))!=len(keys):raise ValueError('Unique original qualifying query identities required')
                result['native_query_identity_sha256']=hashlib.sha256(keys.astype('<i8').tobytes()).hexdigest()
                del keys
            results[phase]=result
            del reader
        guard()
    # Detect any changed/repointed input throughout the full semantic pass.
    for relative,record in records.items():
        guard();s=(root/relative).stat()
        if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=record['identity']:
            raise ValueError('Original input identity changed during complete semantic admission')
    if plan!=load_declared_plan(root):raise ValueError('Input declaration changed during admission')
    guard();return dict(state='complete_original63_payload_headers_both_phase_rows_full_holdout_map_and_zero_targets',
        input_declaration_sha256=inputs.sha(plan_path),bound_file_records=records,raw_array_schemas=schemas,phases=results,
        qualifying_labels_read=False,recovered_outcome_paths_read=False,models_fitted=0,
        remaining='Source-bound durable worker receipt, MPS/physical/RSS/swap/disk admission, full15+15 fit and saved-model query replay.')
