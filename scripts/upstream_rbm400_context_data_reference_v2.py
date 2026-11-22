"""Complete known-only data/ridge assembly for both RBM400 context arms.

No CLI or owner is supplied. A separate reviewed3GiB supervisor must pass its
live guard. Actual execution additionally requires the complete extraction
and both-query source contracts enforced by context_inputs.
"""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import numpy as np
import upstream_rbm400_context_inputs_reference_v2 as inputs
import upstream_rbm400_latent_operators as operators
import upstream_factored_rbm_admission as admission
import upstream_factored_rbm_terminal_metal as terminal

ROOT=inputs.ROOT
REPORT=ROOT/'experiments/upstream-rbm400-context-data-reference-v2'
ART=ROOT/'artifacts/upstream-rbm400-context-data-reference-v2'
RAW_COLUMNS=inputs.preceding.ops.RAW391+list(range(466,561))
PENALTY=1000.


def save(name,value,guard):
    return terminal.durable_payload(ART/name,lambda stream:np.save(stream,value,allow_pickle=False),ROOT,guard)


def copy_bound(source,name,record,guard):
    """Copy original prepared bytes without changing frozen input link counts."""
    with source.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if admission.file_identity(before)!=record['identity']:raise ValueError('Authenticated prepared source FD required')
        def writer(output):
            while True:
                guard();block=stream.read(1048576)
                if not block:break
                output.write(block)
        result=terminal.durable_payload(ART/name,writer,ROOT,guard)
        if (admission.file_identity(before)!=admission.file_identity(os.fstat(stream.fileno()))
                or admission.file_identity(before)!=admission.file_identity(source.stat())
                or result['sha256']!=record['sha256']):
            raise ValueError('Complete original prepared bytes/identity required')
        return result


def scan_query(reader,retain_scalar,guard,known=None):
    """Visit every row, preserve old operator bits and all486 scalar streams."""
    if type(retain_scalar) is not bool:raise ValueError('Both-query mechanical alias decision required')
    dims=operators.dimensions(retain_scalar,True);d=dims['design_columns']
    gram=np.zeros((10,d,d));rhs=np.zeros((10,d))
    old=[hashlib.sha256(),hashlib.sha256()]
    arms={arm:[hashlib.sha256(),hashlib.sha256()] for arm in ('scalar','latent')}
    columns=[hashlib.sha256() for _ in RAW_COLUMNS];new=hashlib.sha256();seen=0
    for lo in range(0,len(reader.prediction),8192):
        guard();hi=min(lo+8192,len(reader.prediction));z,f=reader.block(lo,hi)
        if z.shape!=(hi-lo,d) or f.shape!=(hi-lo,dims['nonlinear_columns']):raise ValueError('Full latent operator widths required')
        original=np.column_stack([z[:,:-2],z[:,-1:]]) if retain_scalar else z
        original_f=np.ascontiguousarray(f[:,:712])
        old[0].update(original.tobytes());old[1].update(original_f.tobytes())
        for j,column in enumerate(RAW_COLUMNS):columns[j].update(np.clip(original[:,column],0,6).astype('<f4').tobytes())
        new.update(np.clip(reader.prediction[lo:hi],0,6).astype('<f4').tobytes())
        for arm,width in [('scalar',712+int(retain_scalar)),('latent',dims['nonlinear_columns'])]:
            arms[arm][0].update(z.tobytes());arms[arm][1].update(np.ascontiguousarray(f[:,:width]).tobytes())
        if known is not None:
            y,users,groups=known
            if not np.array_equal(reader.users[lo:hi],users[lo:hi]):raise ValueError('Canonical prepared and query user addresses must match')
            for group in range(10):
                keep=groups[lo:hi]==group;a=z[keep]
                gram[group]+=a.T@a;rhs[group]+=a.T@y[lo:hi][keep]
        seen+=hi-lo
    guard()
    return dict(rows=seen,original_operator_sha256=[h.hexdigest() for h in old],
                original_scalar_sha256=[h.hexdigest() for h in columns],new_scalar_sha256=new.hexdigest(),
                arm_operator_sha256={arm:[h.hexdigest() for h in hs] for arm,hs in arms.items()}),gram,rhs


def weights(gram,rhs):
    if (gram.ndim!=3 or gram.shape[0]!=10 or gram.shape[1]!=gram.shape[2]
            or rhs.shape!=gram.shape[:2] or not np.isfinite(gram).all() or not np.isfinite(rhs).all()):
        raise ValueError('Every finite ten-group sufficient statistic required')
    return np.stack([inputs.preceding.parent.n.ridge_weights(gram,rhs,np.arange(10)//2!=fold,PENALTY)
                     for fold in range(5)])


def admit(guard):
    if REPORT.exists() or ART.exists():raise ValueError('Prior complete/partial numerical attempt preserved; no automatic retry')
    declaration,records=inputs.source_declaration(guard)
    bindings=declaration['source_sha256']
    for name in ('scripts/upstream_rbm400_context_data_reference_v2.py','scripts/upstream_rbm400_context_inputs_reference_v2.py',
                 'scripts/verify_upstream_rbm400_context_data_reference_v2.py'):
        inputs.bind(bindings,ROOT/name)
    # New local implementation is authenticated too, before any output exists.
    records=inputs.provider.original.verify_bindings(bindings,guard)
    existing,identities=inputs.parent_pairs(declaration);new={}
    for query in inputs.COUNTS:
        relative=declaration['scalar_paths'][query]
        with admission.mapped(ROOT/relative,records[relative]) as prediction:
            if prediction.shape!=(inputs.COUNTS[query],):raise ValueError('Both complete new projected scalar vectors required')
            new[query]=inputs.clipped_hash(prediction,guard)
    alias=inputs.paired_alias(existing,new,identities);keep=alias['retain_scalar']
    terminal.durable_directory(REPORT,ROOT,guard);terminal.durable_directory(ART,ROOT,guard)
    prepared={};streams={};gram=rhs=None
    with ExitStack() as stack:
        arrays=[]
        for name,dtype in [('labels.npy','<f8'),('users.npy','<i4'),('groups.npy','i1')]:
            path=inputs.preceding.RUN/name;relative=str(path.relative_to(ROOT))
            record=records[relative]
            array=stack.enter_context(admission.mapped(path,record))
            if array.dtype!=np.dtype(dtype) or array.shape!=(inputs.COUNTS['probex'],):raise ValueError('Complete original labels/users/ten-groups required')
            arrays.append(array);prepared[name]=copy_bound(path,name,record,guard)
        y,users,groups=arrays
        if (not np.isin(y,[1,2,3,4,5]).all() or not np.isin(groups,np.arange(10)).all()
                or np.any(np.bincount(groups.astype(np.int64),minlength=10)==0)):
            raise ValueError('Original known outcomes and every canonical user group required')
        for query in inputs.COUNTS:
            with inputs.open_query(query,declaration,records,guard,retain_scalar=keep,latent=True) as reader:
                stream,g,r=scan_query(reader,keep,guard,arrays if query=='probex' else None)
            if (stream['rows']!=inputs.COUNTS[query]
                    or stream['original_operator_sha256']!=declaration['parent']['virtual_c_order_sha256'][query]
                    or stream['original_scalar_sha256']!=existing[query] or stream['new_scalar_sha256']!=new[query]):
                raise ValueError('Every original operator/scalar bit and the complete new pair must remain unchanged')
            streams[query]=stream
            if query=='probex':gram,rhs=g,r
        fitted=weights(gram,rhs);predictions=np.empty(len(y),dtype='<f4');seen=np.zeros(len(y),dtype=bool)
        with inputs.open_query('probex',declaration,records,guard,retain_scalar=keep,latent=False) as reader:
            for lo in range(0,len(y),8192):
                guard();hi=min(lo+8192,len(y));z,_=reader.block(lo,hi)
                for fold in range(5):
                    held=groups[lo:hi]//2==fold
                    predictions[lo:hi][held]=np.clip(z[held]@fitted[fold],1.05,4.95).astype('<f4')
                    seen[lo:hi][held]=True
        if not seen.all() or not np.isfinite(predictions).all():raise ValueError('All held-user ridge rows required')
        for name,value in [('gram.npy',gram),('rhs.npy',rhs),('weights.npy',fitted),('ridge.npy',predictions)]:prepared[name]=save(name,value,guard)
    if inputs.provider.original.verify_bindings(bindings,guard)!=records:
        raise ValueError('Complete original source identity must survive data assembly')
    result=dict(state='complete_both_context_arms_data_and_known_ridge_pending_independent_full_replay',
        source_sha256=bindings,alias=alias,arms={arm:operators.dimensions(keep,arm=='latent') for arm in ('scalar','latent')},
        virtual_c_order_sha256=streams,prepared=prepared,rows=inputs.COUNTS['probex'],qualifying_rows=inputs.COUNTS['qual'],
        lambda_fixed=PENALTY,folds=5,groups=10,qualifying_labels_read=False,neural_models_fitted=0,
        nonlinear_component_OOF_predictions_used_as_features=False,
        remaining='Separate complete operator/statistic/ridge proof and original-budget neural execution')
    inputs.extraction.exclusive_json(REPORT/'manifest.json',result,guard)
    return result
