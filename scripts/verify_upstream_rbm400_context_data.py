"""Independent full context-data/operator/ridge replay; no CLI or owner.

Actual execution requires a separately authorized3GiB scientific worker and
all complete original model/cache proofs. No recovered outcomes are opened.
"""
from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import numpy as np
import upstream_rbm400_context_data as data
import upstream_rbm400_context_inputs as inputs
import upstream_rbm400_latent_cache as cache
import upstream_factored_rbm_admission as admission

STATE='complete_independent_both_context_arms_all_operators_aliases_statistics_and_known_ridge'
NAMES={'labels.npy','users.npy','groups.npy','gram.npy','rhs.npy','weights.npy','ridge.npy'}


def known_score(predictions,labels):
    if (predictions.dtype!=np.dtype('<f4') or predictions.shape!=labels.shape or predictions.ndim!=1
            or not len(predictions) or not np.isfinite(predictions).all()
            or not np.isin(labels,[1,2,3,4,5]).all()):
        raise ValueError('Complete finite stored predictions and known outcomes required')
    # Float32 storage moves1.05 slightly below the declared lower bound. Use
    # the same float64 reclip convention as the canonical recovered scorer.
    clipped=np.clip(predictions.astype(np.float64),1.05,4.95)
    return math.sqrt(math.fsum(float(v) for v in (clipped-labels)**2)/len(labels))


def replay_query(reader,keep,guard,labels=None,users=None,groups=None):
    """Independent stream/statistic construction, without producer scan_query."""
    width=563 if keep else 562
    gram=np.zeros((10,width,width));rhs=np.zeros((10,width))
    raw_columns=list(range(62))+list(range(137,561))
    raw_hashes=[hashlib.sha256() for _ in raw_columns]
    original_z,original_f,new=[hashlib.sha256() for _ in range(3)]
    zh=hashlib.sha256();fh_scalar=hashlib.sha256();fh_latent=hashlib.sha256()
    count=0
    for start in range(0,len(reader.prediction),8192):
        guard();stop=min(start+8192,len(reader.prediction));z,f=reader.block(start,stop)
        if z.shape!=(stop-start,width) or f.shape!=(stop-start,862+int(keep)):
            raise ValueError('Every complete operator coordinate required')
        old=np.empty((stop-start,562),dtype=np.float64)
        old[:,:561]=z[:,:561];old[:,-1]=z[:,-1]
        original_z.update(old.tobytes());original_f.update(f[:,:712].tobytes(order='C'))
        for column,digest in zip(raw_columns,raw_hashes):digest.update(np.clip(old[:,column],0,6).astype('<f4').tobytes())
        new.update(np.clip(reader.prediction[start:stop],0,6).astype('<f4').tobytes())
        zh.update(z.tobytes());fh_scalar.update(f[:,:712+int(keep)].tobytes(order='C'));fh_latent.update(f.tobytes())
        if labels is not None:
            if not np.array_equal(reader.users[start:stop],users[start:stop]):raise ValueError('Every canonical user coordinate required')
            for group in range(10):
                selected=groups[start:stop]==group;values=z[selected];targets=labels[start:stop][selected]
                gram[group]=gram[group]+np.matmul(values.T,values)
                rhs[group]=rhs[group]+np.matmul(values.T,targets)
        count+=stop-start
    return dict(rows=count,original_operator_sha256=[original_z.hexdigest(),original_f.hexdigest()],
                original_scalar_sha256=[h.hexdigest() for h in raw_hashes],new_scalar_sha256=new.hexdigest(),
                arm_operator_sha256={'scalar':[zh.hexdigest(),fh_scalar.hexdigest()],
                                     'latent':[zh.hexdigest(),fh_latent.hexdigest()]}),gram,rhs


def coefficients(gram,rhs,stored):
    if (gram.shape[0]!=10 or gram.shape[1]!=gram.shape[2] or rhs.shape!=gram.shape[:2]
            or stored.shape!=(5,rhs.shape[1]) or not np.isfinite(stored).all()):
        raise ValueError('All five finite full-width coefficient vectors required')
    regularizer=np.diag(np.r_[np.full(rhs.shape[1]-1,1000.),0.])
    maxima=[]
    for fold in range(5):
        eligible=np.flatnonzero(np.arange(10)//2!=fold)
        matrix=gram[eligible].sum(axis=0)+regularizer;target=rhs[eligible].sum(axis=0)
        independent=np.linalg.solve(matrix,target)
        if stored[fold].tobytes()!=independent.tobytes():raise ValueError('Every independently solved eligible coefficient bit required')
        residual=float(np.max(np.abs(matrix@stored[fold]-target)))
        if residual>1e-12*(1+float(np.max(np.abs(target)))):
            raise ValueError('Every unpenalized-intercept ridge equation required')
        maxima.append(residual)
    return maxima


def verify(guard):
    declaration,source_records=inputs.source_declaration(guard)
    path=data.REPORT/'manifest.json';manifest=inputs.extraction.read(path)
    if (manifest['state']!='complete_both_context_arms_data_and_known_ridge_pending_independent_full_replay'
            or manifest['qualifying_labels_read'] is not False or manifest['neural_models_fitted']!=0
            or manifest['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or manifest['lambda_fixed']!=1000. or manifest['folds']!=5 or manifest['groups']!=10
            or manifest['rows']!=inputs.COUNTS['probex'] or manifest['qualifying_rows']!=inputs.COUNTS['qual']
            or set(manifest['prepared'])!=NAMES):
        raise ValueError('Complete declared both-arm data/ridge manifest required')
    expected=dict(declaration['source_sha256'])
    for name in ('scripts/upstream_rbm400_context_data.py','scripts/upstream_rbm400_context_inputs.py',
                 'scripts/verify_upstream_rbm400_context_data.py'):
        inputs.bind(expected,data.ROOT/name)
    if expected!=manifest['source_sha256']:raise ValueError('Exact complete original and new data source closure required')
    source_records=inputs.provider.original.verify_bindings(expected,guard)
    # Reconstruct the decision directly from both whole-query hashes.
    existing,identities=inputs.parent_pairs(declaration);pair={}
    for query in inputs.COUNTS:
        relative=declaration['scalar_paths'][query]
        with admission.mapped(data.ROOT/relative,source_records[relative]) as values:
            if values.shape!=(inputs.COUNTS[query],):raise ValueError('Every original scalar query row required')
            pair[query]=inputs.clipped_hash(values,guard)
    hits=[i for i in range(486) if all(existing[q][i]==pair[q] for q in inputs.COUNTS)]
    alias=dict(retain_scalar=not bool(hits),alias_of=identities[hits[0]] if hits else None,clipped_pair_sha256=pair)
    keep=alias['retain_scalar']
    if manifest['alias']!=alias:raise ValueError('Both-query first-original alias decision must independently match')
    if manifest['arms']!={arm:inputs.operators.dimensions(keep,arm=='latent') for arm in ('scalar','latent')}:
        raise ValueError('Complete both-arm widths required')
    bound={}
    for name,row in manifest['prepared'].items():
        p=Path(row['path'])
        if p!=data.ART/name or p.is_symlink():raise ValueError('Exact owned prepared output path required')
        bound[str(p.relative_to(data.ROOT))]=row['sha256']
    inputs.bind(bound,path)
    output_records=inputs.extraction.inputs.verify_files(data.ROOT,bound,guard)
    width=563 if keep else 562;n=inputs.COUNTS['probex']
    schema={'labels.npy':('<f8',(n,)),'users.npy':('<i4',(n,)),'groups.npy':('i1',(n,)),
            'gram.npy':('<f8',(10,width,width)),'rhs.npy':('<f8',(10,width)),
            'weights.npy':('<f8',(5,width)),'ridge.npy':('<f4',(n,))}
    try:
        with ExitStack() as stack:
            arrays={}
            for name,(dtype,shape) in schema.items():
                p=data.ART/name;arrays[name]=stack.enter_context(cache.mapped(p,output_records[str(p.relative_to(data.ROOT))],dtype,shape,guard))
            y,users,groups=[arrays[name] for name in ('labels.npy','users.npy','groups.npy')]
            if (not np.isin(y,[1,2,3,4,5]).all() or not np.isin(groups,np.arange(10)).all()
                    or np.any(np.bincount(groups.astype(np.int64),minlength=10)==0)):
                raise ValueError('Complete original known labels and every user group required')
            for name in ('labels.npy','users.npy','groups.npy'):
                original=str((inputs.preceding.RUN/name).relative_to(data.ROOT))
                if manifest['prepared'][name]['sha256']!=source_records[original]['sha256']:
                    raise ValueError('Prepared labels/users/groups must be exact copies')
            streams={};gram=rhs=None
            for query in inputs.COUNTS:
                with inputs.open_query(query,declaration,source_records,guard,retain_scalar=keep,latent=True) as reader:
                    stream,g,r=replay_query(reader,keep,guard,y if query=='probex' else None,
                                            users if query=='probex' else None,groups if query=='probex' else None)
                if (stream!=manifest['virtual_c_order_sha256'][query] or stream['rows']!=inputs.COUNTS[query]
                        or stream['original_operator_sha256']!=declaration['parent']['virtual_c_order_sha256'][query]
                        or stream['original_scalar_sha256']!=existing[query] or stream['new_scalar_sha256']!=pair[query]):
                    raise ValueError('Every independent original/new operator/scalar query bit required')
                streams[query]=stream
                if query=='probex':gram,rhs=g,r
            if gram.tobytes()!=arrays['gram.npy'].tobytes() or rhs.tobytes()!=arrays['rhs.npy'].tobytes():
                raise ValueError('Every independently reconstructed eligible statistic bit required')
            equations=coefficients(gram,rhs,arrays['weights.npy']);seen=np.zeros(n,dtype=bool)
            with inputs.open_query('probex',declaration,source_records,guard,retain_scalar=keep,latent=False) as reader:
                for start in range(0,n,8192):
                    guard();stop=min(start+8192,n);z,_=reader.block(start,stop)
                    for fold in range(5):
                        mask=groups[start:stop]//2==fold
                        prediction=np.clip(z[mask]@arrays['weights.npy'][fold],1.05,4.95).astype('<f4')
                        stored=arrays['ridge.npy'][start:stop][mask]
                        if prediction.tobytes()!=stored.tobytes():raise ValueError('Every held-user ridge prediction bit required')
                        seen[start:stop][mask]=True
            if not seen.all() or not np.isfinite(arrays['ridge.npy']).all():raise ValueError('Complete finite held-user predictions required')
            score=known_score(arrays['ridge.npy'],y)
        result=dict(state=STATE,manifest_sha256=inputs.extraction.inputs.sha(path),source_sha256=expected,
            all519_parent_streams_previously_admitted=True,all486_retained_scalar_streams_replayed=True,
            both_query_alias_independently_verified=True,every_both_arm_operator_bit_verified=True,
            every_group_statistic_and_ridge_prediction_bit_verified=True,ridge_equations=5*width,
            max_ridge_equation_residual=equations,known_held_user_ridge_rmse=score,
            known_ridge_is_five_user_fold_CV=True,qualifying_labels_read=False,neural_models_fitted=0,
            arms=manifest['arms'],rows=n,qualifying_rows=inputs.COUNTS['qual'])
    finally:
        try:
            if inputs.provider.original.verify_bindings(expected,guard)!=source_records:
                raise ValueError('Source closure identity changed during independent replay')
        finally:
            if inputs.extraction.inputs.verify_files(data.ROOT,bound,guard)!=output_records:
                raise ValueError('Prepared output identity changed during independent replay')
    inputs.extraction.exclusive_json(data.REPORT/'independent-verification.json',result,guard)
    return result
