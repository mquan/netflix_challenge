"""Independent array-semantic admission checks, without production file I/O.

Reconstructs expectations from raw arrays instead of calling producer metadata
helpers. File hashes/schema/source ownership and actual resource admission must
be established by the future production controller before invoking these checks.
"""
import hashlib
import numpy as np


def same(actual,expected,message):
    if not np.array_equal(actual,expected):raise ValueError(message)


def audit_reader(reader,tr,qr,guard,n_items,users_per_batch=500,block_rows=65536):
    if (not callable(guard) or type(users_per_batch) is not int or users_per_batch<1
            or type(block_rows) is not int or block_rows<1):raise ValueError('Bounded replay and explicit resource guard required')
    guard();nu,ni=len(tr['user_cnts']),n_items
    if (type(ni) is not int or ni<1 or nu<1 or reader.n_users!=nu or reader.n_items!=ni):
        raise ValueError('Declared item/user universe differs from producer')
    for raw,names in ((tr,('user_idxs','item_idxs','dates','ratings','is_test','user_cnts')),
                      (qr,('user_idxs','item_idxs','dates','user_cnts'))):
        n=len(raw['user_idxs'])
        if (raw['user_cnts'].shape!=(nu,) or any(raw[k].ndim!=1 or raw[k].dtype.kind not in 'iu' for k in names)
                or any(len(raw[k])!=n for k in names if k!='user_cnts')):
            raise ValueError('Exact raw integer row/count arrays required')
    counts=np.zeros(nu,dtype=np.int64);positive_counts=np.zeros(nu,dtype=np.int64)
    query_counts=np.zeros(nu,dtype=np.int64);visible=np.zeros((ni,5),dtype=np.int64)
    for raw,training in ((tr,True),(qr,False)):
        previous=-1
        for lo in range(0,len(raw['user_idxs']),block_rows):
            guard();hi=min(lo+block_rows,len(raw['user_idxs']))
            u,i,date=[raw[k][lo:hi].astype(np.int64) for k in ('user_idxs','item_idxs','dates')]
            if (np.any(u<0) or np.any(u>=nu) or np.any(i<0) or np.any(i>=ni)
                    or np.any(date<-32768) or np.any(date>32767)):
                raise ValueError('Native raw coordinate bounds required')
            if training:
                if len(u) and (u[0]<previous or np.any(u[1:]<u[:-1])):
                    raise ValueError('Raw training user order is not contiguous')
                if len(u):previous=int(u[-1])
                np.add.at(counts,u,1)
                mask=raw['is_test'][lo:hi]
                if np.any((mask!=0)&(mask!=1)):raise ValueError('Binary native training mask required')
                selected=mask==0;r=raw['ratings'][lo:hi][selected].astype(np.int64)
                if np.any(r<1) or np.any(r>5):raise ValueError('Valid positive training ratings required')
                np.add.at(positive_counts,u[selected],1);np.add.at(visible,(i[selected],r-1),1)
            else:np.add.at(query_counts,u,1)
    same(counts,tr['user_cnts'],'Actual training counts disagree')
    same(query_counts,qr['user_cnts'],'Actual query counts disagree')
    offsets=np.empty(nu+1,dtype=np.int64);offsets[0]=0;np.cumsum(counts,out=offsets[1:])
    qo=np.empty(nu+1,dtype=np.int64);qo[0]=0;np.cumsum(query_counts,out=qo[1:])
    # Independent explicit secondary key preserves source query row order.
    qorder=np.lexsort((np.arange(len(qr['user_idxs']),dtype=np.int64),qr['user_idxs']))
    same(reader.training_offsets,offsets,'Producer training offsets differ')
    same(reader.query_offsets,qo,'Producer query offsets differ')
    same(reader.query_order,qorder,'Producer stable query row permutation differs')
    same(reader.positive_counts,positive_counts,'Producer positive-history eligibility differs')
    same(reader.nonempty_users,np.flatnonzero(positive_counts),'Producer dropped or added eligible users')
    same(reader.visible_counts,visible,'Producer masked visible counts differ')
    if reader.day_offsets.shape!=(nu+1,) or reader.day_offsets[0]!=0:
        raise ValueError('Complete global training-day offsets required')
    hashes={k:hashlib.sha256() for k in ('positive_source_rows','conditional_pairs','day_slots','query_source_rows')}
    total_days=total_conditional=total_positive=total_queries=0
    for start in range(0,nu,users_per_batch):
        guard();ids=np.arange(start,min(nu,start+users_per_batch),dtype=np.int64)
        b=reader.batch(ids,training=False);s,q=b['support'],b['query']
        same(b['global_users'],ids,'Producer changed batch user identity/order')
        for a in list(s.values())+list(q.values())+[b[k] for k in ('global_users','global_days','training_rows','query_rows')]:
            if a.ndim!=1 or a.dtype!=np.dtype('int64'):raise ValueError('Exact one-dimensional int64 batch coordinates required')
        pc=dc=cc=qc=0
        for local,user in enumerate(ids):
            lo,hi=offsets[user:user+2]
            raw_rows=np.arange(lo,hi,dtype=np.int64)
            rows=raw_rows[tr['is_test'][lo:hi]==0];n=len(rows)
            known_days=sorted(set(int(x) for x in tr['dates'][rows]));nd=len(known_days)
            lookup={date:dc+j for j,date in enumerate(known_days)}
            expected_days=np.array([lookup[int(date)] for date in tr['dates'][rows]],dtype=np.int64)
            same(b['training_rows'][pc:pc+n],rows,'Positive source rows were reordered/dropped')
            same(s['users'][pc:pc+n],np.full(n,local),'Positive local user ownership differs')
            same(s['items'][pc:pc+n],tr['item_idxs'][rows],'Positive item bits differ')
            same(s['ratings'][pc:pc+n],tr['ratings'][rows].astype(np.int64)-1,'Positive rating bits differ')
            same(s['days'][pc:pc+n],expected_days,'Positive date→day-slot mapping differs')
            if reader.day_offsets[user]!=total_days or reader.day_offsets[user+1]!=total_days+nd:
                raise ValueError('Producer global training-day offsets differ')
            same(reader.day_values[total_days:total_days+nd],known_days,'Unique sorted raw training days differ')
            same(b['global_days'][dc:dc+nd],np.arange(total_days,total_days+nd),'Local→global day-slot identity differs')
            same(s['day_owner'][dc:dc+nd],np.full(nd,local),'Training-day slot belongs to another user')
            qrows=qorder[qo[user]:qo[user+1]];nq=len(qrows)
            same(b['query_rows'][qc:qc+nq],qrows,'Original query row IDs differ')
            same(q['users'][qc:qc+nq],np.full(nq,local),'Query user ownership differs')
            same(q['items'][qc:qc+nq],qr['item_idxs'][qrows],'Query item bits differ')
            expected_qdays=np.array([lookup.get(int(date),-1) for date in qr['dates'][qrows]],dtype=np.int64)
            same(q['days'][qc:qc+nq],expected_qdays,'Known/unseen query-day mapping differs')
            conditional=sorted(set(int(x) for x in tr['item_idxs'][lo:hi])|set(int(x) for x in qr['item_idxs'][qrows]));nc=len(conditional)
            same(s['conditional_items'][cc:cc+nc],conditional,'Full masked/query conditional item union differs')
            same(s['conditional_users'][cc:cc+nc],np.full(nc,local),'Conditional local user ownership differs')
            hashes['positive_source_rows'].update(rows.astype('<i8').tobytes())
            hashes['query_source_rows'].update(qrows.astype('<i8').tobytes())
            hashes['conditional_pairs'].update(np.column_stack((np.full(nc,user,dtype=np.int64),conditional)).astype('<i8').tobytes())
            hashes['day_slots'].update(np.column_stack((np.full(nd,user,dtype=np.int64),known_days)).astype('<i8').tobytes())
            pc+=n;dc+=nd;cc+=nc;qc+=nq
            total_days+=nd;total_conditional+=nc;total_positive+=n;total_queries+=nq
        lengths={k:pc for k in ('items','users','ratings','days')}
        lengths.update(conditional_items=cc,conditional_users=cc,day_owner=dc)
        if any(len(s[k])!=n for k,n in lengths.items()):raise ValueError('Extra/missing producer support coordinates')
        if (len(b['training_rows'])!=pc or len(b['global_days'])!=dc or len(b['query_rows'])!=qc
                or any(len(q[k])!=qc for k in ('items','users','days')) or len(q['day_owner'])!=dc):
            raise ValueError('Extra/missing producer day/query/source coordinates')
        same(q['day_owner'],s['day_owner'],'Query and training day ownership differ');guard()
    if total_days!=len(reader.day_values) or total_queries!=len(qr['user_idxs']) or total_positive!=int(positive_counts.sum()):
        raise ValueError('Full raw/query/day coverage required')
    guard();return dict(state='complete_independent_raw_to_reader_all_array_semantics',raw_training_rows=len(tr['user_idxs']),
        positive_ratings=total_positive,query_rows=total_queries,conditional_pairs=total_conditional,training_day_slots=total_days,
        nonempty_users=int(np.count_nonzero(positive_counts)),stream_sha256={k:v.hexdigest() for k,v in hashes.items()},
        scope='Array semantics only; complete source/payload admission and actual production execution are separate.')


def identity_keys(raw,n_users,n_items):
    u,i,date=[np.asarray(raw[k]) for k in ('user_idxs','item_idxs','dates')]
    if (u.ndim!=1 or i.shape!=u.shape or date.shape!=u.shape or any(x.dtype.kind not in 'iu' for x in (u,i,date))
            or type(n_users) is not int or type(n_items) is not int or n_users<1 or n_items<1 or n_users*n_items*65536>=2**63
            or np.any(u<0) or np.any(u>=n_users) or np.any(i<0) or np.any(i>=n_items)
            or np.any(date<-32768) or np.any(date>32767)):
        raise ValueError('Bounded unique native row identity fields required')
    return (u.astype(np.int64)*n_items+i.astype(np.int64))*65536+date.astype(np.int64)+32768


def audit_known_projection(training,known_query,known,mapping,n_users,n_items,guard,block_rows=262144):
    if not callable(guard) or type(block_rows) is not int or block_rows<1:raise ValueError('Bounded replay/resource guard required')
    guard();qkeys=identity_keys(known_query,n_users,n_items);keys=identity_keys(known,n_users,n_items)
    if len(np.unique(qkeys))!=len(qkeys) or len(np.unique(keys))!=len(keys):raise ValueError('Unique native/known query identities required')
    if (mapping.dtype!=np.dtype('int64') or mapping.shape!=(len(keys),) or np.any(mapping<0) or np.any(mapping>=len(qkeys))):
        raise ValueError('Full exact known-row map required')
    same(qkeys[mapping],keys,'Known row identities do not match source map')
    if (known['ratings'].shape!=keys.shape or known_query['ratings'].shape!=qkeys.shape
            or any(raw['ratings'].dtype.kind not in 'iu' for raw in (known,known_query))):
        raise ValueError('Complete known-label arrays required')
    if any(np.any(raw['ratings']<1) or np.any(raw['ratings']>5) for raw in (known,known_query)):
        raise ValueError('Original known/holdout ratings must be1..5')
    same(known_query['ratings'][mapping],known['ratings'],'Known labels disagree with native holdout')
    sorted_keys=np.sort(qkeys);positive=0
    n=len(training['user_idxs'])
    if any(training[k].shape!=(n,) for k in ('user_idxs','item_idxs','dates','is_test')):
        raise ValueError('Full original training identity/mask rows required')
    for lo in range(0,n,block_rows):
        guard();hi=min(lo+block_rows,n);mask=training['is_test'][lo:hi]
        if np.any((mask!=0)&(mask!=1)):raise ValueError('Binary original training mask required')
        raw={k:training[k][lo:hi][mask==0] for k in ('user_idxs','item_idxs','dates')}
        values=identity_keys(raw,n_users,n_items);positive+=len(values)
        at=np.searchsorted(sorted_keys,values);inside=at<len(sorted_keys)
        if np.any(sorted_keys[at[inside]]==values[inside]):raise ValueError('Native holdout identity appears in positive training')
    guard();return dict(state='complete_independent_full_native_holdout_and_known_projection_positive_training_disjointness',
        known_rows=len(keys),source_query_rows=len(qkeys),raw_training_rows=n,positive_training_rows=positive,
        known_labels_equal=True,known_identities_absent_from_positive_training=True,native_holdout_identities_absent_from_positive_training=True,
        mapping_bits_sha256=hashlib.sha256(mapping.astype('<i8').tobytes()).hexdigest())


def audit_zero_targets(targets,expected_rows,guard,block_rows=262144):
    if not callable(guard) or type(block_rows) is not int or block_rows<1:raise ValueError('Bounded original guard required')
    if targets.shape!=(expected_rows,) or targets.dtype!=np.dtype('int8'):raise ValueError('Original zero int8 qualifying target shape required')
    for lo in range(0,len(targets),block_rows):
        guard()
        if np.any(targets[lo:lo+block_rows]!=0):raise ValueError('Qualifying target array is not zero')
    guard();return dict(rows=expected_rows,zero_targets=True)
