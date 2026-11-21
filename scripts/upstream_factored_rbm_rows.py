"""Array-only, label-isolated user batches for the separate GPU RBM study.

No production paths, model fitting or admission here. Caller supplies bound,
read-only raw arrays. Query ratings are never requested. Native training row
order is retained, including when dates inside a user's history are unordered.
"""
import numpy as np


def offsets(counts):
    return np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]


class UserRows:
    def __init__(self,training,query,n_items,block_rows=65536):
        if not isinstance(n_items,int) or n_items<1 or not isinstance(block_rows,int) or block_rows<1:
            raise ValueError('Positive item and validation block sizes required')
        self.tr={k:training[k] for k in ('user_idxs','item_idxs','ratings','dates','is_test','user_cnts')}
        self.qr={k:query[k] for k in ('user_idxs','item_idxs','dates','user_cnts')}
        self.n_users=len(self.tr['user_cnts']);self.n_items=n_items
        if self.n_users<1:raise ValueError('Nonempty user universe required')
        for source in (self.tr,self.qr):
            n=len(source['user_idxs'])
            if (source['user_cnts'].shape!=(self.n_users,)
                    or any(v.ndim!=1 or v.dtype.kind not in 'iu' for v in source.values())
                    or any(len(v)!=n for k,v in source.items() if k!='user_cnts')
                    or np.any(source['user_cnts']<0) or int(source['user_cnts'].sum(dtype=np.int64))!=n):
                raise ValueError('One-dimensional integer arrays and exact user counts required')
            for lo in range(0,n,block_rows):
                hi=min(lo+block_rows,n);u=source['user_idxs'][lo:hi];i=source['item_idxs'][lo:hi];date=source['dates'][lo:hi]
                if (np.any(u<0) or np.any(u>=self.n_users) or np.any(i<0) or np.any(i>=n_items)
                        or np.any(date<-32768) or np.any(date>32767)):
                    raise ValueError('Raw coordinate outside native bounds')
        self.training_offsets=offsets(self.tr['user_cnts'])
        counts=np.zeros((n_items,5),dtype=np.int64)
        for lo in range(0,len(self.tr['user_idxs']),block_rows):
            hi=min(lo+block_rows,len(self.tr['user_idxs']))
            expected=np.searchsorted(self.training_offsets[1:],np.arange(lo,hi),side='right')
            if not np.array_equal(expected,self.tr['user_idxs'][lo:hi]):
                raise ValueError('Native user-grouped row order/counts required')
            mask=self.tr['is_test'][lo:hi]
            if np.any((mask!=0)&(mask!=1)):raise ValueError('Binary native mask required')
            positive=mask==0;ratings=self.tr['ratings'][lo:hi][positive]
            if np.any(ratings<1) or np.any(ratings>5):raise ValueError('Positive-state raw ratings must be1..5')
            item=self.tr['item_idxs'][lo:hi][positive].astype(np.int64)
            counts+=np.bincount(item*5+ratings.astype(np.int64)-1,minlength=n_items*5).reshape(n_items,5)
        self.visible_counts=counts
        # The query is normally in item order. Only addresses participate in
        # this stable permutation; returned source row IDs restore query order.
        qu=self.qr['user_idxs'];actual=np.bincount(qu.astype(np.int64),minlength=self.n_users)
        if not np.array_equal(actual,self.qr['user_cnts']):raise ValueError('Query per-user counts disagree')
        self.query_order=np.argsort(qu,kind='stable');self.query_offsets=offsets(actual)
        days=[];positive_counts=np.zeros(self.n_users,dtype=np.int64)
        day_counts=np.zeros(self.n_users,dtype=np.int64)
        for user in range(self.n_users):
            lo,hi=self.training_offsets[user:user+2];positive=self.tr['is_test'][lo:hi]==0
            day=np.unique(self.tr['dates'][lo:hi][positive]).astype(np.int16)
            days.append(day);positive_counts[user]=np.count_nonzero(positive);day_counts[user]=len(day)
        self.day_values=np.concatenate(days);self.day_offsets=offsets(day_counts)
        self.positive_counts=positive_counts

    @property
    def nonempty_users(self):
        return np.flatnonzero(self.positive_counts)

    def batch(self,user_ids,training=False):
        ids=np.asarray(user_ids)
        if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0)
                or np.any(ids>=self.n_users) or len(np.unique(ids))!=len(ids)):
            raise ValueError('Distinct nonempty batch of valid user IDs required')
        if training and np.any(self.positive_counts[ids]==0):
            raise ValueError('Training batches count only nonempty users')
        positive_rows=[];item=[];rating=[];users=[];day=[];ci=[];cu=[]
        day_owner=[];global_days=[];query_rows=[];qi=[];qu=[];qd=[]
        day_base=0
        for local,user in enumerate(ids):
            lo,hi=self.training_offsets[user:user+2]
            positive=self.tr['is_test'][lo:hi]==0
            source_rows=np.arange(lo,hi,dtype=np.int64)[positive];positive_rows.append(source_rows)
            item.append(self.tr['item_idxs'][source_rows]);rating.append(self.tr['ratings'][source_rows].astype(np.int64)-1)
            users.append(np.full(len(source_rows),local,dtype=np.int64))
            dl,dh=self.day_offsets[user:user+2];known_days=self.day_values[dl:dh]
            di=np.searchsorted(known_days,self.tr['dates'][source_rows]);day.append(di+day_base)
            global_days.append(np.arange(dl,dh,dtype=np.int64));day_owner.append(np.full(dh-dl,local,dtype=np.int64))
            qlo,qhi=self.query_offsets[user:user+2];qr=self.query_order[qlo:qhi];query_rows.append(qr)
            query_items=self.qr['item_idxs'][qr]
            qi.append(query_items);qu.append(np.full(len(qr),local,dtype=np.int64))
            qdates=self.qr['dates'][qr];query_days=np.full(len(qr),-1,dtype=np.int64)
            if len(known_days):
                position=np.searchsorted(known_days,qdates);inside=position<len(known_days)
                found=np.zeros(len(qr),dtype=bool);found[inside]=known_days[position[inside]]==qdates[inside]
                query_days[found]=position[found]+day_base
            qd.append(query_days)
            # Source build_r_items includes masked training addresses as well
            # as every query address, then sorts/deduplicates per user.
            conditional=np.union1d(self.tr['item_idxs'][lo:hi],query_items)
            ci.append(conditional);cu.append(np.full(len(conditional),local,dtype=np.int64))
            day_base+=len(known_days)
        cat=lambda rows:np.concatenate(rows).astype(np.int64,copy=False)
        owners=cat(day_owner)
        support=dict(items=cat(item),users=cat(users),ratings=cat(rating),days=cat(day),
                     conditional_items=cat(ci),conditional_users=cat(cu),day_owner=owners)
        query=dict(items=cat(qi),users=cat(qu),days=cat(qd),day_owner=owners)
        return dict(support=support,query=query,global_users=ids.astype(np.int64),global_days=cat(global_days),
                    training_rows=cat(positive_rows),query_rows=cat(query_rows))
