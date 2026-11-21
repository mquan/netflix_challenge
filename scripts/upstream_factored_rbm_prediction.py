"""Tiled prediction equations for the separate, not-yet-admitted GPU RBM.

Requires independently validated/deduplicated support coordinates. No labels,
fitting, RNG, model loading, query sorting, clipping, or production admission.
Unknown training-day IDs must be -1; known slots must belong to the query user.
"""


def predict(model, support, query, tile_rows=16384):
    import torch
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError('Positive integer row tile required')
    p, q, d, bv, bh, bu, but = [model[k] for k in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but')]
    i, u, rating, ci, cu = [support[k] for k in
        ('items', 'users', 'ratings', 'conditional_items', 'conditional_users')]
    qi, qu, qday, owner = [query[k] for k in ('items', 'users', 'days', 'day_owner')]
    if p.ndim != 3 or p.shape[1] != 5 or q.ndim != 2:
        raise ValueError('Five-category factored model required')
    ni, _, nf = p.shape; nh = q.shape[1]; nb = len(bu)
    if (nb < 1 or q.shape != (nf, nh) or d.shape != (ni, nh) or bv.shape != (ni, 5)
            or bh.shape != (nh,) or bu.shape != (nb, 5) or but.ndim != 2 or but.shape[1] != 5
            or owner.shape != (len(but),) or any(t.dtype != p.dtype or t.device != p.device for t in (q,d,bv,bh,bu,but))):
        raise ValueError('Exact same-device model and all-day ownership shapes required')
    if (i.ndim != 1 or u.shape != i.shape or rating.shape != i.shape or ci.ndim != 1
            or cu.shape != ci.shape or qi.ndim != 1 or qu.shape != qi.shape or qday.shape != qi.shape
            or any(t.dtype != torch.int64 or t.device != p.device for t in (i,u,rating,ci,cu,qi,qu,qday,owner))):
        raise ValueError('Same-device integer support and query coordinates required')
    for value, lower, upper in ((i,0,ni),(u,0,nb),(rating,0,5),(ci,0,ni),(cu,0,nb),
                                (qi,0,ni),(qu,0,nb),(qday,-1,len(but)),(owner,0,nb)):
        if bool(torch.any(value < lower)) or bool(torch.any(value >= upper)):
            raise ValueError('Coordinate out of range')
    known = qday >= 0
    if bool(torch.any(owner[qday[known]] != qu[known])):
        raise ValueError('Query day slot belongs to another user')
    with torch.no_grad():
        def tiles(n):
            for lo in range(0,n,tile_rows): yield slice(lo,min(n,lo+tile_rows))
        z = torch.zeros((nb,nf),dtype=p.dtype,device=p.device)
        conditional = torch.zeros((nb,nh),dtype=p.dtype,device=p.device)
        for t in tiles(len(i)): z.index_add_(0,u[t],p[i[t],rating[t]])
        for t in tiles(len(ci)): conditional.index_add_(0,cu[t],d[ci[t]])
        hidden = 2 * torch.sigmoid(bh + z @ q + conditional) - 1
        # Empty histories deliberately retain bh and conditional addresses,
        # matching get_hidden_expected rather than substituting a zero cache.
        zh = hidden @ q.T
        probabilities = torch.empty((len(qi),5),dtype=p.dtype,device=p.device)
        for t in tiles(len(qi)):
            items,users,days = qi[t],qu[t],qday[t]
            bias = bu[users].clone(); present = days >= 0
            bias[present] += but[days[present]]
            logits = bv[items] + bias + torch.einsum('lkf,lf->lk',p[items],zh[users])
            ex = torch.exp(logits-logits.max(dim=1,keepdim=True).values)
            probabilities[t] = ex/ex.sum(dim=1,keepdim=True)
        scores = (probabilities * torch.arange(1,6,dtype=p.dtype,device=p.device)).sum(dim=1)
        return dict(prediction=scores,probabilities=probabilities,hidden_expected=hidden)
