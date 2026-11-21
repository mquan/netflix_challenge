"""Array-only math for the declared RBM400 representation experiment.

No files, models, labels, training, production admission or GPU work here.
Callers must separately verify complete original models and user coverage.
"""
import numpy as np

FACTORS=50
ALIGNMENT_RIDGE=1e-6
MAX_CONDITION=1e6


def matrix(value,columns):
    result=np.asarray(value)
    if result.ndim!=2 or result.shape[1]!=columns or not np.isfinite(result).all():
        raise ValueError('Finite complete matrix with exact factor width required')
    return result.astype(np.float64)


def item_contrast(p):
    p=np.asarray(p)
    if p.ndim!=3 or p.shape[1:]!=(5,FACTORS) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite complete five-category50factor item tensor required')
    weights=np.arange(-2,3,dtype=np.float64)/np.sqrt(np.float64(10))
    result=np.zeros((len(p),FACTORS),dtype=np.float64)
    # Fixed category order, independent of an einsum contraction planner.
    for k in range(5):result+=p[:,k,:].astype(np.float64)*weights[k]
    with np.errstate(over='ignore',invalid='ignore'):stored=result.astype('<f4')
    if not np.isfinite(stored).all():raise ValueError('Finite float32 item contrast cache required')
    return stored


def alignment(known_items,qualifying_items):
    known=matrix(known_items,FACTORS);qual=matrix(qualifying_items,FACTORS)
    if known.shape!=qual.shape or len(known)<FACTORS:
        raise ValueError('All shared canonical movie coordinates required')
    transform=np.linalg.solve(qual.T@qual+ALIGNMENT_RIDGE*np.eye(FACTORS),qual.T@known)
    singular=np.linalg.svd(transform,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    if (not np.isfinite(transform).all() or not np.isfinite(condition)
            or singular[-1]<=np.finfo(np.float64).eps*FACTORS*singular[0]
            or condition>MAX_CONDITION):
        raise ValueError('Full-rank well-conditioned fixed label-free alignment required')
    inverse_transpose=np.linalg.inv(transform).T
    if not np.isfinite(inverse_transpose).all():raise ValueError('Finite contragredient user map required')
    return dict(item_map=transform,user_map=inverse_transpose,condition=condition)


def align_qualifying(users,items,maps):
    users=matrix(users,FACTORS);items=matrix(items,FACTORS)
    a=matrix(maps['item_map'],FACTORS);b=matrix(maps['user_map'],FACTORS)
    if a.shape!=(FACTORS,FACTORS) or b.shape!=a.shape:
        raise ValueError('Exact full factor maps required')
    if not np.allclose(a@b.T,np.eye(FACTORS),rtol=0,atol=1e-9):
        raise ValueError('User and item maps must preserve the latent dot product')
    aligned_users=users@b;aligned_items=items@a
    if not np.isfinite(aligned_users).all() or not np.isfinite(aligned_items).all():
        raise ValueError('Finite aligned factor cache required')
    with np.errstate(over='ignore',invalid='ignore'):
        stored_users=aligned_users.astype('<f4');stored_items=aligned_items.astype('<f4')
    if not np.isfinite(stored_users).all() or not np.isfinite(stored_items).all():
        raise ValueError('Finite float32 aligned caches required')
    return stored_users,stored_items


class Context:
    """Virtual U,I,U*I query columns; uncovered users are never zero-filled."""
    def __init__(self,users,items,covered):
        self.users=np.asarray(users);self.items=np.asarray(items);self.covered=np.asarray(covered)
        if (self.users.ndim!=2 or self.users.shape[1]!=FACTORS or self.items.ndim!=2
                or self.items.shape[1]!=FACTORS or self.covered.dtype!=np.bool_
                or self.covered.shape!=(len(self.users),) or not len(self.items)
                or not np.isfinite(self.users[self.covered]).all() or not np.isfinite(self.items).all()):
            raise ValueError('Exact finite factor caches and explicit user coverage required')

    def values(self,users,items):
        users=np.asarray(users);items=np.asarray(items)
        if (users.ndim!=1 or items.shape!=users.shape or users.dtype.kind not in 'iu'
                or items.dtype.kind not in 'iu' or np.any(users<0) or np.any(users>=len(self.users))
                or np.any(items<0) or np.any(items>=len(self.items))):
            raise ValueError('Canonical query integer addresses required')
        if not self.covered[users].all():raise ValueError('Every query user must have its original posterior cache')
        u=self.users[users].astype('<f4');i=self.items[items].astype('<f4')
        output=np.concatenate((u,i,u*i),axis=1).astype('<f4')
        if not np.isfinite(output).all():raise ValueError('Finite full150coordinate rows required')
        return output
