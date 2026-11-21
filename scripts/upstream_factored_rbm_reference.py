"""Small float64 mathematical oracle for a future, separately named GPU RBM.

This is not a production fitter or a bitwise reproduction of Rust's RNG/f32
reductions. Fixed external uniforms make CD trajectories independently testable.
Implements the softmax/bipolar/factored, temperature-one, LR_BIF=0 rx.rs arm.
"""
import numpy as np


def expected(acts):
    with np.errstate(over='ignore'):
        return 2 / (1 + np.exp(-acts)) - 1


def batch_gradients(model, batch, hidden_uniforms, visible_uniforms, tile_rows=16384):
    """Return one nonempty-user batch's unnormalised CD gradients, without mutation.

    bu/but contain the selected users and ALL their training-day slots. Batch
    user IDs and day IDs are local; day_owner verifies their correspondence.
    Conditional item pairs must already be the deduplicated train/query union.
    """
    p, q, d, bv, bh, bu, but = [np.asarray(model[k], dtype=np.float64)
                               for k in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but')]
    i, u, rating, day, ci, cu, owner = [np.asarray(batch[k]) for k in
        ('items', 'users', 'ratings', 'days', 'conditional_items', 'conditional_users', 'day_owner')]
    hu, vu = np.asarray(hidden_uniforms), np.asarray(visible_uniforms)
    if p.ndim != 3 or p.shape[1] != 5 or q.ndim != 2:
        raise ValueError('Five-category factored model required')
    ni, _, nf = p.shape; nh = q.shape[1]; nb = len(bu); nt = len(but)
    if (q.shape != (nf, nh) or d.shape != (ni, nh) or bv.shape != (ni, 5)
            or bh.shape != (nh,) or bu.shape != (nb, 5) or but.shape != (nt, 5)
            or owner.shape != (nt,) or not isinstance(tile_rows, int) or tile_rows < 1):
        raise ValueError('Exact model and all-day shapes required')
    if (i.ndim != 1 or not len(i) or any(x.shape != i.shape for x in (u, rating, day))
            or ci.ndim != 1 or cu.shape != ci.shape
            or any(x.dtype.kind not in 'iu' for x in (i, u, rating, day, ci, cu, owner))):
        raise ValueError('Integer rating/conditional/day coordinates required')
    for x, upper in ((i, ni), (u, nb), (rating, 5), (day, nt), (ci, ni), (cu, nb), (owner, nb)):
        if np.any(x < 0) or np.any(x >= upper):
            raise ValueError('Coordinate out of range')
    if (not np.array_equal(np.unique(u), np.arange(nb)) or not np.array_equal(owner[day], u)
            or len(set(zip(cu.tolist(), ci.tolist()))) != len(ci)):
        raise ValueError('Nonempty users, correct day ownership and deduplicated conditionals required')
    if (hu.ndim != 3 or hu.shape[1:] != (nb, nh) or hu.shape[0] < 1
            or vu.shape != (hu.shape[0], len(i))
            or any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x >= 1) for x in (hu, vu))):
        raise ValueError('Complete fixed CD-step uniforms in [0,1) required')
    if any(not np.isfinite(x).all() for x in (p, q, d, bv, bh, bu, but)):
        raise ValueError('Finite original parameters required')
    conditional = np.zeros((nb, nh)); np.add.at(conditional, cu, d[ci])

    def factors(state):
        z = np.zeros((nb, nf)); np.add.at(z, u, p[i, state])
        return z

    zp = factors(rating)
    acts = bh + zp @ q + conditional
    ep = expected(acts)
    state = rating.copy()
    for step in range(len(hu)):
        # Native hidden sampling uses sigmoid(a), not (expected(a)+1)/2
        # computed after rounding, and bipolar values are {-1,+1}.
        with np.errstate(over='ignore'):
            hs = np.where(hu[step] < 1 / (1 + np.exp(-acts)), 1., -1.)
        zh = hs @ q.T
        for lo in range(0, len(i), tile_rows):
            hi = min(lo + tile_rows, len(i)); ii, uu, dd = i[lo:hi], u[lo:hi], day[lo:hi]
            logits = bv[ii] + bu[uu] + but[dd] + np.einsum('lkf,lf->lk', p[ii], zh[uu])
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            # rx.rs samples against unnormalised exponentials with <=,
            # including exact categorical boundaries; final fallback is K-1.
            draw = vu[step, lo:hi] * ex.sum(axis=1)
            state[lo:hi] = np.minimum((draw[:, None] > ex.cumsum(axis=1)).sum(axis=1), 4)
        acts = bh + factors(state) @ q + conditional
    zn = factors(state); en = expected(acts); delta = ep - en
    gp = np.zeros_like(p); np.add.at(gp, (i, rating), (ep @ q.T)[u])
    np.add.at(gp, (i, state), -(en @ q.T)[u])
    gd = np.zeros_like(d); np.add.at(gd, ci, delta[cu])
    gbv = np.zeros_like(bv); np.add.at(gbv, (i, rating), 1); np.add.at(gbv, (i, state), -1)
    gbu = np.zeros_like(bu); np.add.at(gbu, (u, rating), 1); np.add.at(gbu, (u, state), -1)
    gbut = np.zeros_like(but); np.add.at(gbut, (day, rating), 1); np.add.at(gbut, (day, state), -1)
    return dict(p=gp, q=zp.T @ ep - zn.T @ en, d=gd, bv=gbv, bh=delta.sum(axis=0),
                bu=gbu, but=gbut, visible_state=state, positive_expected=ep, negative_expected=en,
                batch_users=nb)


def update(model, momentum, gradients, lr=.02, mom=.9, wd=.001, lr_bu=.001, lr_but=.0005):
    """Apply native update scaling, including untouched shared parameters.

    All gradients must have been computed against the same pre-update model.
    The per-user bias slots are disjoint within a batch, so their per-user
    updates commute with each other and with the shared batch update.
    """
    count = gradients['batch_users']
    if not isinstance(count, int) or count < 1:
        raise ValueError('Actual nonempty-user batch count required')
    new_model, new_momentum = {}, {}
    for key in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but'):
        value, velocity, grad = [np.asarray(x[key], dtype=np.float64) for x in (model, momentum, gradients)]
        if value.shape != velocity.shape or value.shape != grad.shape:
            raise ValueError('Exact momentum/gradient shape required')
        rate = lr_bu if key == 'bu' else lr_but if key == 'but' else lr / count
        decay = .01 if key in ('bu', 'but') else wd if key in ('p', 'q', 'd') else 0
        new_momentum[key] = mom * velocity + rate * (grad - decay * value)
        new_model[key] = value + new_momentum[key]
    return new_model, new_momentum
