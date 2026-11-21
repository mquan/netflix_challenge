from upstream_factored_rbm_metal_reduction import add_rows_
"""Tiled Torch kernel for the separately named factored bipolar RBM study.

No data loading, training runner, RNG, admission policy, or score selection here.
Caller supplies validated, same-device arrays and complete fixed CD uniforms.
All rating/conditional rows are visited; tiles do not change user batch size.
Torch is imported lazily so declarations/reference checks need no GPU runtime.
"""


def batch_gradients(model, batch, hidden_uniforms, visible_uniforms, tile_rows=16384):
    import torch
    if not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError('Positive integer row tile required')
    p, q, d, bv, bh, bu, but = [model[k] for k in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but')]
    i, u, rating, day, ci, cu = [batch[k] for k in
        ('items', 'users', 'ratings', 'days', 'conditional_items', 'conditional_users')]
    nb, nf, nh = len(bu), q.shape[0], q.shape[1]
    if (nb < 1 or p.ndim != 3 or p.shape[1:] != (5, nf)
            or d.shape != (p.shape[0], nh) or bv.shape != (p.shape[0], 5)
            or bh.shape != (nh,) or bu.shape != (nb, 5) or but.ndim != 2 or but.shape[1] != 5
            or hidden_uniforms.ndim != 3 or hidden_uniforms.shape[1:] != (nb, nh)
            or len(hidden_uniforms) < 1 or visible_uniforms.shape != (len(hidden_uniforms), len(i))):
        raise ValueError('Complete factored model and fixed CD uniforms required')
    with torch.no_grad():
        def zeros(shape): return torch.zeros(shape, dtype=p.dtype, device=p.device)
        def tiles(n):
            for lo in range(0, n, tile_rows): yield slice(lo, min(n, lo + tile_rows))
        conditional = zeros((nb, nh))
        for t in tiles(len(ci)): add_rows_(conditional, cu[t], d[ci[t]])

        def factors(state):
            z = zeros((nb, nf))
            for t in tiles(len(i)): add_rows_(z, u[t], p[i[t], state[t]])
            return z

        zp = factors(rating); acts = bh + zp @ q + conditional
        ep = 2 * torch.sigmoid(acts) - 1
        state = rating.clone()
        for step in range(len(hidden_uniforms)):
            hs = torch.where(hidden_uniforms[step] < torch.sigmoid(acts), 1., -1.).to(p.dtype)
            zh = hs @ q.T
            for t in tiles(len(i)):
                logits = bv[i[t]] + bu[u[t]] + but[day[t]] + torch.einsum('lkf,lf->lk', p[i[t]], zh[u[t]])
                ex = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
                draw = visible_uniforms[step, t] * ex.sum(dim=1)
                state[t] = (draw[:, None] > ex.cumsum(dim=1)).sum(dim=1).clamp(max=4)
            acts = bh + factors(state) @ q + conditional
        zn = factors(state); en = 2 * torch.sigmoid(acts) - 1; delta = ep - en
        gp = torch.zeros_like(p); flat = gp.reshape(-1, nf)
        qep, qen = ep @ q.T, en @ q.T
        gd = torch.zeros_like(d)
        gbv, gbu, gbut = torch.zeros_like(bv), torch.zeros_like(bu), torch.zeros_like(but)
        for t in tiles(len(i)):
            add_rows_(flat, i[t] * 5 + rating[t], qep[u[t]])
            add_rows_(flat, i[t] * 5 + state[t], -qen[u[t]])
            ones = torch.ones_like(i[t], dtype=p.dtype)
            for dest, row in ((gbv, i[t]), (gbu, u[t]), (gbut, day[t])):
                add_rows_(dest.view(-1), row * 5 + rating[t], ones)
                add_rows_(dest.view(-1), row * 5 + state[t], -ones)
        for t in tiles(len(ci)): add_rows_(gd, ci[t], delta[cu[t]])
        return dict(p=gp, q=zp.T @ ep - zn.T @ en, d=gd, bv=gbv, bh=delta.sum(dim=0),
                    bu=gbu, but=gbut, visible_state=state, positive_expected=ep, negative_expected=en,
                    batch_users=nb)


def update(model, momentum, gradients, lr=.02, mom=.9, wd=.001, lr_bu=.001, lr_but=.0005):
    import torch
    count = gradients['batch_users']
    if not isinstance(count, int) or count < 1:
        raise ValueError('Actual nonempty-user count required')
    values, velocities = {}, {}
    with torch.no_grad():
        for key in ('p', 'q', 'd', 'bv', 'bh', 'bu', 'but'):
            value, velocity, grad = [x[key] for x in (model, momentum, gradients)]
            if value.shape != velocity.shape or value.shape != grad.shape:
                raise ValueError('Exact momentum/gradient shape required')
            rate = lr_bu if key == 'bu' else lr_but if key == 'but' else lr / count
            decay = .01 if key in ('bu', 'but') else wd if key in ('p', 'q', 'd') else 0
            velocities[key] = mom * velocity + rate * (grad - decay * value)
            values[key] = value + velocities[key]
    return values, velocities
