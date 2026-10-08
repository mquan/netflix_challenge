#!/usr/bin/env python3
"""ENNBlend (BigChaos section 6.6) over a round4_boost.py bank, averaged with LightGBM and NN2.

Reads the inputs of an existing round4_boost.py report (--like) through
round8_nn_blend.bank(), so the prediction columns, tree features, contextual
ridge, outer user folds and user-disjoint inner split are exactly those of
round4_boost.py / round8_nn_blend.py (hashes checked, ridge control checked
against the report to every digit).

Per outer user fold:
  * N tiny nets (--nets), each seeing --subset randomly chosen prediction
    columns (fixed by --subset-seed, same subsets in every fold) plus
    log user support and log item support (meta0 / meta1 of the tree features,
    robust-standardized on outer-training rows as round8 does), one hidden
    layer of --hidden tanh units. All N nets are trained at once as one
    batched model (weights [N, in, hidden]); the loss is the sum of the nets'
    own mean squared errors, so every net gets exactly the gradient it would
    get alone and AdamW is per-element, i.e. this is N independent trainings.
  * Target: the contextual-ridge residual, as NN2 (round8_nn_blend.py); --target rating
    trains the nets on y-3.6 instead (BigChaos style). Either way the stacker
    predicts y - ridge from [1, net outputs, ridge-3.6].
  * Inner stage: nets fit on inner-fit users (target y - inner ridge), each net's
    epoch chosen on inner-valid users (per-net patience).
  * Stacker: per user-support bin (--bins quantile bins of log user support,
    edges from outer-training rows), a ridge (with intercept and the inner ridge
    prediction as one more column) maps the N inner net outputs on inner-valid
    rows to y - inner ridge. The ridge penalty is
    chosen from --stack-alphas by a 2-way user split of the inner-valid rows.
    Inner-valid outputs come from nets that never trained on those rows, so
    they are out-of-sample for the stacker (only the per-net epoch count was
    selected on them). No outer-held label enters any fit.
  * Refit: the nets are retrained on all outer-training users (target y - outer
    ridge, as round8) for their chosen epoch counts, and the held fold is
    predicted as outer ridge + stacker(refit net outputs).
Scored alone, and in predeclared fixed averages: 1/3 ENN + 1/3 LightGBM
(the --like report's refit column) + 1/3 NN2 (--nn2), and ENN/LightGBM 50/50.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from gbdt_blend import load, blend_fold
from moonshot_nn_blend import design, ridge_fit
from round8_nn_blend import bank, standardize, rmse

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


class ENN(torch.nn.Module):
    """N independent 1-hidden-layer nets with batched weights.

    w1 is dense [N, F, hidden] over all F feature columns but multiplied by a fixed
    0/1 mask, so net n only sees its own columns (masked weights get zero gradient
    and, with no weight decay on them, stay zero). One [B, F] x [F, N*hidden] matmul
    is much faster on CPU than gathering a [B, N, k] input tensor."""

    def __init__(self, index, features, hidden, seed):
        super().__init__()
        nets, width = index.shape
        g = torch.Generator().manual_seed(seed)
        bound = 1/np.sqrt(width)  # torch.nn.Linear default init for a width-input layer
        mask = torch.zeros(nets, features, 1)
        mask[torch.arange(nets)[:, None], index] = 1
        self.register_buffer('mask', mask)
        self.w1 = torch.nn.Parameter((torch.rand(nets, features, hidden, generator=g)*2-1)*bound*mask)
        self.b1 = torch.nn.Parameter((torch.rand(nets, hidden, generator=g)*2-1)*bound)
        self.w2 = torch.nn.Parameter(torch.zeros(nets, hidden))  # zero output layer, as round8
        self.b2 = torch.nn.Parameter(torch.zeros(nets))

    def forward(self, x):  # x: [B, F] -> [B, N]
        nets, features, hidden = self.w1.shape
        w = (self.w1*self.mask).permute(1, 0, 2).reshape(features, nets*hidden)
        h = torch.tanh((x@w).view(len(x), nets, hidden)+self.b1)
        return (h*self.w2).sum(-1)+self.b2


def predict(model, tx, rows):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(tx[b]) for b in rows.split(16384)]).numpy()


def train(tx, index, rows, r, a, seed, epochs, valid=None):
    """Train all nets on rows (target r). With valid=(rows, target): per-net best epoch
    (patience a.patience) and best-state weights. Without valid: epochs is a per-net array;
    each net's weights are snapshotted at the end of its own epoch count (0 = init)."""
    torch.manual_seed(seed)
    model = ENN(index, tx.shape[1], a.hidden, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = torch.from_numpy(r.astype(np.float32))
    n = len(index)
    snap = {k: v.detach().clone() for k, v in model.state_dict().items() if k != 'mask'}
    if valid is None:
        target = torch.as_tensor(np.asarray(epochs))
        last = int(target.max()) if n else 0
    else:
        vr = valid[1].astype(np.float64)
        best = np.full(n, np.mean(vr**2))  # zero-output init is epoch 0
        best_epoch = np.zeros(n, dtype=np.int64)
        last = epochs
    history = [float(np.sqrt(best.mean()))] if valid is not None else []
    for epoch in range(1, last+1):
        model.train()
        for batch in torch.randperm(len(rows)).split(a.batch):
            opt.zero_grad(set_to_none=True)
            out = model(tx[rows[batch]])
            (out-tr[batch, None]).square().mean(0).sum().backward(); opt.step()
        if valid is None:
            done = target == epoch
            if done.any():
                for k in snap: snap[k][done] = model.state_dict()[k][done]
            continue
        score = np.mean((predict(model, tx, valid[0])-vr[:, None])**2, axis=0)
        better = score < best
        best[better], best_epoch[better] = score[better], epoch
        mask = torch.from_numpy(better)
        if better.any():
            for k in snap: snap[k][mask] = model.state_dict()[k][mask]
        history.append(float(np.sqrt(score.mean())))
        if a.verbose: print(json.dumps(dict(epoch=epoch, valid_rmse=history[-1], improved=int(better.sum()))), flush=True)
        if np.all(epoch-best_epoch >= a.patience): break
    model.load_state_dict(snap, strict=False)
    return model, (None if valid is None else best_epoch), history


def ridge_solve(x, t, alpha):
    """Ridge with unpenalized intercept column 0; penalty scaled by the mean feature energy."""
    a = x.T@x
    scale = np.mean(np.diag(a)[1:]) if x.shape[1] > 1 else 1.0
    pen = np.eye(x.shape[1])*alpha*scale; pen[0, 0] = 0
    return np.linalg.solve(a+pen, x.T@t)


def stack_fit(p, base, t, bins, nbins, split, alphas):
    """Per-bin ridge on [1, net outputs p, ridge base - 3.6] -> t (= y - ridge base);
    alpha chosen by a 2-way user split (pooled SSE)."""
    x = np.column_stack([np.ones(len(p)), p.astype(np.float64), base-3.6])
    sse = []
    for alpha in alphas:
        total = 0.0
        for half in (0, 1):
            for b in range(nbins):
                f, e = (bins == b) & (split != half), (bins == b) & (split == half)
                total += float(np.sum((x[e]@ridge_solve(x[f], t[f], alpha)-t[e])**2))
        sse.append(total)
    alpha = alphas[int(np.argmin(sse))]
    weights = np.stack([ridge_solve(x[bins == b], t[bins == b], alpha) for b in range(nbins)])
    return weights, alpha, [float(np.sqrt(s/len(t))) for s in sse]


def stack_apply(weights, p, base, bins):
    x = np.column_stack([np.ones(len(p)), p.astype(np.float64), base-3.6])
    return np.einsum('rf,rf->r', x, weights[bins])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--like', required=True, help='round4_boost.py report (.json) whose inputs and refit to use')
    ap.add_argument('--nn2', help='round8_nn_blend.py output prefix (e.g. artifacts/r8/R8-A2f-NN2) on the same --like')
    ap.add_argument('--output', required=True)
    ap.add_argument('--nets', type=int, default=400)
    ap.add_argument('--subset', type=int, default=4, help='prediction columns per net')
    ap.add_argument('--subset-seed', type=int, default=2026)
    ap.add_argument('--hidden', type=int, default=8)
    ap.add_argument('--target', choices=['rating', 'residual'], default='residual',
                    help='net target: y-3.6, or the contextual-ridge residual as NN2 (stacker target is always the residual)')
    ap.add_argument('--bins', type=int, default=4, help='user-support quantile bins for the stacking ridge')
    ap.add_argument('--stack-alphas', type=float, nargs='+', default=ALPHAS)
    ap.add_argument('--lr', type=float, default=.003)
    ap.add_argument('--weight-decay', type=float, default=.0)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--verbose', action='store_true', help='log every inner epoch')
    ap.add_argument('--max-folds', type=int, default=5, help='debug: run only the first K outer folds')
    a = ap.parse_args()
    if pathlib.Path(a.output+'.json').exists(): ap.error('output already exists; use a new experiment name')
    torch.set_num_threads(a.threads)
    like = json.loads(pathlib.Path(a.like).read_text())
    d = load(a.data); y = d['probe']['rating'].astype(np.float64); users = d['probe']['user']
    preds, tree, context, hashes = bank(like['config'], d)
    if set(hashes) != set(like['inputs']) or any(like['inputs'][k] != v for k, v in hashes.items()):
        raise ValueError('inputs differ from --like report')
    boost = np.fromfile(a.like[:-len('.json')]+'.refit.f32', dtype='<f4').astype(np.float64)
    nn2 = None
    if a.nn2:
        nn2_report = json.loads(pathlib.Path(a.nn2+'.json').read_text())
        if nn2_report['inputs'] != like['inputs']: raise ValueError('--nn2 inputs differ from --like report')
        nn2 = np.fromfile(a.nn2+'.nn.f32', dtype='<f4').astype(np.float64)
        if len(nn2) != len(y): raise ValueError('--nn2 length')
        # Row order check: recomputed per-fold RMSE must reproduce the NN2 report.
        f = blend_fold(users)
        got = [rmse(nn2[f == k], y[f == k]) for k in range(5)]
        if not np.allclose(got, nn2_report['nn_fold_rmse'], rtol=0, atol=1e-7):
            raise ValueError(f'--nn2 fold RMSE {got} != report {nn2_report["nn_fold_rmse"]}')
    # Log supports: meta0/meta1 of gbdt_blend.meta_features, after the support columns in bank().
    support_at = len(json.loads((pathlib.Path(like['config']['bank_root'])/'experiments/BOOST-support-inputs.json')
                                .read_text()))+len(like['config']['extra_support'])
    log_user, log_item = tree[:, support_at], tree[:, support_at+1]
    user_n = np.diff(d['train_offsets'].astype(np.int64))[users.astype(np.int64)]
    item_n = np.bincount(d['train_items'].astype(np.int64)-1, minlength=d['items'])[d['probe']['item'].astype(np.int64)-1]
    if not (np.allclose(log_user, np.log1p(user_n), atol=1e-5) and np.allclose(log_item, np.log1p(item_n), atol=1e-5)):
        raise ValueError('log support columns not where expected in the tree features')
    rx = design(preds, context)[0]
    features = np.column_stack([preds-3.6, log_user, log_item])
    p = preds.shape[1]
    rng = np.random.default_rng(a.subset_seed)
    cols = np.stack([rng.choice(p, a.subset, replace=False) for _ in range(a.nets)])
    index = torch.from_numpy(np.column_stack([cols, np.full((a.nets, 1), p), np.full((a.nets, 1), p+1)]))
    print(json.dumps(dict(config=vars(a), prediction_columns=p, features=features.shape[1],
                          support_columns=[support_at, support_at+1])), flush=True)
    outer = blend_fold(users)
    inner = blend_fold(users.astype(np.uint64)*np.uint64(7)+np.uint64(1)) == 0
    split = blend_fold(users.astype(np.uint64)*np.uint64(13)+np.uint64(5)) % 2  # stacker alpha choice only
    out, diag, records = np.full(len(y), np.nan), np.full(len(y), np.nan), []
    for fold in range(a.max_folds):
        start = time.monotonic()
        tr, held = outer != fold, outer == fold
        fit, valid = tr & ~inner, tr & inner
        tx = torch.from_numpy(standardize(features, tr))
        inner_base = 3.6+rx@ridge_fit(rx[fit], y[fit])
        base = 3.6+rx@ridge_fit(rx[tr], y[tr])
        ridge_rmse = rmse(base[held], y[held])
        if ridge_rmse != like['ridge_fold_rmse'][fold]:
            raise ValueError(f'ridge control {ridge_rmse!r} != report {like["ridge_fold_rmse"][fold]!r}')
        edges = np.quantile(log_user[tr], np.arange(1, a.bins)/a.bins)
        bins = np.searchsorted(edges, log_user, side='right')
        fit_rows, valid_rows = (torch.from_numpy(np.flatnonzero(m)) for m in (fit, valid))
        tr_rows, held_rows = (torch.from_numpy(np.flatnonzero(m)) for m in (tr, held))
        seed = a.seed+fold
        inner_t = y-inner_base if a.target == 'residual' else y-3.6
        outer_t = y-base if a.target == 'residual' else y-3.6
        model, chosen, history = train(tx, index, fit_rows, inner_t[fit], a, seed, a.epochs,
                                       valid=(valid_rows, inner_t[valid]))
        t_inner = time.monotonic()-start
        pv = predict(model, tx, valid_rows)
        weights, alpha, alpha_rmse = stack_fit(pv, inner_base[valid], y[valid]-inner_base[valid], bins[valid],
                                               a.bins, split[valid], a.stack_alphas)
        ph_inner = predict(model, tx, held_rows)
        model, _, _ = train(tx, index, tr_rows, outer_t[tr], a, seed, chosen)
        ph = predict(model, tx, held_rows)
        out[held] = base[held]+stack_apply(weights, ph, base[held], bins[held])
        diag[held] = inner_base[held]+stack_apply(weights, ph_inner, inner_base[held], bins[held])
        rec = dict(fold=fold, seconds=time.monotonic()-start, inner_seconds=t_inner,
                   epochs_run=len(history)-1, chosen_epochs=dict(
                       min=int(chosen.min()), median=float(np.median(chosen)), max=int(chosen.max()),
                       zero=int((chosen == 0).sum())),
                   inner_valid_rmse_by_epoch=[round(h, 6) for h in history],
                   stack_alpha=alpha, stack_alpha_rmse=alpha_rmse,
                   inner_ridge_valid_rmse=float(np.sqrt(np.mean((y[valid]-inner_base[valid])**2))), bin_edges=np.expm1(edges).tolist(),
                   ridge_rmse=ridge_rmse, enn_rmse=rmse(out[held], y[held]),
                   enn_inner_nets_rmse=rmse(diag[held], y[held]),
                   boost_rmse=rmse(boost[held], y[held]),
                   enn_boost_rmse=rmse((out[held]+boost[held])/2, y[held]))
        if nn2 is not None:
            rec.update(nn2_rmse=rmse(nn2[held], y[held]),
                       nn2_boost_rmse=rmse((nn2[held]+boost[held])/2, y[held]),
                       avg3_rmse=rmse((out[held]+boost[held]+nn2[held])/3, y[held]))
        records.append(rec); print(json.dumps(rec), flush=True)
    done = np.isin(outer, np.arange(a.max_folds))
    folds = range(a.max_folds)
    report = dict(config=vars(a), inputs=hashes, subsets=cols.tolist(), folds=records,
                  complete=a.max_folds == 5, torch_version=torch.__version__)
    columns = [('enn', out), ('boost', boost), ('enn_boost', (out+boost)/2)]
    if nn2 is not None:
        avg3 = (out+boost+nn2)/3
        columns += [('nn2', nn2), ('nn2_boost', (nn2+boost)/2), ('avg3', avg3)]
    for name, value in columns:
        report[name+'_rmse'] = rmse(value[done], y[done])
        report[name+'_fold_rmse'] = [rmse(value[outer == f], y[outer == f]) for f in folds]
    out.astype('<f4').tofile(a.output+'.enn.f32')
    if nn2 is not None: avg3.astype('<f4').tofile(a.output+'.avg3.f32')
    pathlib.Path(a.output+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in report if k.endswith('rmse')}), flush=True)


if __name__ == '__main__': main()
