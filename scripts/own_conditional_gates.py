"""Vary the contribution of the frozen stage-01 experts with the user's context."""
import json
import numpy as np
import torch
from torch import nn

import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_full_context as full
import own_overnight_blend as blend

OUT = ne.ROOT / 'own-conditional-gates-20261126'
PARENT = blend.OUT / 'stage-01'


def inputs(keys, p, q, known):
    phase = 'probex' if known else 'qual'
    raw = np.load(full.OUT / (phase + '.npy'), mmap_mode='r')
    receipt = json.loads((full.OUT / (phase + '-sources.json')).read_text())
    assert ne.sha(full.OUT / (phase + '.npy')) == receipt['feature_sha256']
    directions = [blend.direction(k, p, q, known)[0].astype('f') for k in keys]
    # Preserve all expert directions and sample the broad panel's context across its groups.
    context = np.ascontiguousarray(raw[:, ::3])
    experts = np.column_stack(directions)
    return torch.from_numpy(context), torch.from_numpy(experts)


class Gate(nn.Module):
    def __init__(self, cx, ex):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cx + ex, 48), nn.SiLU(), nn.Dropout(.15),
                                 nn.Linear(48, ex + 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, e):
        weights = self.net(torch.cat([x, torch.clamp(e * 8, -5, 5)], 1)).tanh()
        return .08 * weights[:, 0] + (weights[:, 1:] * e).sum(1) * .5


def fit_predict(inputs_train, inputs_query, tr, val, errors):
    x, e = inputs_train
    qx, qe = inputs_query
    torch.manual_seed(20261126)
    sample = x[tr[::12]]
    low, high = torch.quantile(sample, torch.tensor([.02, .98]), dim=0)
    center = (high + low) * .5
    scale = torch.clamp(high - low, min=.005) * .5
    def norm(v):
        return ((v-center)/scale).clamp(-4,4)
    net = Gate(x.shape[1], e.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.04)
    result = np.zeros(len(val))
    for epoch in range(12):
        net.train()
        for idx in tr[torch.randperm(len(tr))].split(4096):
            corr = net(norm(x[idx]), e[idx])
            loss = (corr-errors[idx]).square().mean() + .025*corr.square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (3,7,11):
            net.eval()
            with torch.inference_mode():
                for pos in range(0,len(val),16384):
                    ii = val[pos:pos+16384]
                    result[pos:pos+len(ii)] += net(norm(qx[ii]),qe[ii]).numpy()/3
    return result


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    y,f,p,q = anchor.history_control()
    rec = json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32') == rec['qualifying_hashes']['selected']
    known = np.load(PARENT/'known-selected.npy')
    target = torch.from_numpy((y-known).astype('f'))
    x = inputs(rec['columns'],p,q,True)
    held = np.empty(len(y))
    for j in range(5):
        tr, val = [torch.from_numpy(np.flatnonzero(v)) for v in (f!=j,f==j)]
        held[val.numpy()] = fit_predict(x,x,tr,val,target)
        print('fold',j,ne.rmse(known[val]+held[val],y[val]),flush=True)
    np.save(OUT/'oof.npy',held)
    stats = {str(w):blend.metrics(ne.clipped(known+w*held),y,f) for w in (0,.25,.5,.75,1)}
    ok = [k for k in stats if stats[k]['pooled']<stats['0']['pooled']
          and sum(a<b for a,b in zip(stats[k]['folds'],stats['0']['folds']))>=4]
    key = min(ok,key=lambda k:stats[k]['pooled']) if ok else 'control'
    report = dict(selected=key,known=stats,baseline_sha256=rec['qualifying_hashes']['selected'],
                  qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        qx = inputs(rec['columns'],p,q,False)
        delta = fit_predict(x,qx,torch.arange(len(y)),torch.arange(len(q)),target)
        np.save(OUT/'qual-correction.npy',delta)
        prod = np.fromfile(PARENT/'selected.f32',dtype='<f4')
        out = OUT/'selected.f32'
        ne.clipped(prod+float(key)*delta).astype('<f4').tofile(out)
        report['qualifying_hashes']['selected'] = ne.sha(out)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        ne.audit(OUT,ne.LABELS)


if __name__ == '__main__':
    main()
