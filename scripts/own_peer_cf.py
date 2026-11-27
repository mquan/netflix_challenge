"""Evaluate a single frozen trainx CF96 encoder under matching observable-peer interventions in both roles."""
import json
import numpy as np
import own_cf_slate as slate
import own_full_context as full
import own_neighbor_experiment as ne
import own_peer_replay as peer
import own_retained_history as ret
OUT=ne.ROOT/'own-peer-cf-20261126'


def main():
    OUT.mkdir(exist_ok=True);data=peer.Cohort();model=slate.Data('trainx','probex')
    mapping=ret.QUAL/'dataset/dense-to-raw-user.npy';sig=json.loads((mapping.parent/'manifest.json').read_text())
    assert ne.sha(mapping)==sig['outputs'][mapping.name]
    raw=np.load(mapping);canon=ne.history('trainx','uidx_to_uid');ids=np.searchsorted(canon,raw)
    np.testing.assert_array_equal(canon[ids],raw)
    stars=np.arange(1,6,dtype='f')
    result=np.lib.format.open_memmap(OUT/'probabilities.npy',mode='w+',dtype='f2',shape=(len(data.rows),8,5))
    native=np.empty(data.n,'f');users=np.flatnonzero(data.grid[:,0]>=0)
    for j in range(0,len(users),256):
        us=users[j:j+256];u=ids[us];g=data.grid[us];valid=g>=0;rows=g.clip(min=0)
        a=data.anchors[us];active=a>=0;a=a.clip(min=0);am=data.m[a];ast=data.rows['rating'][a].astype('i')-1
        prior=active&data.seen[a];anchor=model.ep[am,ast]
        total=model.total[u]-(anchor*prior[:,:,None]).sum(1);count=model.uc[u]-prior.sum(1)
        m=data.m[rows];day=data.day[rows].astype('f')-314
        diff=(day-model.md[u,None])/model.sd;dev=np.sign(diff)*abs(diff)**.4
        bias=model.b[m]+model.ub[u,None]+model.drift[u,None]*dev[:,:,None]
        op=model.op[m];_,masks=data.masks(rows)
        for mode,mask in enumerate(masks):
            z=total[:,None]+mask.astype('f')@anchor
            h=np.tanh(z/np.sqrt(np.maximum(1,count[:,None]+mask.sum(2))).astype('f')[:,:,None]+model.eb)
            logits=np.einsum('bnrh,bnh->bnr',op,h)+bias;logits-=logits.max(2,keepdims=True)
            p=np.exp(logits);p/=p.sum(2,keepdims=True)
            result[rows[valid],mode]=p[valid]
            if mode==1:
                old=valid&(rows<data.n);native[rows[old]]=p[old]@stars
        if j%(256*400)==0:print('users',j,flush=True)
    result.flush();diff=float(np.max(abs(native[data.ix]-full.cf96('probex'))));assert diff<.0003,diff
    report=dict(native_probex_replay_max_difference=diff,shared_checkpoint='trainx CF96 epoch 95',trained_query_rows_used_as_targets=False,
                qualifying_labels_read=False,probabilities_sha256=ne.sha(OUT/'probabilities.npy'))
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
