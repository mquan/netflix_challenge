"""Calibrate the terminal blend on held users, with a separate command to score the final frozen file."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_overnight_receipt as original
import own_retained_history as ret
import own_schema_branch as branch
import own_schema_receipt as receipt

OUT=ne.ROOT/'own-final-calibration-20261126'
DEST=Path('experiments/OWN-final-overnight-20261126.json')
KNOTS=np.array([1.05,1.16,1.38,1.7,2.1,2.55,3.,3.45,3.85,4.15,4.45,4.67,4.8,4.89,4.95])
STATE=branch.OUT
RECEIPT=receipt.DEST
RIDGE=450.
SHRINKAGE=(.25,.5,.75,1.)


def chosen():
    root=blend.OUT/receipt.read(STATE/'queue.json')['parent']
    while True:
        baseline=receipt.read(root/'freeze.json')
        if baseline['selected']=='candidate':return root,baseline
        assert baseline['selected']=='parent';root=blend.OUT/baseline['parent']


def raw_sum(rec,known):
    _,folds,p,q=anchor.history_control()
    base=p if known else q
    result=base.copy();weights=np.array(rec['fold_weights'] if known else [rec['weights']]);at=0
    refs=rec['known_sources' if known else 'qual_sources']
    for key in rec['columns']:
        a,ref=blend.direction(key,p,q,known);assert ref==refs[key];end=at+a.shape[1]
        if known:
            for k in range(5):
                m=folds==k;result[m]+=a[m]@weights[k,at:end]
        else:result+=a@weights[0,at:end]
        at=end
    assert at==weights.shape[1]
    return result


def role(phase,p,raw):
    train='trainx' if phase=='probex' else 'fulltrain'
    prior=np.load(cf.OUT/(phase+'.npy'),mmap_mode='r')[:,7:12]
    star=np.arange(1,6,dtype='f');mu=prior@star
    np.testing.assert_allclose(mu,full.cf96(phase),rtol=0,atol=.0004)
    u,m=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
    nu,ni=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    att=ne.ROOT/'own-cf-temporal-attention-20261126'/(phase+'.npy')
    assert receipt.streamed_sha(att)==receipt.read((blend.OUT/'schema-07')/'freeze.json')['known_sources' if phase=='probex' else 'qual_sources']['attention:own-cf-temporal-attention-20261126']['sha256']
    temporal=np.load(att,mmap_mode='r')
    context=np.column_stack((np.ones(len(p)),np.tanh(mu-p),np.tanh(temporal-p),np.sqrt((prior@(star**2)-mu**2).clip(.001))-.9,
             prior[:,0]+prior[:,4]-.5,np.tanh((np.log1p(nu[u])-5)/2),np.tanh((np.log1p(ni[m])-8)/2)))
    idx=np.searchsorted(KNOTS,p,side='right').clip(1,len(KNOTS)-1)-1
    t=((p-KNOTS[idx])/(KNOTS[idx+1]-KNOTS[idx])).clip(0,1)
    halo=(raw-p).clip(-.30,.30)
    extras=np.column_stack((halo,(p<1.051),(p>4.949),halo*(mu-p),halo*(temporal-p)))
    return idx,t,context,extras


def block(role,rows):
    idx,t,c,e=(a[rows] for a in role);width=c.shape[1];n=len(idx)
    smooth=np.zeros((n,len(KNOTS),width));ix=np.arange(n)
    smooth[ix,idx]=(1-t[:,None])*c;smooth[ix,idx+1]=t[:,None]*c
    return np.column_stack((smooth.reshape(n,-1),e))


def view(data,b,y,f):
    n=block(data,slice(0,1)).shape[1];g=np.zeros((5,n,n));rhs=np.zeros((5,n))
    for j in range(0,len(b),32768):
        sl=slice(j,min(len(b),j+32768));x=block(data,sl);err=y[sl]-b[sl]
        for k in range(5):
            take=f[sl]==k;xx=x[take];g[k]+=xx.T@xx;rhs[k]+=xx.T@err[take]
    def fit(gg,rr):
        mat=gg.copy();mat.flat[::n+1]+=RIDGE
        return np.linalg.solve(mat,rr)
    w=fit(g.sum(0),rhs.sum(0))
    ws=[fit(g.sum(0)-g[k],rhs.sum(0)-rhs[k]) for k in range(5)]
    delta=np.empty(len(y))
    for j in range(0,len(b),32768):
        sl=slice(j,min(len(b),j+32768));x=block(data,sl)
        for k in range(5):
            rows=f[sl]==k;delta[sl][rows]=x[rows]@ws[k]
    return delta,w,ws


def prepare():
    root,rec=chosen();dest=OUT/root.name
    if (dest/'freeze.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);y,f,_,_=anchor.history_control()
    b=np.load(root/'known-selected.npy');raw=raw_sum(rec,True)
    assert np.max(abs(ne.clipped(raw)-b))<1e-11
    data=role('probex',b,raw);held,w,ws=view(data,b,y,f);del data
    stats={'control':blend.metrics(b,y,f)}
    for shrink in SHRINKAGE:stats[str(shrink)]=blend.metrics((b+shrink*held).clip(1,5),y,f)
    eligible=[k for k in stats if ret.qualifies(stats[k],stats['control'])]
    key=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    rep=dict(parent=str(root),parent_freeze_sha256=receipt.streamed_sha(root/'freeze.json'),parent_vector_sha256=rec['qualifying_hashes']['selected'],
            selected=key,known=stats,probex_rows=len(y),knots=KNOTS.tolist(),ridge=RIDGE,weights=w.tolist(),fold_weights=[v.tolist() for v in ws],
            qualifying_hashes={},qualifying_labels_used_for_fitting_or_selection=False,known_comparison_is_nested=False)
    branch.write(dest/'selection.json',rep)
    path=dest/'selected.f32';qb=np.fromfile(root/'selected.f32','<f4');assert receipt.streamed_sha(root/'selected.f32')==rep['parent_vector_sha256']
    if key=='control':
        if not path.exists():os.link(root/'selected.f32',path)
    else:
        raw=raw_sum(rec,False);assert np.max(abs(ne.clipped(raw).astype('f')-qb))<5e-7
        data=role('qual',qb.astype(float),raw);pred=np.empty(len(qb),'<f4')
        for j in range(0,len(qb),32768):
            sl=slice(j,min(len(qb),j+32768));pred[sl]=np.clip(qb[sl]+float(key)*(block(data,sl)@w),1,5)
        assert np.isfinite(pred).all();pred.tofile(path)
    rep['qualifying_hashes']['selected']=receipt.streamed_sha(path)
    branch.write(dest/'freeze.json',rep);print('prepared',root.name,key,stats[key]['pooled'],flush=True)


def exposure_scores(users, anchor_users, ratings, forecasts):
    counts=np.bincount(anchor_users,minlength=max(int(users.max()),int(anchor_users.max()))+1)[users]
    result={}
    for level in range(5):
        take=counts==level if level<4 else counts>=level
        n=int(take.sum())
        if not n:continue
        scores={k:ne.rmse(v[take],ratings[take]) for k,v in forecasts.items()}
        result[str(level) if level<4 else '4_or_more']=dict(rows=n,proportion=n/len(users),scores=scores)
    return result


def compare_frozen_vectors(report, alternative):
    """Paired audit after both scores are frozen, with users as independent clusters."""
    labels=np.loadtxt(ne.LABELS,delimiter=',',skiprows=1,dtype=np.int8)
    users=ne.history('qual','user_idxs')
    baseline=ne.BASE/'followup'/'separate_context_and_tree.f32'
    baseline_hash=receipt.read(ne.BASE/'followup'/'freeze.json')['qualifying_hashes']['separate_context_and_tree']
    assert receipt.streamed_sha(baseline)==baseline_hash
    files={'baseline':baseline,'primary':report['qualifying_vector'],'without_stack':alternative['qualifying_vector']}
    scores={'baseline':report['baseline'],'primary':report['qualifying'],'without_stack':alternative['qualifying']}
    comparisons=[('primary','baseline'),('without_stack','baseline'),('without_stack','primary')]
    extra=report.get('stable_subspace_diagnostic',{})
    if 'qualifying_vector' in extra:
        files['stable_linear']=extra['qualifying_vector'];scores['stable_linear']=extra['qualifying']
        comparisons.extend([('stable_linear','baseline'),('stable_linear','without_stack')])
    forecasts={name:np.fromfile(path,'<f4').astype(float) for name,path in files.items()}
    squared={name:np.square(v-labels[:,0]) for name,v in forecasts.items()}
    assert all(len(v)==len(users) for v in squared.values())
    comparison={}
    for group in (0,1):
        name=f'source{group}';ix=labels[:,1]==group;u=users[ix];n=len(u);clusters=np.count_nonzero(np.bincount(u))
        rmse={k:float(np.sqrt(v[ix].mean())) for k,v in squared.items()}
        for k in files:
            assert n==scores[k][name]['rows'] and abs(rmse[k]-scores[k][name]['rmse'])<1e-12
        rows={}
        for a,b in comparisons:
            influence=(squared[a][ix]-rmse[a]**2)/(2*rmse[a])-(squared[b][ix]-rmse[b]**2)/(2*rmse[b])
            sums=np.bincount(u,weights=influence)
            se=float(np.sqrt((sums@sums)*clusters/(clusters-1))/n);diff=rmse[a]-rmse[b]
            rows[a+'_minus_'+b]=dict(rmse_difference=diff,user_cluster_standard_error=se,
                                       approximate_95_percent_interval=[diff-1.96*se,diff+1.96*se])
        comparison[name]=dict(users=clusters,rows=n,comparisons=rows)
    old_users=anchor.old_probe()[0];held=ret.map_rows();observed=np.ones(len(old_users),bool);observed[held]=False
    dev_users=ne.history('probex','user_idxs');np.testing.assert_array_equal(old_users[held],dev_users)
    development={'primary_parent':np.load(blend.OUT/report['parent_stage']/'known-selected.npy'),
                 'without_stack':np.load(blend.OUT/alternative['stage']/'known-selected.npy')}
    panels={'probex':exposure_scores(dev_users,old_users[observed],ne.history('probex','ratings'),development)}
    for group in (0,1):
        ix=labels[:,1]==group
        panels[f'source{group}']=exposure_scores(users[ix],old_users,labels[ix,0],{k:v[ix] for k,v in forecasts.items()})
    return dict(method='Paired RMSE delta method with one cluster per Netflix user. Intervals describe user sampling variability conditional on the fitted models and do not adjust for adaptive experiments.',
                baseline_vector=str(baseline),baseline_sha256=baseline_hash,groups=comparison,
                by_number_of_visible_original_probe_peers=panels)


def publish():
    root,rec=chosen();dest=OUT/root.name;rep=receipt.read(dest/'freeze.json')
    assert rep['parent_freeze_sha256']==receipt.streamed_sha(root/'freeze.json')
    assert rep['parent_vector_sha256']==receipt.streamed_sha(root/'selected.f32')
    assert rep['qualifying_hashes']['selected']==receipt.streamed_sha(dest/'selected.f32')
    if not (dest/'audit.json').exists():ne.audit(dest,ne.LABELS)
    spec=importlib.util.spec_from_file_location('streaming_verifier',original.VERIFY)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    scores=verifier.verify_audit(dest,rep['qualifying_hashes'])['selected'];parent=receipt.read(RECEIPT)
    assert parent['selected_at']==root.name
    report=dict(as_of=datetime.now().astimezone().isoformat(),target=.85725,scope=parent['scope'],
        selection='Lower pooled ProbeX RMSE and improvement on at least four of five user folds.',qualifying=scores,
        qualifying_rows=sum(v['rows'] for v in scores.values()),qualifying_vector=str(dest/'selected.f32'),sha256=rep['qualifying_hashes']['selected'],
        selected_calibration=rep['selected'],known=rep['known'][rep['selected']],parent_stage=root.name,parent_receipt=str(RECEIPT),
        parent_receipt_sha256=receipt.streamed_sha(RECEIPT),parent_scores=parent['qualifying'],baseline=parent['baseline'],
        qualifying_labels_used_for_fitting_or_selection=False,known_comparison_is_nested=False,recovered_labels_are_official_evaluation=False,
        improvement_over_baseline={k:parent['baseline'][k]['rmse']-v['rmse'] for k,v in scores.items()},
        distance_above_target={k:v['rmse']-.85725 for k,v in scores.items()},freeze=str(dest/'freeze.json'),freeze_sha256=receipt.streamed_sha(dest/'freeze.json'),
        independent_verifier=str(original.VERIFY),verifier_sha256=receipt.streamed_sha(original.VERIFY),label_sha256=receipt.streamed_sha(ne.LABELS),
        exploratory_design=parent.get('exploratory_design'),calibration_recipe=str(Path(__file__).resolve().relative_to(Path.cwd())),calibration_recipe_sha256=receipt.streamed_sha(__file__))
    ablation=ne.ROOT/'own-paired-ablation-20261126'
    if parent.get('paired_input_audit') and (ablation/'queue.json').exists():
        report['primary_parent_recipe_reconstruction']=parent.get('recipe_reconstruction') or receipt.verify_recipe(root,rec)
        if rep['selected']=='control':
            assert report['sha256']==parent['sha256']
        report['pairing_audit']=parent['paired_input_audit']
        if 'combined_status_counts' in parent:report['candidate_status_counts']=parent['combined_status_counts']
        state=receipt.read(ablation/'queue.json');alt=blend.OUT/state['parent']
        af=receipt.read(alt/'freeze.json');digest=receipt.streamed_sha(alt/'selected.f32')
        assert digest==af['qualifying_hashes']['selected'];scored=alt
        while not (scored/'audit.json').is_file():
            rf=receipt.read(scored/'freeze.json');assert rf['selected']=='parent';scored=blend.OUT/rf['parent']
        alt_scores=verifier.verify_audit(scored,{'selected':digest})['selected']
        protocol=ablation/'protocol.json'
        alternative=dict(protocol=str(protocol),protocol_sha256=receipt.streamed_sha(protocol),stage=alt.name,
            selected_at=scored.name,qualifying=alt_scores,qualifying_vector=str(alt/'selected.f32'),sha256=digest,stages=state['stages'],
            recipe_reconstruction=receipt.verify_recipe(alt,receipt.read(scored/'freeze.json')),
            distance_above_target={k:v['rmse']-report['target'] for k,v in alt_scores.items()},
            improvement_over_baseline={k:report['baseline'][k]['rmse']-v['rmse'] for k,v in alt_scores.items()})
        report['preregistered_ablation']=alternative
        separate=ne.ROOT/'own-direct-canonical-20261126'/'freeze.json'
        if separate.is_file():
            diagnostic=receipt.read(separate)
            report['separate_bounded_ensemble_diagnostic']=dict(path=str(separate),sha256=receipt.streamed_sha(separate),
                selected=diagnostic['selected'],development=diagnostic['known'],control=diagnostic['control'])
        exposure=ne.ROOT/'own-exposure-balance-20261126'/'freeze.json'
        if exposure.is_file():
            diagnostic=receipt.read(exposure)
            report['unlabeled_exposure_check']=dict(path=str(exposure),sha256=receipt.streamed_sha(exposure),
                selected=diagnostic['selected'],query_mass_without_development_support=diagnostic['query_mass_without_development_support'],
                binning='Observed original Probe peers, any same-day peer, cohort size, user activity, and movie popularity. Counts are computed from rating visibility and query metadata.')
        separate=ne.ROOT/'own-stable-subspace-20261126'
        if (separate/'freeze.json').is_file():
            diagnostics=receipt.read(separate/'freeze.json')
            result=dict(freeze=str(separate/'freeze.json'),freeze_sha256=receipt.streamed_sha(separate/'freeze.json'),
                        protocol=str(separate/'protocol.json'),protocol_sha256=receipt.streamed_sha(separate/'protocol.json'),
                        selected=diagnostics['selected'],retained_axes=diagnostics['retained'],
                        development=diagnostics['known'],role='Supplementary experiment outside both comparison branches.')
            if diagnostics['qualifying_hashes']:
                result['qualifying']=verifier.verify_audit(separate,diagnostics['qualifying_hashes'])['selected']
                result['sha256']=diagnostics['qualifying_hashes']['selected']
                result['qualifying_vector']=str(separate/'selected.f32')
                result['fit_sha256']=receipt.streamed_sha(separate/'fit.npz')
                refit=separate/'independent-refit.json'
                if refit.is_file():
                    rerun=receipt.read(refit);assert rerun['sha256']==result['sha256'] and rerun['different_float32_predictions']==0
                    result['fresh_fit_replay']=dict(path=str(refit),receipt_sha256=receipt.streamed_sha(refit),**rerun)
            report['stable_subspace_diagnostic']=result
        nonlinear=ne.ROOT/'own-stable-nonlinear-20261126'
        if (nonlinear/'freeze.json').is_file():
            nf=receipt.read(nonlinear/'freeze.json')
            result=dict(freeze=str(nonlinear/'freeze.json'),freeze_sha256=receipt.streamed_sha(nonlinear/'freeze.json'),
                protocol=str(nonlinear/'protocol.json'),protocol_sha256=receipt.streamed_sha(nonlinear/'protocol.json'),
                selected=nf['selected'],development=nf['known'],control=nf['control'],movie_popularity_groups=nf['popularity_groups'],
                parent_reconstruction_different_float32=nf['parent_reconstruction_different_float32'])
            if nf['qualifying_hashes']:
                result['qualifying']=verifier.verify_audit(nonlinear,nf['qualifying_hashes'])['selected']
            report['stable_nonlinear_diagnostic']=result
        report['post_freeze_paired_comparison']=compare_frozen_vectors(report,alternative)
    branch.write(DEST,report);print(scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--publish',action='store_true');parser.add_argument('--paired',action='store_true');args=parser.parse_args()
    if args.paired:
        import own_paired_branch
        STATE=own_paired_branch.OUT;RECEIPT=Path('experiments/OWN-paired-overnight-20261126.json')
    if args.publish:publish()
    else:
        while True:
            prepare()
            if not args.watch or datetime.now()>=datetime(2026,11,26,4,59,20):break
            time.sleep(20)
