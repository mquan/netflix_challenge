#!/usr/bin/env python3
"""Independent source-reader and actual saved-tree/canonical score replay."""
import upstream_own391_tree_worker as study
import math
from pathlib import Path
import numpy as np
import lightgbm as lgb

u,RUN,REPORT=study.u,study.RUN,study.REPORT


def score(values,y):
    return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(values.astype(np.float64),1.05,4.95),y))/len(y))


def main():
    output=REPORT/'independent-verification.json'
    if output.exists() or (RUN/'tree.f32').exists():raise ValueError('Existing full tree replay retained')
    complete=u.read(REPORT/'fit-completion.json')
    if (complete['state']!='complete_original_budget_matched391_cpu_tree_fits_pending_replay' or
        complete['manifest_sha256']!=u.sha(REPORT/'manifest.json') or complete['qualifying_labels_read'] is not False):
        raise ValueError('All original full-budget fits required')
    m,x,bases,weights,streams=study.receive(independent=True)
    if streams!=complete['streams']:raise ValueError('Every original operator/baseline stream must independently match')
    y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy');users=np.load(RUN/'users.npy');order=np.argsort(users,kind='stable')
    if (y.shape!=(1128164,) or groups.shape!=y.shape or set(np.unique(groups))!=set(range(10)) or
        np.any((users[order][1:]==users[order][:-1])&(groups[order][1:]//2!=groups[order][:-1]//2))):
        raise ValueError('Complete canonical known rows and held-user folds required')
    predictions=np.empty(len(y),dtype='<f4');records=[];reopened=0
    for fold in range(5):
        receipt_path=REPORT/('fold-%s.json'%fold);r=u.read(receipt_path);root=RUN/'tree'/str(fold)
        if (complete['receipt_sha256'][str(fold)]!=u.sha(receipt_path) or r['fold']!=fold or
            r['manifest_sha256']!=u.sha(REPORT/'manifest.json') or len(r['chosen_iterations'])!=3 or len(r['selections'])!=3 or
            any(type(c)!=int or not 0<=c<=2000 for c in r['chosen_iterations'])):
            raise ValueError('Every original selected seed budget required')
        expected={'ridge.npz','prediction.f32'}|{'tree-%s.txt'%s for s,c in zip(m['config']['seeds'],r['chosen_iterations']) if c}
        if set(r['outputs'])!=expected or {p.name for p in root.iterdir()}!=expected:raise ValueError('Actual complete tree artifact inventory differs')
        for name,digest in r['outputs'].items():
            if u.sha(root/name)!=digest:raise ValueError('Actual original tree/prediction changed')
        with np.load(root/'ridge.npz') as saved:
            if not np.array_equal(saved['inner'].view('u8'),weights[fold*2].view('u8')) or not np.array_equal(saved['full'].view('u8'),weights[fold*2+1].view('u8')):
                raise ValueError('Every original eligible ridge coefficient bit must match')
        training=groups//2!=fold;held=~training;valid=training&(groups%2==1)
        inner,full=bases[:,fold*2],bases[:,fold*2+1];x[:,-1]=full
        correction=np.zeros(int(held.sum()));actual_models=0
        baseline_mse=float(np.mean((y[valid]-inner[valid])**2))
        for seed,count,selection in zip(m['config']['seeds'],r['chosen_iterations'],r['selections']):
            if (selection['seed']!=seed or selection['actual_seed']!=seed+fold or selection['chosen']!=count or
                not 1<=selection['best_iteration']<=2000 or abs(selection['baseline_mse']-baseline_mse)>1e-12 or
                count!=(selection['best_iteration'] if selection['best_mse']<baseline_mse else 0)):
                raise ValueError('Original inner-only budget selection record differs')
            if count:
                model=lgb.Booster(model_file=str(root/('tree-%s.txt'%seed)))
                if model.num_trees()!=count:raise ValueError('Actual original saved tree budget differs')
                correction=np.add(correction,model.predict(x[held],num_threads=2)/3)
                del model;actual_models+=1;reopened+=1
        prediction=np.clip(full[held]+correction,1.05,4.95).astype('<f4');actual=np.fromfile(root/'prediction.f32',dtype='<f4')
        if (actual.shape!=prediction.shape or r['rows']!=len(prediction) or not np.isfinite(prediction).all() or
            not np.array_equal(actual.view('u4'),prediction.view('u4'))):
            raise ValueError('Every actual held prediction bit must replay')
        predictions[held]=prediction;records.append(dict(fold=fold,rows=len(prediction),actual_models_reopened=actual_models,
            actual_prediction_bitwise_equal=True,receipt_sha256=u.sha(receipt_path),rmse=score(prediction,y[held])))
    if reopened!=complete['actual_full_models']:raise ValueError('All actual full models must reopen')
    nn=np.fromfile(RUN/'partner-nn.f32',dtype='<f4')
    if nn.shape!=predictions.shape or not np.isfinite(nn).all():raise ValueError('Complete independently verified fixed partner required')
    half=np.clip(.5*predictions.astype(np.float64)+.5*nn.astype(np.float64),1.05,4.95).astype('<f4')
    predictions.tofile(RUN/'tree.f32');half.tofile(RUN/'fixed-half-nn352.f32');study.checked()
    scores=[score(values,y) for values in [predictions,half]]
    u.write(output,dict(state='complete_independent_matched391_CPU_tree_models_inputs_ridge_predictions_and_canonical_scores',
        manifest_sha256=u.sha(REPORT/'manifest.json'),fit_completion_sha256=u.sha(REPORT/'fit-completion.json'),
        verifier_sha256=u.sha(Path(__file__)),source_sha256=m['source_sha256'],records=records,actual_models_reopened=reopened,
        all_original_operator_and_baseline_stream_bits_equal=True,all_actual_saved_tree_predictions_bitwise_equal=True,
        tree_rmse=scores[0],fixed_half_nn352_rmse=scores[1],reference_rmse=m['reference_rmse'],improves_reference=scores[1]<m['reference_rmse'],
        tree_sha256=u.sha(RUN/'tree.f32'),fixed_half_sha256=u.sha(RUN/'fixed-half-nn352.f32'),
        partner_retains_historical_A_B_realization_caveat=True,qualifying_labels_read=False,qualifying_predictions_generated=False))
    print('Independent matched391 tree/fixedhalf:',scores,flush=True)


if __name__=='__main__':main()
