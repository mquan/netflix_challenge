#!/usr/bin/env python3
"""Fit one convex blend coefficient on known OOF predictions; freeze qualifying output."""
from pathlib import Path
import datetime as dt
import math
import numpy as np
import upstream_item154_stream_production as source
u=source.u
RUN=u.ROOT/'artifacts/upstream-item154-weight-calibration'
REPORT=u.ROOT/'experiments/upstream-item154-weight-calibration'
DEV=u.ROOT/'experiments/upstream-rbm50-tree-item154-mps'
KNOWN_TREE=u.ROOT/'artifacts/upstream-rbm50-tree/tree.f32'
KNOWN_NN=u.ROOT/'artifacts/upstream-item154-mps-context/nn.f32'
FULL_TREE=source.OLD_RUN/'tree/qualifying.f32'
FULL_NN=source.RUN/'nn/qualifying.f32'
LABELS=u.FRESH/'data/probex/ratings.npy'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def checked_sources():
    proof=u.read(DEV/'verification.json');check=u.read(DEV/'independent-mean-check.json')
    if (proof['state']!='complete_verified_fixed_combination' or not proof['original_saved_model_replays_verified'] or
        check['verification_sha256']!=u.sha(DEV/'verification.json') or not check['fixed_mean_bitwise_equal'] or proof['qualifying_labels_read'] is not False):
        raise ValueError('Actual verified original own development families required')
    bindings=dict(proof['source_sha256'])
    for p in [DEV/'verification.json',DEV/'independent-mean-check.json']:bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    # Production lineage is checked without parsing or reading recovered label
    # payloads. Existing audit outcomes are not inputs to the coefficient fit.
    production=u.read(source.REPORT/'independent-verification.json')
    if not production['actual_new_full_neural_replay_verified'] or not production['original_full_model_replays_reused_and_verified']:
        raise ValueError('Actual full family model replays required')
    for name,digest in production['bindings'].items():
        if 'recovered-audit/' in name:continue
        if name in bindings and bindings[name]!=digest:raise ValueError('Conflicting own source binding')
        bindings[name]=digest
    for p in [source.REPORT/'plan.json',source.REPORT/'nn.json',source.REPORT/'nn-replay.json',source.REPORT/'report.json',
              KNOWN_TREE,KNOWN_NN,FULL_TREE,FULL_NN,LABELS,Path(__file__),u.ROOT/'scripts/verify_upstream_item154_weight_calibration.py']:
        bindings[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Original own model/source binding changed: '+name)
    return bindings

def freeze():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing calibration evidence preserved')
    REPORT.mkdir(parents=True)
    u.write(REPORT/'plan.json',dict(created=stamp(),script_sha256=u.sha(Path(__file__)),
        rule='Fix the same current item154 NN and expanded tree, fit exactly one convex tree coefficient w in[0,1] by known OOF least squares: sum((tree-NN)*(rating-NN))/sum((tree-NN)^2), clipped to[0,1]. No search, qualifying outcomes or family changes. Apply the fitted coefficient to already actually replayed own full models. Freeze before the new recovered audit.',
        development_score_scope='Apparent known-fit calibration score, not a new nested user-fold validation score and not a replacement for the existing0.8592309819864167 cross-validation best.',
        known_rows=1128164,qualifying_rows=2817131,qualifying_labels_used_for_fitting_or_selection=False,prior_audit_scores_used_for_calibration=False))
    bindings=checked_sources();y=np.load(LABELS).astype(np.float64)
    tree=np.fromfile(KNOWN_TREE,dtype='<f4').astype(np.float64);nn=np.fromfile(KNOWN_NN,dtype='<f4').astype(np.float64)
    if any(a.shape!=(1128164,) or not np.isfinite(a).all() for a in [y,tree,nn]):raise ValueError('Complete finite canonical known rows required')
    difference=tree-nn;denominator=math.fsum(float(d)*float(d) for d in difference)
    numerator=math.fsum(float(d)*float(r-n) for d,r,n in zip(difference,y,nn))
    if denominator<=0:raise ValueError('Distinct family predictions required')
    weight=float(np.clip(numerator/denominator,0.,1.));weights=[weight,1.-weight]
    RUN.mkdir();known=(weights[0]*tree+weights[1]*nn).astype('<f4');known.tofile(RUN/'known-calibration.f32')
    score=math.sqrt(math.fsum((float(p)-float(t))**2 for p,t in zip(known,y))/len(y))
    coefficients=dict(state='complete_known_fit_single_convex_coefficient',observed=stamp(),weights_tree_nn=weights,numerator=numerator,denominator=denominator,
        known_fit_rmse=score,known_rows=len(y),known_prediction_sha256=u.sha(RUN/'known-calibration.f32'),source_sha256=bindings,
        score_is_nested_cross_validation=False,qualifying_label_payload_read=False,qualifying_labels_used_for_fitting_or_selection=False)
    u.write(REPORT/'calibration.json',coefficients)
    # No qualifying label payload has been opened before the immutable vector.
    fulltree=np.fromfile(FULL_TREE,dtype='<f4').astype(np.float64);fullnn=np.fromfile(FULL_NN,dtype='<f4').astype(np.float64)
    if any(a.shape!=(2817131,) or not np.isfinite(a).all() for a in [fulltree,fullnn]):raise ValueError('Complete finite own qualifying families required')
    prediction=(weights[0]*fulltree+weights[1]*fullnn).astype('<f4');prediction.tofile(RUN/'qualifying.f32')
    for name,digest in bindings.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Own source changed while freezing')
    u.write(REPORT/'frozen.json',dict(state='complete_frozen_own_known_calibrated_prediction',observed=stamp(),plan_sha256=u.sha(REPORT/'plan.json'),
        calibration_sha256=u.sha(REPORT/'calibration.json'),prediction_sha256=u.sha(RUN/'qualifying.f32'),weights_tree_nn=weights,
        qualifying_labels_read=False,qualifying_labels_used_for_fitting_or_selection=False,known_score_is_nested_cross_validation=False))
    print(dict(weights=weights,known_fit_rmse=score,qualifying_frozen=True),flush=True)
if __name__=='__main__':freeze()
