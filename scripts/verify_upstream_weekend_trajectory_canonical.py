#!/usr/bin/env python3
"""Separate complete canonical NN/fixed-half arithmetic after actual model replay."""
import argparse
import math
from pathlib import Path
import numpy as np
import upstream_weekend_trajectory_mps as study

u=study.u


def score(values,labels):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(np.clip(values.astype(np.float64),1.05,4.95),labels))/len(labels))


def main(variant):
    run,report=study.paths(variant);output=report/'independent-canonical-verification.json'
    if output.exists():raise ValueError('Existing separate matched neural canonical proof retained')
    mp=report/'manifest.json';vp=report/'independent-verification.json';m=u.read(mp);v=u.read(vp)
    if (v['state']!='complete_independent_complete_trajectory_actual_models_full_eligible_statistics_and_predictions' or
        v['manifest_sha256']!=study.data.ops.sha(mp) or m['code_sha256']!=study.hashes() or m['config']!=study.CONFIGS[variant] or
        not v['complete_original_statistics_bitwise_equal'] or not v['all_original_models_preserved_without_refitting'] or
        not v['matched_B_known_and_qualifying'] or v['qualifying_labels_read'] is not False or
        [r['fold'] for r in v['records']]!=list(range(5)) or
        not all(r['both_complete_eligible_scalers_independently_recomputed'] and r['all_normalized_input_bits_equal'] and r['actual_saved_pipeline_bitwise_equal'] for r in v['records'])):
        raise ValueError('All original eligible preprocessing and actual full saved-model replays required')
    study.data.verify_bindings(m['source_sha256']);labels=np.load(run/'labels.npy');groups=np.load(run/'groups.npy')
    nn=np.fromfile(run/'nn.f32',dtype='<f4');half=np.fromfile(run/'fixed-half-tree154.f32',dtype='<f4')
    if (nn.shape!=(1128164,) or half.shape!=nn.shape or not np.isfinite(nn).all() or not np.isfinite(half).all() or
        study.data.ops.sha(run/'nn.f32')!=v['nn_sha256'] or study.data.ops.sha(run/'fixed-half-tree154.f32')!=v['fixed_half_sha256']):
        raise ValueError('Complete original independently replayed NN and half vectors required')
    reference=np.empty(len(nn),dtype='<f4');models=0
    for row in v['records']:
        receipt=u.read(u.ROOT/row['receipt_path']);root=run/'nn'/str(row['fold'])
        if study.data.ops.sha(u.ROOT/row['receipt_path'])!=row['receipt_sha256'] or receipt['outputs']!=row['outputs']:
            raise ValueError('Original complete actual fit receipt differs')
        for filename,digest in receipt['outputs'].items():
            if study.data.ops.sha(root/filename)!=digest:raise ValueError('Every actual saved pipeline file must remain unchanged')
        if sum(c>0 for c in receipt['chosen_iterations'])!=row['actual_models_reopened']:raise ValueError('Every selected actual model must reopen')
        values=np.fromfile(root/'prediction.f32',dtype='<f4');held=groups//2==row['fold']
        if values.shape!=(int(held.sum()),):raise ValueError('Complete held-fold vector required')
        reference[held]=values;models+=row['actual_models_reopened']
    if not np.array_equal(reference.view('u4'),nn.view('u4')):raise ValueError('Every complete canonical NN bit must match original held folds')
    tree=np.fromfile(u.ROOT/m['controls'][0]['prediction'],dtype='<f4')
    if tree.shape!=nn.shape or not np.isfinite(tree).all():raise ValueError('Complete original tree OOF control required')
    expected=np.clip(.5*nn.astype(np.float64)+.5*tree.astype(np.float64),1.05,4.95).astype('<f4')
    if not np.array_equal(expected.view('u4'),half.view('u4')):raise ValueError('Every stored fixed-half bit must reconstruct')
    scores=[score(values,labels) for values in [nn,half]]
    if abs(scores[0]-v['nn_rmse'])>1e-12 or abs(scores[1]-v['fixed_half_tree154_rmse'])>1e-12:raise ValueError('Separate full compensated canonical score differs')
    old=u.read(u.ROOT/'experiments/upstream-own391-paired-h128-mps/independent-canonical-verification.json')
    local_path=u.ROOT/'experiments/upstream-complete-trajectory-h128-mps/independent-canonical-verification.json'
    local=u.read(local_path)
    references=dict(original391=old['fixed_half_tree154_rmse'],preceding_local_trajectory=local['fixed_half_tree154_rmse'])
    if (m['declared_reference_scores']!=references or m['declared_global_reference_rmse']!=min(references.values()) or
        m['preceding_local_trajectory_reference']['sha256']!=study.data.ops.sha(local_path) or
        m['preceding_local_trajectory_reference']['fixed_half_rmse']!=references['preceding_local_trajectory']):
        raise ValueError('Both proof-bound original and preceding-local known references must independently match')
    study.data.verify_bindings(m['source_sha256'])
    u.write(output,dict(state='complete_separate_complete_trajectory_canonical_neural_and_fixed_half_scores',variant=variant,
        manifest_sha256=study.data.ops.sha(mp),complete_actual_model_replay_sha256=study.data.ops.sha(vp),verifier_sha256=study.data.ops.sha(Path(__file__)),
        actual_models_rehashed=models,all_five_original_folds_verified=True,every_nn_and_fixed_half_float32_bit_equal=True,
        nn_rmse=scores[0],fixed_half_tree154_rmse=scores[1],prediction_sha256=study.data.ops.sha(run/'fixed-half-tree154.f32'),
        declared_global_reference_rmse=m['declared_global_reference_rmse'],improves_previous=scores[1]<m['declared_global_reference_rmse'],
        declared_reference_scores=m['declared_reference_scores'],
        preceding_local_trajectory_reference=m['preceding_local_trajectory_reference'],
        improves_original391=scores[1]<m['declared_reference_scores']['original391'],
        improves_preceding_local_trajectory=scores[1]<m['declared_reference_scores']['preceding_local_trajectory'],
        source_sha256=m['source_sha256'],qualifying_labels_read=False,goal_complete=False,
        tree_control_retains_historical_A_B_realization_caveat=True,neural_inputs_matched_B_on_both_queries=True))
    print('Separate canonical matched391',variant,'NN/fixedhalf:',scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=study.CONFIGS,default='h128');args=parser.parse_args();main(args.variant)
