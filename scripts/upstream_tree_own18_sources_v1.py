"""Read-only own18 providers: all original own17 families plus the full tree pair.

No model fitting, query-label reading, score selection or standalone launch.
The eventual owned own18 controller must freeze this source component first.
"""
from pathlib import Path
import upstream_full_context_catboost_bag_execution_v1 as tree_owner
import upstream_rbm400_own17_calibration_reference_v4 as previous
import verify_upstream_rbm400_own17_calibration_reference_v4 as previous_replay

ROOT=tree_owner.ROOT
COUNTS={'known':1128164,'qualifying':2817131}
BASE_IDS=previous.BASE_IDS+[previous.FAMILY_ID]
FAMILY_ID='own-selected-full-context-catboost-fold-bag'
CANDIDATES=('full-tree-own18-simplex','full-tree-own18-ridge10','selected-full-context-tree-control')


def declaration(guard):
    """Return all 18 original ordered families and matched raw tree controls."""
    # The existing seventeen-family proof loader legitimately imports Torch.
    # This is the later blend-provider phase: it never trains or predicts with
    # CatBoost. The original tree inference/replay roots remain Torch-free.
    e=tree_owner
    if not callable(guard):raise ValueError('Original owned resource/authorization guard required')
    # This validates the known-only fit, original 17 paths/order/closure, all
    # six frozen vectors and independent preaudit, without opening Q labels.
    plan,calibration,frozen,records,preaudit=previous_replay.require_preaudit(guard)
    if (calibration['family_ids']!=BASE_IDS or [r['id'] for r in calibration['families']]!=BASE_IDS
            or len(BASE_IDS)!=17 or len(set(BASE_IDS+[FAMILY_ID]))!=18):
        raise ValueError('All original seventeen own families in their original order required')
    bindings=dict(calibration['source_sha256'])
    def record(path,digest=None):
        digest=digest or e.study.sha(path)
        e.selected.merge(bindings,{str(path.relative_to(ROOT)):digest})
        return e.consumer.verified_json(path,digest,ROOT,guard)
    own_plan=record(e.REPORT/'plan.json');e.check_sources(own_plan)
    e.selected.merge(bindings,own_plan['source_sha256'])
    done=record(e.REPORT/'completion.json')
    admitted_path=e.SOURCE_REPORT/'pair-admission.json'
    admitted=record(admitted_path,done['verification_sha256'])
    if (done['state']!=e.COMPLETE or done['production_pair_admitted'] is not True or done['models_fitted']!=0
            or done['known_rows']!=COUNTS['known'] or done['qualifying_rows']!=COUNTS['qualifying']
            or done['qualifying_labels_read'] is not False or admitted['state']!=e.ADMITTED
            or admitted['selection_sha256']!=done['selection_sha256'] or admitted['arm']!=done['selected_arm']
            or admitted['known_rows']!=COUNTS['known'] or admitted['qualifying_rows']!=COUNTS['qualifying']
            or admitted['fold_slots']!=5 or admitted['seed_slots']!=15 or admitted['models_fitted']!=0
            or admitted['qualifying_labels_read'] is not False or admitted['production_data_admitted'] is not False
            or admitted['every_original_raw_ridge_model_fold_bag_and_matched_known_bit_independently_verified'] is not True
            or admitted['both_actual_owned_source_exit_and_authorization_closures_verified'] is not True
            or set(admitted['outputs'])!={'known','qualifying'}):
        raise ValueError('Complete owned full tree pair, independent replay and exact matched populations required')
    completed=done['completed_phase_and_observed_resource_proof_sha256']
    expected={str((e.RESULTS/(p+'.json')).relative_to(ROOT)) for p in e.PHASES}
    expected|={str((e.REPORT/(p+'-resource-peaks.json')).relative_to(ROOT)) for p in e.PHASES}
    if set(completed)!=expected:raise ValueError('All four actual pair phase and four resource receipts required')
    for phase in e.PHASES:
        path=e.RESULTS/(phase+'.json');value=record(path,completed[str(path.relative_to(ROOT))])
        e.validate_result(phase,value,e.study.sha(e.REPORT/'plan.json'))
        path=e.REPORT/(phase+'-resource-peaks.json');value=record(path,completed[str(path.relative_to(ROOT))])
        if (value['state']!='complete_observed_native_original_family_resource_samples' or value['phase']!=phase
                or value['continuous_peak_claimed'] is not False or value['original_baseline_sha256']!=e.study.BASELINE_SHA):
            raise ValueError('Original observed-resource scope and unchanged baseline required')
    release=record(e.REPORT/'owner-release.json')
    if release['state']!='all_original_scientific_roles_terminal_after_complete_selected_pair_replay':
        raise ValueError('External owner must finish with all captured scientific roles terminal')
    for phase in e.PHASES:
        capture=e.read(e.REPORT/(phase+'-ownership.json'))
        if e.family.live_roles(capture,e.roles(phase,capture)):
            raise ValueError('No still-live original pair root or source may enter own18')
        record(e.REPORT/(phase+'-ownership.json'))
    e.selected.merge(bindings,admitted['source_sha256'])
    control={}
    for query,filename in [('known','known-prediction.f32'),('qualifying','prediction.f32')]:
        row=admitted['outputs'][query];path=e.bag.paths()[0]/filename
        if Path(row['path'])!=path or row['bytes']!=4*COUNTS[query]:
            raise ValueError('Exact complete original tree-only matched raw vector required')
        e.selected.merge(bindings,{str(path.relative_to(ROOT)):row['sha256']})
        control[query]=dict(row,format='raw')
    for name in ('plan.json','calibration.json','known-completion.json','frozen.json','independent-preaudit-verification.json'):
        record(previous.REPORT/name)
    e.selected.merge(bindings,{str(Path(__file__).relative_to(ROOT)):e.study.sha(Path(__file__))})
    # Retain the original qualifying-target exclusion before any fitting caller
    # receives these providers. This routine only hashes prediction/source data.
    previous.verify_bindings(bindings,guard)
    row=dict(id=FAMILY_ID,family='own-selected-full-context-catboost',format='raw',selected_arm=admitted['arm'],
        known_prediction=str(Path(control['known']['path']).relative_to(ROOT)),
        qualifying_prediction=str(Path(control['qualifying']['path']).relative_to(ROOT)))
    return list(calibration['families'])+[row],bindings,control
