"""Durable full-arm tree study component; no CLI, queue or production admission.

The future owned controller must authorize each phase and the complete
producer/consumer group. Preparation belongs in a dedicated source process;
fitting remains Torch-free and requires the actual owned producer-exit gate.
"""
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import upstream_full_context_catboost as backend
import upstream_full_context_catboost_consumer as consumer
import upstream_full_context_catboost_wire as wire
import upstream_full_context_catboost_artifacts as artifacts
from upstream_factored_rbm_terminal_metal import durable_directory, durable_payload

ROOT = Path(__file__).resolve().parents[1]
ROWS = 1128164
QUALIFYING_ROWS = 2817131
ARMS = ('scalar', 'latent')
SEEDS = (2026, 7, 13)
BASELINE = ROOT / 'experiments/upstream-factored-rbm-execution-deterministic-metal/first-scientific-baseline.json'
BASELINE_SHA = '689232383581ed51cbdffc8a1da5ae047ee94d690ea51b64bccbbab20b447168'
PREPARED = 'prepared_complete_full_context_tree_arm_pending_owned_execution_and_fits'
FITTED = 'complete_five_full_context_tree_folds_pending_independent_full_replay'


def paths(arm):
    if arm not in ARMS: raise ValueError('Exact full scalar/latent arm required')
    stem = 'upstream-full-context-catboost-' + arm
    return ROOT / 'artifacts' / stem, ROOT / 'experiments' / stem


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''): h.update(block)
    return h.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'catboost')}


def code_bindings():
    names = ['upstream_full_context_catboost_study.py', 'upstream_full_context_catboost.py',
        'upstream_full_context_catboost_consumer.py', 'upstream_full_context_catboost_producer_v2.py',
        'upstream_full_context_catboost_wire.py', 'upstream_full_context_catboost_stream.py',
        'upstream_full_context_catboost_fold.py', 'upstream_full_context_catboost_artifacts.py',
        'upstream_full_context_catboost_family.py', 'verify_upstream_full_context_catboost_fold.py']
    files = [ROOT / 'scripts' / name for name in names] + [ROOT / 'OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def ownership_bindings(ownership, guard):
    if set(ownership) != {'plan', 'monitor', 'baseline', 'gate', 'ownership'}:
        raise ValueError('All five externally authorized immutable prepare bindings required')
    for row in ownership.values():
        consumer.verified_json(Path(row['path']), row['sha256'], ROOT, guard)
    if ownership['baseline']['sha256'] != BASELINE_SHA:
        raise ValueError('Original byte-identical scientific baseline required')


def save_json(path, value, guard):
    raw = wire.encoded(value) + b'\n'
    return durable_payload(path, lambda stream: stream.write(raw), ROOT, guard)


def save_array(path, values, guard):
    return durable_payload(path, lambda stream: np.save(stream, values, allow_pickle=False), ROOT, guard)


def producer_contract(metadata, metadata_sha):
    # The source process legitimately imports Torch. Its producer has already
    # enforced complete CPU source admission; the isolated consumer repeats
    # that admission before fitting. Do not import or call its no-Torch gate
    # from source-process preparation.
    dims=consumer.operators.dimensions(metadata['retain_scalar'],metadata['arm']=='latent')
    if (metadata['format']!='own_full_context_tree_original_source_v1' or metadata['arm'] not in ARMS
            or metadata['rows']!=ROWS or metadata['qualifying_rows']!=QUALIFYING_ROWS
            or metadata['dimensions']!=dims or metadata['ridge_penalty']!=1000. or metadata['ridge_equations']!=10
            or metadata['qualifying_labels_read'] is not False or metadata['models_fitted']!=0
            or metadata['nonlinear_component_OOF_predictions_used_as_features'] is not False):
        raise ValueError('Complete original admitted producer metadata required')
    return wire.contract(ROWS,retain_scalar=metadata['retain_scalar'],latent=metadata['arm']=='latent',
        weights_sha256=metadata['weights_sha256'],admission_sha256=metadata_sha,operator_sha256=metadata['operator_sha256'])


def prepare_arm(arm, prepared, ownership, guard):
    run, report = paths(arm)
    if run.exists() or report.exists():
        raise ValueError('Existing complete or partial tree namespace preserved; no retry')
    if not callable(guard): raise ValueError('External owned full-group guard required')
    ownership_bindings(ownership, guard)
    metadata, expected, weights = [prepared[key] for key in ('metadata', 'contract', 'weights')]
    metadata_sha = hashlib.sha256(wire.encoded(metadata)).hexdigest()
    contract = producer_contract(metadata, metadata_sha)
    if (metadata['arm'] != arm or metadata['rows'] != ROWS or metadata['qualifying_rows'] != QUALIFYING_ROWS
            or wire.encoded(expected) != wire.encoded(contract)
            or weights.shape != (5, 2, metadata['dimensions']['design_columns']) or weights.dtype != np.dtype('<f8')
            or not np.isfinite(weights).all() or hashlib.sha256(weights.tobytes()).hexdigest() != metadata['weights_sha256']):
        raise ValueError('Both complete original row populations and ten exact eligible ridge systems required')
    records = consumer.verify_files(ROOT, metadata['source_sha256'], guard)
    code = code_bindings(); guard()
    durable_directory(run, ROOT, guard); durable_directory(report, ROOT, guard)
    outputs = dict(source=save_json(report / 'source.json', metadata, guard),
                   contract=save_json(report / 'contract.json', contract, guard),
                   weights=save_array(run / 'weights.npy', weights, guard))
    manifest = dict(state=PREPARED, arm=arm, rows=ROWS, qualifying_rows=QUALIFYING_ROWS,
        dimensions=metadata['dimensions'], parameters=backend.PARAMETERS, max_rounds=backend.MAX_ROUNDS,
        patience=backend.PATIENCE, seed_order=list(SEEDS), folds=5, versions=versions(),
        source_metadata_sha256=metadata_sha, code_sha256=code, authorized_prepare_bindings=ownership, outputs=outputs,
        schedule='all_independent_inner_selections_then_fresh_full_refits',
        qualifying_labels_read=False, production_study_admitted=False, models_fitted=0)
    if consumer.verify_files(ROOT, metadata['source_sha256'], guard) != records or code_bindings() != code:
        raise ValueError('Every original source identity and tree implementation must survive preparation')
    return save_json(report / 'manifest.json', manifest, guard)


def prepare(ownership, guard):
    # Importing this function's module remains Torch-free. Only the dedicated
    # source worker may invoke prepare and import the original source adapter.
    for arm in ARMS:
        if any(p.exists() for p in paths(arm)):
            raise ValueError('Earlier partial/full paired study preserved; no restart')
    import upstream_full_context_catboost_producer_v2 as producer
    return {arm: prepare_arm(arm, producer.prepare(arm, guard), ownership, guard) for arm in ARMS}


def load(arm, manifest_sha256, guard):
    consumer.no_torch(); run, report = paths(arm)
    m = consumer.verified_json(report / 'manifest.json', manifest_sha256, ROOT, guard)
    if (m['state'] != PREPARED or m['arm'] != arm or m['rows'] != ROWS or m['qualifying_rows'] != QUALIFYING_ROWS
            or m['parameters'] != backend.PARAMETERS or m['max_rounds'] != 2000 or m['patience'] != 100
            or m['seed_order'] != list(SEEDS) or m['folds'] != 5 or m['versions'] != versions()
            or m['code_sha256'] != code_bindings() or m['qualifying_labels_read'] is not False
            or m['production_study_admitted'] is not False or m['models_fitted'] != 0
            or m['schedule'] != 'all_independent_inner_selections_then_fresh_full_refits'
            or set(m['outputs']) != {'source', 'contract', 'weights'}):
        raise ValueError('Original complete source-bound full-budget tree preparation required')
    ownership_bindings(m['authorized_prepare_bindings'], guard)
    required = dict(source=report/'source.json', contract=report/'contract.json', weights=run/'weights.npy')
    for name, path in required.items():
        row = m['outputs'][name]
        if Path(row['path']) != path or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Exact original complete prepared tree files required')
    metadata = consumer.verified_json(required['source'], m['outputs']['source']['sha256'], ROOT, guard)
    contract = consumer.verified_json(required['contract'], m['outputs']['contract']['sha256'], ROOT, guard)
    expected = consumer.contract(metadata, m['source_metadata_sha256'], ROOT)
    if metadata['arm'] != arm or metadata['dimensions'] != m['dimensions'] or wire.encoded(contract) != wire.encoded(expected):
        raise ValueError('Matched original source, arm and complete raw stream contract required')
    records = consumer.verify_files(ROOT, {str(required['weights'].relative_to(ROOT)):m['outputs']['weights']['sha256']}, guard)
    with consumer.cache.mapped(required['weights'], records[str(required['weights'].relative_to(ROOT))], '<f8',
            (5, 2, m['dimensions']['design_columns']), guard) as weights:
        if hashlib.sha256(weights.tobytes()).hexdigest() != metadata['weights_sha256']:
            raise ValueError('All ten original eligible ridge coefficients required')
    return m, metadata, contract


def fit(arm, manifest_sha256, source, *, guard, producer_receipt=None, producer_receipt_sha256=None, producer_identity=None, producer_exit_gate=None):
    consumer.no_torch(); run, report = paths(arm)
    if (run/'folds').exists() or (report/'fit-started.json').exists() or (report/'fit-completion.json').exists():
        raise ValueError('Original partial/full learned tree state preserved; no restart')
    m, metadata, expected = load(arm, manifest_sha256, guard)
    save_json(report/'fit-started.json', dict(manifest_sha256=manifest_sha256, qualifying_labels_read=False), guard)
    features, baselines, wire_proof = wire.receive(source, expected, guard=guard)
    if producer_exit_gate is not None:
        if (not callable(producer_exit_gate) or producer_receipt is not None
                or producer_receipt_sha256 is not None or producer_identity is not None):
            raise ValueError('One unambiguous authorized producer-exit source required')
        # The producer cannot exit while blocked on a full pipe. First receive
        # the complete stream, then let the owned gate wait for its observed
        # zero exit and immutable receipt without dropping the matrix/stack.
        gate=producer_exit_gate(wire_proof)
        if set(gate)!={'path','sha256','producer'}:
            raise ValueError('Actual owned producer receipt and captured identity required')
        producer_receipt,producer_receipt_sha256,producer_identity=gate['path'],gate['sha256'],gate['producer']
    if producer_receipt is None or producer_receipt_sha256 is None or producer_identity is None:
        raise ValueError('Actual original producer exit receipt required before any fitting')
    outputs = {}; folds = {}; inventory = {}
    def directory(fold):
        path=run/'folds'/str(fold); durable_directory(path, ROOT, guard); return path
    def save_model(fold, model, seed, count):
        name='tree-%s.cbm'%seed
        row=artifacts.save_model(directory(fold)/name, model, seed=seed+fold, iterations=count,
                                 width=m['dimensions']['nn_inputs'], root=ROOT, guard=guard)
        outputs.setdefault(fold,{})[name]=row; inventory.setdefault(fold,{})[str(seed)]=row
    def save_population(fold, stage, proof):
        if stage not in ('inner','full'): raise ValueError('Original inner/full population required')
        name=stage+'-pipeline.json'; outputs.setdefault(fold,{})[name]=save_json(directory(fold)/name,proof,guard)
    def save_fold(fold, result, prediction):
        if result['fold']!=fold or result['seed_order']!=list(SEEDS) or len(result['chosen_iterations'])!=3:
            raise ValueError('All three original ordered seed slots required')
        counts=result['chosen_iterations']
        if any(type(c) is not int or not 0<=c<=backend.MAX_ROUNDS for c in counts):
            raise ValueError('Original full-budget known-only selected counts required')
        if prediction.ndim!=1 or prediction.dtype!=np.dtype('<f4') or not np.isfinite(prediction).all():
            raise ValueError('Complete finite float32 original held predictions required')
        for seed,count in zip(SEEDS,counts):
            record=inventory.get(fold,{}).get(str(seed))
            if count and (record is None or record['seed']!=seed+fold or record['iterations']!=count or record['width']!=m['dimensions']['nn_inputs']):
                raise ValueError('Actual saved model must match every original selected seed/count slot')
            if not count and record is not None:
                raise ValueError('Zero original iteration slot cannot contain a model')
        rows=outputs.setdefault(fold,{})
        rows['prediction.npy']=save_array(directory(fold)/'prediction.npy',prediction,guard)
        models={'tree-%s.cbm'%s for s,c in zip(SEEDS,counts) if c}
        expected_files=models|{'inner-pipeline.json','full-pipeline.json','prediction.npy'}
        if set(rows)!=expected_files or {p.name for p in directory(fold).iterdir()}!=expected_files:
            raise ValueError('Exact complete saved positive model/population/prediction inventory required')
        value=dict(result, manifest_sha256=manifest_sha256, outputs=rows, models=inventory.get(fold,{}),
                   actual_saved_model_count=len(models), qualifying_labels_read=False)
        path=report/'folds'/('%s.json'%fold);folds[str(path.relative_to(ROOT))]=save_json(path,value,guard)['sha256']
    result=consumer.fit(features,baselines,metadata,m['source_metadata_sha256'],wire_proof,
        producer_receipt=producer_receipt,producer_receipt_sha256=producer_receipt_sha256,producer_identity=producer_identity,
        root=ROOT,guard=guard,save_model=save_model,save_population=save_population,save_fold=save_fold,release=gc.collect)
    if (len(folds)!=5 or result['prediction'].shape!=(ROWS,) or result['prediction'].dtype!=np.dtype('<f4')
            or not np.isfinite(result['prediction']).all()):
        raise ValueError('Every original held row and all five saved fold receipts required')
    prediction=save_array(run/'prediction.npy',result['prediction'],guard)
    load(arm,manifest_sha256,guard)
    value=dict(state=FITTED,arm=arm,manifest_sha256=manifest_sha256,fold_proof_sha256=folds,prediction=prediction,
        all_five_folds_complete=True,original_seed_slots=15,actual_saved_model_count=sum(len(v) for v in inventory.values()),
        zero_slots_retained_in_denominator=True,qualifying_labels_read=False,production_study_admitted=False,
        remaining='Complete independent raw/operator/ridge/saved-model replay and canonical known-only score')
    save_json(report/'fit-completion.json',value,guard)
    return value
