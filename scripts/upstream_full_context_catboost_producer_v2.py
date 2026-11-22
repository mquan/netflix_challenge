"""Complete original data-source adapter for the planned tree producer.

Import only in the dedicated source process: the original context adapter
imports Torch. No queue, CLI, model fit or production launch is provided.
The future owned supervisor must bind the exact source metadata and wire
contract before authorizing a producer or a Torch-isolated consumer.
"""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import numpy as np
import upstream_rbm400_neural_inputs_reference_v4 as inputs
import upstream_rbm400_latent_cache_reference_v2 as cache
import upstream_full_context_catboost_wire as wire

ROOT = inputs.ROOT
ROWS = 1128164
QUALIFYING_ROWS = 2817131
PLAN = ROOT / 'OWN_FULL_CONTEXT_CATBOOST_FOLLOWUP_PLAN.md'


def sha(path):
    return inputs.cpu.inputs.sha(path)


def eligible_weights(gram, rhs, guard):
    """Ten exact original ridge1000 solves; sum only eligible groups."""
    if (not callable(guard) or rhs.shape not in ((10, 562), (10, 563))
            or gram.shape != (10, rhs.shape[1], rhs.shape[1])
            or gram.dtype != np.float64 or rhs.dtype != np.float64
            or not np.isfinite(gram).all() or not np.isfinite(rhs).all()):
        raise ValueError('All ten full-width original finite sufficient statistics required')
    result = np.empty((5, 2, rhs.shape[1]), dtype='<f8')
    groups = np.arange(10)
    for fold in range(5):
        selected = groups // 2 != fold
        for population, eligible in enumerate((selected & (groups % 2 == 0), selected)):
            guard()
            result[fold, population] = inputs.context.preceding.parent.n.ridge_weights(gram, rhs, eligible, 1000.)
    if not np.isfinite(result).all():
        raise ValueError('Every original eligible ridge coefficient must be finite')
    guard(); return result


def code_paths():
    return [Path(__file__), Path(wire.__file__), Path(wire.__file__).with_name('upstream_full_context_catboost_stream.py'),
            Path(inputs.__file__), Path(cache.__file__), PLAN]


def prepare(arm, guard):
    """Authenticate completed CPU proof, then derive all exact eligible systems.

    The returned metadata is not an owned launch gate. It must be durably
    published and source-bound by the future full-study supervisor first.
    """
    if arm not in ('scalar', 'latent') or not callable(guard):
        raise ValueError('Explicit complete arm and external whole-group guard required')
    completion = inputs.cpu.read(inputs.cpu.REPORT / 'completion.json')
    if (completion['state'] != 'complete_both_RBM400_context_arms_data_and_independent_known_ridge_replay'
            or completion['qualifying_labels_read'] is not False):
        raise ValueError('Actual complete independently verified CPU data required before preparation')
    declaration, manifest, _ = inputs.sources(guard)
    if manifest['rows'] != ROWS or manifest['qualifying_rows'] != QUALIFYING_ROWS:
        raise ValueError('Every original known and qualifying row required')
    bindings = dict(declaration['source_sha256'])
    for path in code_paths(): inputs.context.bind(bindings, path)
    records = inputs.context.provider.original.verify_bindings(bindings, guard)
    declaration['source_sha256'] = bindings
    d = manifest['arms']['scalar']['design_columns']
    with ExitStack() as stack:
        values = {}
        for name, shape in [('gram.npy', (10, d, d)), ('rhs.npy', (10, d))]:
            path = inputs.data.ART / name
            if Path(manifest['prepared'][name]['path']) != path:
                raise ValueError('Exact original complete CPU statistic paths required')
            values[name] = stack.enter_context(cache.mapped(path, records[str(path.relative_to(ROOT))], '<f8', shape, guard))
        weights = eligible_weights(values['gram.npy'], values['rhs.npy'], guard)
    compact = {}
    for name, dtype in [('labels.npy', '<f8'), ('users.npy', '<i4'), ('groups.npy', '|i1')]:
        path = inputs.data.ART / name
        if Path(manifest['prepared'][name]['path']) != path:
            raise ValueError('Exact original known label/user/group source paths required')
        record = records[str(path.relative_to(ROOT))]
        compact[name] = dict(path=str(path), dtype=dtype, shape=[ROWS], **record)
    metadata = dict(format='own_full_context_tree_original_source_v1', arm=arm, rows=ROWS,
        qualifying_rows=QUALIFYING_ROWS, retain_scalar=manifest['alias']['retain_scalar'],
        dimensions=manifest['arms'][arm], source_sha256=bindings, known_arrays=compact,
        parent_CPU_manifest_sha256=sha(inputs.data.REPORT / 'manifest.json'),
        parent_CPU_verification_sha256=sha(inputs.cpu_proof_path()),
        weights_sha256=hashlib.sha256(weights.tobytes(order='C')).hexdigest(),
        operator_sha256=manifest['virtual_c_order_sha256']['probex']['arm_operator_sha256'][arm],
        ridge_penalty=1000., ridge_equations=10, nonlinear_component_OOF_predictions_used_as_features=False,
        qualifying_labels_read=False, models_fitted=0)
    digest = hashlib.sha256(wire.encoded(metadata)).hexdigest()
    expected = wire.contract(ROWS, retain_scalar=metadata['retain_scalar'], latent=arm == 'latent',
        weights_sha256=metadata['weights_sha256'], admission_sha256=digest, operator_sha256=metadata['operator_sha256'])
    if inputs.context.provider.original.verify_bindings(bindings, guard) != records:
        raise ValueError('Every original source identity must survive full tree preparation')
    return dict(metadata=metadata, contract=expected, declaration=declaration, records=records, weights=weights)


def send(prepared, output, guard):
    """Visit every original known row; no full matrix in this process."""
    if not callable(guard): raise ValueError('External owned whole-group producer guard required')
    metadata, expected = prepared['metadata'], prepared['contract']
    if (metadata['format'] != 'own_full_context_tree_original_source_v1'
            or metadata['arm'] not in ('scalar', 'latent') or metadata['rows'] != ROWS
            or metadata['qualifying_rows'] != QUALIFYING_ROWS or metadata['qualifying_labels_read'] is not False
            or metadata['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or metadata['models_fitted'] != 0 or metadata['ridge_equations'] != 10 or metadata['ridge_penalty'] != 1000.
            or hashlib.sha256(wire.encoded(metadata)).hexdigest() != expected['admission_sha256']
            or metadata['source_sha256'] != prepared['declaration']['source_sha256']):
        raise ValueError('Exact complete original known-only source and wire declaration required')
    for path in code_paths():
        if metadata['source_sha256'].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('Original producer source changed after preparation')
    def verify_sources():
        if inputs.context.provider.original.verify_bindings(metadata['source_sha256'], guard) != prepared['records']:
            raise ValueError('Every original source identity must survive full handoff')
        return expected['admission_sha256']
    with inputs.context.open_query('probex', prepared['declaration'], prepared['records'], guard,
            retain_scalar=metadata['retain_scalar'], latent=metadata['arm'] == 'latent') as reader:
        return wire.send(reader, prepared['weights'], expected, output, verify_sources=verify_sources, guard=guard)
