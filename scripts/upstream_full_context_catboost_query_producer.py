"""Original full qualifying operator adapter; no selection, model fit or owner.

Runs only in the isolated Torch-capable source role. The future owner must
bind the complete selected known arm and its exact eligible weight bytes to
this separately authenticated full qualifying stream before authorizing use.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
import upstream_full_context_catboost_producer_v2 as known
import upstream_full_context_catboost_consumer as consumer

ROOT=known.ROOT
QUALIFYING_ROWS=2817131
FORMAT='own_full_context_tree_original_qualifying_source_v1'


def code_paths():
    return [Path(__file__)]


def verify_sources(bindings,guard):
    return known.inputs.context.provider.original.verify_bindings(bindings,guard)


def prepare(arm,guard):
    if arm not in ('scalar','latent') or not callable(guard):
        raise ValueError('Exact original full qualifying arm and whole-family guard required')
    original=known.prepare(arm,guard)
    metadata=original['metadata'];parent=known.inputs.data.REPORT/'manifest.json'
    manifest=consumer.verified_json(parent,metadata['parent_CPU_manifest_sha256'],ROOT,guard)
    if (manifest['qualifying_rows']!=QUALIFYING_ROWS or metadata['qualifying_rows']!=QUALIFYING_ROWS
            or manifest['arms'][arm]!=metadata['dimensions']
            or manifest['alias']['retain_scalar']!=metadata['retain_scalar']):
        raise ValueError('Every original full qualifying row and feature/context column required')
    operator=manifest['virtual_c_order_sha256']['qual']['arm_operator_sha256'][arm]
    if len(operator)!=2 or any(not known.wire.digest_ok(h) for h in operator):
        raise ValueError('Both full original qualifying operator hashes required')
    bindings=dict(metadata['source_sha256'])
    query_code={}
    for path in code_paths():
        if path.resolve()!=path or not path.is_relative_to(ROOT):
            raise ValueError('Original canonical full qualifying producer code required')
        relative=str(path.relative_to(ROOT));digest=known.sha(path)
        if relative in bindings and bindings[relative]!=digest:raise ValueError('Conflicting original source declaration')
        bindings[relative]=digest;query_code[relative]=digest
    if not query_code:raise ValueError('Full original qualifying source implementation binding required')
    records=verify_sources(bindings,guard)
    declaration=copy.deepcopy(original['declaration']);declaration['source_sha256']=bindings
    query=dict(format=FORMAT,query='qual',arm=arm,rows=QUALIFYING_ROWS,
        known_rows=metadata['rows'],dimensions=metadata['dimensions'],retain_scalar=metadata['retain_scalar'],
        original_known_source_metadata=copy.deepcopy(metadata),
        original_known_source_metadata_sha256=hashlib.sha256(known.wire.encoded(metadata)).hexdigest(),
        source_sha256=bindings,query_code_sha256=query_code,weights_sha256=metadata['weights_sha256'],
        operator_sha256=operator,models_fitted=0,qualifying_labels_read=False,
        selection_authorized=False,production_data_admitted=False)
    digest=hashlib.sha256(known.wire.encoded(query)).hexdigest()
    contract=known.wire.contract(QUALIFYING_ROWS,retain_scalar=query['retain_scalar'],latent=arm=='latent',
        weights_sha256=query['weights_sha256'],admission_sha256=digest,operator_sha256=operator)
    if verify_sources(bindings,guard)!=records:raise ValueError('Every full original source identity must survive qualifying preparation')
    return dict(metadata=query,contract=contract,weights=original['weights'],declaration=declaration,records=records)


def send(prepared,output,guard):
    if not callable(guard):raise ValueError('External original whole-family guard required')
    metadata=prepared['metadata'];expected=prepared['contract'];original=metadata['original_known_source_metadata']
    digest=hashlib.sha256(known.wire.encoded(metadata)).hexdigest()
    if (metadata['format']!=FORMAT or metadata['query']!='qual' or metadata['arm'] not in ('scalar','latent')
            or metadata['rows']!=QUALIFYING_ROWS or original['qualifying_rows']!=QUALIFYING_ROWS
            or metadata['arm']!=original['arm'] or metadata['known_rows']!=original['rows']
            or metadata['dimensions']!=original['dimensions'] or metadata['retain_scalar']!=original['retain_scalar']
            or metadata['models_fitted']!=0 or metadata['qualifying_labels_read'] is not False
            or metadata['selection_authorized'] is not False or metadata['production_data_admitted'] is not False
            or original['format']!='own_full_context_tree_original_source_v1'
            or original['qualifying_labels_read'] is not False or original['models_fitted']!=0
            or original['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or original['ridge_penalty']!=1000. or original['ridge_equations']!=10
            or metadata['weights_sha256']!=original['weights_sha256']
            or hashlib.sha256(known.wire.encoded(original)).hexdigest()!=metadata['original_known_source_metadata_sha256']
            or metadata['source_sha256']!=prepared['declaration']['source_sha256']
            or any(metadata['source_sha256'].get(name)!=value for name,value in original['source_sha256'].items())):
        raise ValueError('Complete original matching known/qualifying source and ten eligible ridge systems required')
    contract=known.wire.contract(QUALIFYING_ROWS,retain_scalar=metadata['retain_scalar'],latent=metadata['arm']=='latent',
        weights_sha256=metadata['weights_sha256'],admission_sha256=digest,operator_sha256=metadata['operator_sha256'])
    if known.wire.encoded(contract)!=known.wire.encoded(expected):raise ValueError('Exact original full qualifying stream contract required')
    current={str(p.relative_to(ROOT)):known.sha(p) for p in code_paths()}
    if not current or current!=metadata['query_code_sha256'] or any(metadata['source_sha256'].get(p)!=h for p,h in current.items()):
        raise ValueError('Full original qualifying producer implementation must remain unchanged')
    def check():
        if verify_sources(metadata['source_sha256'],guard)!=prepared['records']:
            raise ValueError('Every original source identity must survive full qualifying stream')
        return digest
    with known.inputs.context.open_query('qual',prepared['declaration'],prepared['records'],guard,
            retain_scalar=metadata['retain_scalar'],latent=metadata['arm']=='latent') as reader:
        return known.wire.send(reader,prepared['weights'],expected,output,verify_sources=check,guard=guard)
