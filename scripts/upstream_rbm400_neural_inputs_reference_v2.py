"""Full verified context-data admission and one-matrix neural materialization.

No queue, model fitting or qualifying outcome access occurs here. These inputs
become usable only after the original owned CPU assembly and full replay.
"""
import hashlib
from pathlib import Path
import numpy as np
import upstream_rbm400_context_data_reference_v2 as data
import upstream_rbm400_context_inputs_reference_v2 as context
import upstream_rbm400_context_execution_reference_v2 as cpu
import verify_upstream_rbm400_context_data_reference_v2 as independent
import upstream_rbm400_latent_operators as operators

ROOT=context.ROOT


def sources(guard):
    declaration,records=context.source_declaration(guard)
    cpu.check_sources(cpu.read(cpu.REPORT/'plan.json'))
    done_path=cpu.REPORT/'completion.json';proof_path=data.REPORT/'independent-verification.json'
    manifest_path=data.REPORT/'manifest.json'
    done,proof,manifest=[cpu.read(p) for p in (done_path,proof_path,manifest_path)]
    if (done['state']!='complete_both_RBM400_context_arms_data_and_independent_known_ridge_replay'
            or done['verification_sha256']!=cpu.inputs.sha(proof_path) or done['qualifying_labels_read'] is not False
            or proof['state']!=independent.STATE or proof['manifest_sha256']!=cpu.inputs.sha(manifest_path)
            or proof['qualifying_labels_read'] is not False or proof['neural_models_fitted']!=0
            or proof['every_both_arm_operator_bit_verified'] is not True
            or proof['every_group_statistic_and_ridge_prediction_bit_verified'] is not True
            or proof['both_query_alias_independently_verified'] is not True
            or proof['source_sha256']!=manifest['source_sha256']
            or manifest['neural_models_fitted']!=0 or manifest['qualifying_labels_read'] is not False
            or manifest['nonlinear_component_OOF_predictions_used_as_features'] is not False
            or manifest['lambda_fixed']!=1000. or manifest['folds']!=5 or manifest['groups']!=10
            or set(manifest['prepared'])!=independent.NAMES
            or manifest['rows']!=1128164 or manifest['qualifying_rows']!=2817131):
        raise ValueError('Both complete original CPU data phases and independent full proof required')
    retain=manifest['alias']['retain_scalar']
    expected={arm:operators.dimensions(retain,arm=='latent') for arm in ('scalar','latent')}
    if manifest['arms']!=expected or proof['arms']!=expected:raise ValueError('Exact complete paired-alias arm dimensions required')
    if (proof['ridge_equations']!=5*expected['scalar']['design_columns']
            or proof['rows']!=1128164 or proof['qualifying_rows']!=2817131):
        raise ValueError('Every original known-ridge equation and both complete query populations required')
    bindings=dict(manifest['source_sha256'])
    for name,digest in declaration['source_sha256'].items():context.bind(bindings,ROOT/name,digest)
    for name,digest in proof['source_sha256'].items():context.bind(bindings,ROOT/name,digest)
    for name,row in manifest['prepared'].items():
        if Path(row['path'])!=data.ART/name:raise ValueError('Owned complete prepared data paths required')
        context.bind(bindings,data.ART/name,row['sha256'])
    for name,digest in cpu.read(cpu.REPORT/'plan.json')['source_sha256'].items():context.bind(bindings,ROOT/name,digest)
    for path in (done_path,proof_path,manifest_path,cpu.REPORT/'plan.json',Path(__file__)):
        context.bind(bindings,path)
    full_records=context.provider.original.verify_bindings(bindings,guard)
    # Full declaration is used by open_query's unconditional source/identity
    # checks. It also binds the CPU outputs used to construct every neural row.
    declaration['source_sha256']=bindings
    return declaration,manifest,full_records


def materialize(reader,weights,expected_streams,*,retain_scalar,latent,guard):
    dims=operators.dimensions(retain_scalar,latent)
    if (weights.shape!=(dims['design_columns'],) or weights.dtype!=np.float64
            or not np.isfinite(weights).all() or len(expected_streams)!=2 or not callable(guard)):
        raise ValueError('Complete finite original eligible ridge weights and operator hashes required')
    guard();count=len(reader.prediction)
    features=np.empty((count,dims['nn_inputs']),dtype='<f4');baseline=np.empty(count,dtype=np.float64)
    digests=[hashlib.sha256(),hashlib.sha256()];raw_digest=hashlib.sha256()
    for start in range(0,count,8192):
        guard();stop=min(start+8192,count);z,f=reader.block(start,stop)
        if (z.shape!=(stop-start,dims['design_columns']) or f.shape!=(stop-start,dims['nonlinear_columns'])
                or z.dtype!=np.float64 or f.dtype!=np.dtype('<f4') or not np.isfinite(z).all() or not np.isfinite(f).all()):
            raise ValueError('Complete original full-width operators required')
        digests[0].update(z.tobytes());digests[1].update(f.tobytes())
        baseline[start:stop]=np.clip(z@weights,1.05,4.95)
        features[start:stop,:-1]=f;features[start:stop,-1]=baseline[start:stop]
        raw_digest.update(features[start:stop].tobytes())
    if [h.hexdigest() for h in digests]!=expected_streams:
        raise ValueError('Every original full neural operator bit must match the complete CPU proof')
    guard()
    return features,baseline,dict(operator_sha256=expected_streams,raw_neural_input_sha256=raw_digest.hexdigest(),
                                  rows=count,width=dims['nn_inputs'],baseline_stored_in_input_as_float32=True)


def originals(arm,weights,declaration,manifest,records,guard):
    if arm not in ('scalar','latent'):raise ValueError('Exact predeclared neural arm required')
    keep=manifest['alias']['retain_scalar'];latent=arm=='latent'
    with context.open_query('probex',declaration,records,guard,retain_scalar=keep,latent=latent) as reader:
        return materialize(reader,weights,manifest['virtual_c_order_sha256']['probex']['arm_operator_sha256'][arm],
                           retain_scalar=keep,latent=latent,guard=guard)
