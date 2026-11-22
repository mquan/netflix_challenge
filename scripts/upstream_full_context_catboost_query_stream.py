"""Bounded receiver for future full qualifying inference/replay; no admission.

Compatible with the original authenticated wire sender. Blocks are provisional
until the full footer, EOF and parent-observed successful producer exit have
been verified. This function neither publishes predictions nor starts models.
"""
import hashlib
import numpy as np
import upstream_full_context_catboost_wire as wire


def receive(source, expected, consume, *, guard):
    wire.validate_contract(expected)
    if not callable(consume) or not callable(guard):
        raise ValueError('External provisional block consumer and whole-family guard required')
    magic=bytearray(len(wire.MAGIC));wire.read_into(source,magic,guard)
    if magic!=wire.MAGIC or wire.encoded(wire.read_metadata(source,guard))!=wire.encoded(expected):
        raise ValueError('Exact full original frozen query contract required before allocation')
    fhash,rawhash=hashlib.sha256(),hashlib.sha256()
    bhashes=[hashlib.sha256() for _ in range(10)]
    blocks=0;rows=0;max_rows=0
    for lo in range(0,expected['rows'],wire.BLOCK):
        guard();hi=min(lo+wire.BLOCK,expected['rows']);size=hi-lo
        features=np.empty((size,expected['width']),dtype='<f4')
        baselines=np.empty((5,2,size),dtype='<f8')
        wire.read_into(source,features,guard);wire.read_into(source,baselines,guard)
        if (not np.isfinite(features).all() or np.any(features[:,-1]!=0)
                or not np.isfinite(baselines).all() or np.any(baselines<1.05) or np.any(baselines>4.95)):
            raise ValueError('Every finite original feature, zero placeholder and ten clipped baselines required')
        raw=features.tobytes();baseline=baselines.tobytes()
        fhash.update(features[:,:-1].tobytes(order='C'));rawhash.update(raw)
        for fold in range(5):
            for population in range(2):bhashes[fold*2+population].update(baselines[fold,population].tobytes())
        features.flags.writeable=False;baselines.flags.writeable=False
        consume(lo,hi,features,baselines)
        guard()
        if features.tobytes()!=raw or baselines.tobytes()!=baseline:
            raise ValueError('Provisional inference must not mutate original received query/operator bytes')
        rows+=size;blocks+=1;max_rows=max(max_rows,size)
        del features,baselines,raw,baseline
    footer=wire.read_metadata(source,guard)
    reconstructed=dict(raw_sha256=rawhash.hexdigest(),baseline_sha256=[h.hexdigest() for h in bhashes],
        operator_sha256=expected['operator_sha256'],admission_sha256=expected['admission_sha256'])
    if (rows!=expected['rows'] or fhash.hexdigest()!=expected['operator_sha256'][1]
            or wire.encoded(footer)!=wire.encoded(reconstructed)):
        raise ValueError('Every original full-query raw feature and baseline byte must match complete footer')
    guard()
    if source.read(1)!=b'':raise ValueError('No trailing original full-query stream bytes permitted')
    guard()
    return dict(contract_sha256=hashlib.sha256(wire.encoded(expected)).hexdigest(),**reconstructed,
        rows=rows,blocks=blocks,maximum_receiver_block_rows=max_rows,exact_footer_and_EOF=True,
        receiver_retains_no_full_raw_query_matrix=True,actual_producer_successful_exit_verified=False,
        prediction_publication_authorized=False,production_data_admitted=False,models_fitted=0,
        qualifying_labels_read=False)
