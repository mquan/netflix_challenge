"""Bounded binary producer/consumer handoff for the planned CPU tree study.

No process is launched, source admitted, or model fitted here. The source
producer may import Torch; the receiver imports only NumPy. A future owned
consumer must verify the producer's actual zero exit through its supervisor
AFTER receiving EOF and BEFORE training. The returned arrays alone never
prove that exit, source ownership, or the original full resource reservation.
"""
import hashlib
import json
import struct
import numpy as np
from upstream_full_context_catboost_stream import dimensions

MAGIC = b'OWN-CB-STREAM-1\n'
BLOCK = 8192
CHUNK = 1048576
METADATA_LIMIT = 16384


def digest_ok(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def contract(rows, *, retain_scalar, latent, weights_sha256, admission_sha256, operator_sha256):
    design, nonlinear = dimensions(retain_scalar, latent)
    if (type(rows) is not int or rows <= 0 or not digest_ok(weights_sha256)
            or not digest_ok(admission_sha256) or len(operator_sha256) != 2
            or any(not digest_ok(h) for h in operator_sha256)):
        raise ValueError('Exact declared full row count and original source/operator digests required')
    return dict(format='own_full_context_tree_binary_stream_v1', rows=rows, width=nonlinear + 1,
                design_columns=design, retain_scalar=retain_scalar, latent=latent,
                weights_sha256=weights_sha256, admission_sha256=admission_sha256,
                operator_sha256=list(operator_sha256), block_rows=BLOCK,
                feature_dtype='<f4', baseline_dtype='<f8', ridge_populations=10,
                qualifying_labels_read=False)


def validate_contract(value):
    expected = contract(value['rows'], retain_scalar=value['retain_scalar'], latent=value['latent'],
        weights_sha256=value['weights_sha256'], admission_sha256=value['admission_sha256'],
        operator_sha256=value['operator_sha256'])
    if encoded(value) != encoded(expected):
        raise ValueError('Exact complete original wire schema required')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def write_all(output, data, guard):
    view = memoryview(data).cast('B')
    for lo in range(0, len(view), CHUNK):
        part = view[lo:lo + CHUNK]
        while len(part):
            guard(); count = output.write(part)
            if type(count) is not int or not 0 < count <= len(part):
                raise IOError('Complete original binary handoff write required')
            part = part[count:]


def read_into(source, destination, guard):
    view = memoryview(destination).cast('B')
    for lo in range(0, len(view), CHUNK):
        part = view[lo:lo + CHUNK]
        while len(part):
            guard(); count = source.readinto(part)
            if type(count) is not int or not 0 < count <= len(part):
                raise ValueError('Truncated original binary handoff')
            part = part[count:]


def write_metadata(output, value, guard):
    raw = encoded(value)
    if not 0 < len(raw) <= METADATA_LIMIT:
        raise ValueError('Bounded original handoff metadata required')
    write_all(output, struct.pack('!I', len(raw)), guard); write_all(output, raw, guard)


def read_metadata(source, guard):
    prefix = bytearray(4); read_into(source, prefix, guard)
    count, = struct.unpack('!I', prefix)
    if not 0 < count <= METADATA_LIMIT:
        raise ValueError('Bounded original handoff metadata required')
    raw = bytearray(count); read_into(source, raw, guard)
    value = json.loads(raw)
    if encoded(value) != raw:
        raise ValueError('Canonical original metadata without duplicate keys required')
    return value


def send(reader, weights, expected, output, *, verify_sources, guard):
    """Retain only bounded operator/output blocks in the source process."""
    validate_contract(expected)
    if (not callable(guard) or not callable(verify_sources)
            or len(reader.prediction) != expected['rows']
            or weights.shape != (5, 2, expected['design_columns']) or weights.dtype != np.dtype('<f8')
            or not np.isfinite(weights).all()
            or hashlib.sha256(weights.tobytes(order='C')).hexdigest() != expected['weights_sha256']):
        raise ValueError('Complete exact ten original eligible ridge systems required')
    guard()
    if verify_sources() != expected['admission_sha256']:
        raise ValueError('Actual original source admission required before sending')
    write_all(output, MAGIC, guard); write_metadata(output, expected, guard)
    zhash, fhash, rawhash = [hashlib.sha256() for _ in range(3)]
    bhashes = [hashlib.sha256() for _ in range(10)]
    for lo in range(0, expected['rows'], BLOCK):
        guard(); hi = min(lo + BLOCK, expected['rows']); z, f = reader.block(lo, hi)
        if (z.shape != (hi - lo, expected['design_columns']) or z.dtype != np.dtype('<f8')
                or f.shape != (hi - lo, expected['width'] - 1) or f.dtype != np.dtype('<f4')
                or not np.isfinite(z).all() or not np.isfinite(f).all()):
            raise ValueError('Every original finite operator column required')
        zhash.update(z.tobytes(order='C')); fhash.update(f.tobytes(order='C'))
        features = np.empty((hi - lo, expected['width']), dtype='<f4')
        features[:, :-1] = f; features[:, -1] = 0
        baselines = np.empty((5, 2, hi - lo), dtype='<f8')
        for fold in range(5):
            for population in range(2):
                guard()
                values = np.clip(z @ weights[fold, population], 1.05, 4.95)
                if not np.isfinite(values).all():
                    raise ValueError('Finite complete eligible ridge predictions required')
                baselines[fold, population] = values
                bhashes[2 * fold + population].update(values.tobytes())
        rawhash.update(features.tobytes())
        write_all(output, features, guard); write_all(output, baselines, guard)
        del features, baselines, z, f
    if [zhash.hexdigest(), fhash.hexdigest()] != expected['operator_sha256']:
        raise ValueError('Every original CPU operator stream must match before handoff completion')
    if hashlib.sha256(weights.tobytes(order='C')).hexdigest() != expected['weights_sha256']:
        raise ValueError('Every original eligible ridge coefficient must remain unchanged')
    if verify_sources() != expected['admission_sha256']:
        raise ValueError('All source identities must remain unchanged after streaming')
    footer = dict(raw_sha256=rawhash.hexdigest(), baseline_sha256=[h.hexdigest() for h in bhashes],
                  operator_sha256=expected['operator_sha256'], admission_sha256=expected['admission_sha256'])
    write_metadata(output, footer, guard); output.flush(); guard()
    return footer


def receive(source, expected, *, guard):
    """Allocate exactly one complete raw matrix in the isolated receiver.

    Returns only after exact footer and EOF. The supervisor must still prove
    the actual producer's successful exit; EOF alone is insufficient.
    """
    validate_contract(expected)
    if not callable(guard): raise ValueError('External whole-group guard required')
    magic = bytearray(len(MAGIC)); read_into(source, magic, guard)
    if magic != MAGIC or encoded(read_metadata(source, guard)) != encoded(expected):
        raise ValueError('Original frozen handoff declaration required before allocation')
    guard()
    features = np.empty((expected['rows'], expected['width']), dtype='<f4')
    baselines = np.empty((5, 2, expected['rows']), dtype='<f8')
    fhash, rawhash = hashlib.sha256(), hashlib.sha256()
    bhashes = [hashlib.sha256() for _ in range(10)]
    for lo in range(0, expected['rows'], BLOCK):
        hi = min(lo + BLOCK, expected['rows'])
        read_into(source, features[lo:hi], guard)
        packet = np.empty((5, 2, hi - lo), dtype='<f8'); read_into(source, packet, guard)
        if (not np.isfinite(features[lo:hi]).all() or np.any(features[lo:hi, -1] != 0)
                or not np.isfinite(packet).all() or np.any(packet < 1.05) or np.any(packet > 4.95)):
            raise ValueError('Finite original raw features, zero placeholder and clipped baselines required')
        baselines[:, :, lo:hi] = packet
        for fold in range(5):
            for population in range(2):
                bhashes[2 * fold + population].update(packet[fold, population].tobytes())
        fhash.update(features[lo:hi, :-1].tobytes(order='C')); rawhash.update(features[lo:hi].tobytes())
        del packet
    footer = read_metadata(source, guard)
    reconstructed = dict(raw_sha256=rawhash.hexdigest(), baseline_sha256=[h.hexdigest() for h in bhashes],
        operator_sha256=expected['operator_sha256'], admission_sha256=expected['admission_sha256'])
    if fhash.hexdigest() != expected['operator_sha256'][1] or encoded(footer) != encoded(reconstructed):
        raise ValueError('Every transferred original feature and baseline byte must match')
    guard()
    if source.read(1) != b'': raise ValueError('No trailing binary handoff bytes permitted')
    guard()
    proof = dict(contract_sha256=hashlib.sha256(encoded(expected)).hexdigest(), **reconstructed,
                 exact_footer_and_EOF=True, actual_producer_successful_exit_verified=False,
                 production_data_admitted=False, models_fitted=0, qualifying_labels_read=False)
    return features, baselines, proof
