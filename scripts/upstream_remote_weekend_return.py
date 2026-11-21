#!/usr/bin/env python3
"""Stream full fresh weekend returns; never import a recovered target."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import numpy as np
import upstream_repro as u
import upstream_complete_trajectory_ops as ops
from upstream_remote15h_native_stream import Download

PLAN = u.ROOT/'experiments/upstream-remote-weekend/plan.json'
PREFIX = 'experiments/upstream-remote-weekend/execution/'
TAG = 'remote-weekend-20261120'
CHUNK = 2**20


def read(path): return json.loads(Path(path).read_text())


def digest(data): return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or not path.parts or path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name:
        raise ValueError('Exact safe relative archive path required')
    return path


def hashes(values):
    if not isinstance(values, dict) or not values: raise ValueError('Nonempty original hash map required')
    for name, sha in values.items():
        safe_name(name)
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Full original SHA256 required')
    return values


def job_spec(name):
    return next(j for j in read(PLAN)['jobs'] if j['name'] == name)


def native_shapes(job):
    name = job['name']
    shapes = {f'preds_3/{column}.{query}.npy': (rows,)
              for column in job['published_columns']
              for query, rows in [('probe_3', 1760494), ('qual', 2817131)]}
    shapes.update({f'preds_3/{name}.{split}.npy': (rows,)
                   for split, rows in [('train_3', 98720013), ('fulltrain', 100480507)]})
    if job['kind'] == 'rbm':
        width = 5*int(job['environment'].get('NFACT', job['environment']['H']))
        shapes.update({f'preds_3/{name}.ifeat.{split}.npy': (17770, width)
                       for split in ['train_3', 'fulltrain']})
    return shapes


def validate_metadata(job, blobs):
    """Bind membership to the frozen coordinator recipe, not a return's claims."""
    name = job['name']; plan = read(PLAN)
    index, proof, receipt, launch, transfer = [json.loads(blobs[key]) for key in
        ['archive-index.json', 'export-verification.json', name+'-receipt.json', name+'-launch.json', 'transfer.json']]
    if (index['state'] != 'complete_all_original_native_outputs_and_all_epoch_pairs_lossless_return'
        or index['job'] != name or index['plan_sha256'] != ops.sha(PLAN)
        or index['qualifying_labels_read'] is not False or index['raw_dataset_uploaded'] is not False
        or index['all_original_files_retained'] is not True
        or index['source_verification_sha256'] != digest(blobs['export-verification.json'])
        or proof['state'] != 'verified_complete_original_remote_weekend_producer'
        or proof['job'] != name or proof['source_pin'] != plan['source_pin']
        or any(proof[k] is not True for k in ['source_data_binary_rehashed', 'both_full_original_epoch_traces', 'independent_known_projection'])
        or any(proof[k] is not False for k in ['qualifying_labels_read', 'public_prediction_values_read'])
        or receipt['state'] != 'native_complete_pending_export' or receipt['job'] != name
        or receipt['log_sha256'] != digest(blobs[name+'.log'])
        or receipt['launch_sha256'] != digest(blobs[name+'-launch.json'])
        or launch['job'] != name or launch['plan_sha256'] != ops.sha(PLAN)
        or launch['command'] != job['command'] or launch['environment'] != job['environment']
        or transfer['job'] != name or transfer['release_tag'] != TAG or transfer['release_draft'] is not True
        or transfer['numeric_arrays_in_git'] is not False):
        raise ValueError('Exact original fresh recipe/run/export/transfer bindings required')
    trace = [int(x) for x in re.findall(rb'^Epoch\s+(\d+)\s*\|', blobs[name+'.log'], re.M)]
    if trace != list(range(1, job['epochs_per_phase']+1))*2:
        raise ValueError('Both complete original phase traces required')
    outputs = hashes(receipt['outputs']); shapes = native_shapes(job)
    if set(outputs) != set(shapes) | {f'preds_3/{name}.out'}:
        raise ValueError('All native epoch/terminal/training/factor outputs required')
    if outputs[f'preds_3/{name}.out'] != digest(blobs[name+'.log']):
        raise ValueError('Native output log differs from original run')
    if proof['all_native_output_sha256'] != {'native/'+p: sha for p, sha in outputs.items()}:
        raise ValueError('All native output hashes must bind original receipt')
    records = proof['records']
    if [r['column'] for r in records] != job['published_columns']:
        raise ValueError('Every original epoch and terminal compact pair required')
    required = {column+'.npz' for column in job['published_columns']}
    required.update(['plan.json', name+'-receipt.json', name+'-launch.json', name+'.log',
                     'preparation.json', 'build.log', 'split3.log', 'upstream_remote_weekend.py'])
    files = hashes(proof['files_sha256'])
    if set(files) != required: raise ValueError('Exact label-free export metadata and all compact pairs required')
    for row in records:
        if (row['file'] != row['column']+'.npz' or files[row['file']] != row['sha256']
            or row['native_probe_sha256'] != outputs[f"preds_3/{row['column']}.probe_3.npy"]
            or row['native_qual_sha256'] != outputs[f"preds_3/{row['column']}.qual.npy"]):
            raise ValueError('Compact pairs must bind original complete native queries')
    for filename in [name+'-receipt.json', name+'-launch.json', name+'.log']:
        if files[filename] != digest(blobs[filename]): raise ValueError('Pinned execution metadata differs')
    if files['plan.json'] != ops.sha(PLAN) or files['upstream_remote_weekend.py'] != plan['controller_sha256']['scripts/upstream_remote_weekend.py']:
        raise ValueError('Original source/controller plan must remain unchanged')
    source = hashes(proof['original_source_and_binary_sha256'])
    expected_source = {'source/'+p: sha for p, sha in plan['source_files_sha256'].items()}
    if set(source) != set(expected_source) | {'source/target/release/fit', 'source/target/release/split_3'}:
        raise ValueError('Exact original source files and both binaries required')
    if any(source[p] != sha for p, sha in expected_source.items()): raise ValueError('Pinned source differs')
    expected = {**files, **proof['all_native_output_sha256'], **source,
                'verification.json': digest(blobs['export-verification.json'])}
    hashes(expected)
    parts = index['parts']; combined = {}; part_names = []
    if not 1 <= len(parts) <= 32: raise ValueError('Bounded complete archive part list required')
    for number, part in enumerate(parts, 1):
        if (part['file'] != name+'-part%02d.tar.gz'%number or not 0 < part['bytes'] <= 1100*2**20
            or not re.fullmatch('[0-9a-f]{64}', part['sha256'])):
            raise ValueError('Exact ordered bounded archive parts required')
        part_names.append(part['file'])
        for key, sha in hashes(part['members_sha256']).items():
            if key in combined: raise ValueError('Duplicate cross-part member')
            combined[key] = sha
    if combined != expected: raise ValueError('Archive must cover every original export file exactly once')
    assets = transfer['assets']; named = {r['name']: r for r in assets}
    if len(named) != len(assets) or set(named) != set(part_names) | {name+'-index.json'}:
        raise ValueError('Exact transferred parts and prefixed index required')
    for part in parts:
        asset = named[part['file']]
        if asset['bytes'] != part['bytes'] or asset['sha256'] != part['sha256']:
            raise ValueError('Transfer and archive part bindings differ')
    asset = named[name+'-index.json']
    if asset['bytes'] != len(blobs['archive-index.json']) or asset['sha256'] != digest(blobs['archive-index.json']):
        raise ValueError('Uploaded archive index must match pinned index')
    return index, proof, expected


class HashedReader:
    def __init__(self, source):
        self.source = source; self.hash = hashlib.sha256(); self.count = 0

    def read(self, size):
        if not 0 <= size <= 4*CHUNK: raise ValueError('Bounded stream read required')
        payload = self.source.read(size); self.count += len(payload); self.hash.update(payload)
        return payload

    def drain(self):
        while self.read(CHUNK): pass


def copy_member(source, output, shape, size, check):
    tracked = HashedReader(source)
    if shape is not None:
        version = np.lib.format.read_magic(tracked)
        if version == (1, 0): header = np.lib.format.read_array_header_1_0(tracked)
        elif version == (2, 0): header = np.lib.format.read_array_header_2_0(tracked)
        else: raise ValueError('Original numeric NPY format required')
        actual_shape, fortran, dtype = header
        if actual_shape != shape or fortran or dtype != np.dtype('<f4') or tracked.count+4*math.prod(shape) != size:
            raise ValueError('Exact original finite float32 native shape required')
        # Materialized query NPY headers must also be retained byte-for-byte.
        # Header reconstruction would change Rust-produced original file bytes.
        if output is not None: raise ValueError('Use the tee reader for materialized native arrays')
    remaining = size-tracked.count
    while remaining:
        block = tracked.read(min(CHUNK, remaining))
        if not block: raise ValueError('Truncated original archive member')
        if shape is not None and (len(block)%4 or not np.isfinite(np.frombuffer(block, dtype='<f4')).all()):
            raise ValueError('Nonfinite or truncated native numerical output')
        if output: output.write(block)
        remaining -= len(block); check()
    if tracked.read(1): raise ValueError('Extra original archive member payload')
    return tracked.count, tracked.hash.hexdigest()


class Tee:
    def __init__(self, source, output): self.source, self.output = source, output
    def read(self, size):
        value = self.source.read(size); self.output.write(value); return value


def extract_part(source, part, job, destination, expected, check):
    """Verify compressed bytes, decoded membership, every payload and all native values."""
    tracked = HashedReader(source); seen = {}; payload_bytes = 0
    shapes = {'native/'+p: shape for p, shape in native_shapes(job).items()}
    with gzip.GzipFile(fileobj=tracked, mode='rb') as decoded:
        with tarfile.open(fileobj=decoded, mode='r|', ignore_zeros=True) as bundle:
            for member in bundle:
                path = safe_name(member.name)
                if len(path.parts) < 2 or path.parts[0] != job['name']:
                    raise ValueError('Exact producer-prefixed member required')
                relative = str(PurePosixPath(*path.parts[1:]))
                if relative not in part['members_sha256'] or relative in seen or not member.isfile():
                    raise ValueError('Exact unique regular archive members required')
                payload_bytes += member.size
                if not 0 <= member.size <= 2**30 or payload_bytes > 2**30:
                    raise ValueError('Original archive uncompressed part budget exceeded')
                selected = not relative.startswith('native/') or relative.endswith(('.probe_3.npy', '.qual.npy'))
                output = None
                if selected:
                    if member.size > 128*2**20: raise ValueError('Bounded query/source/metadata materialization required')
                    target = destination.joinpath(*PurePosixPath(relative).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.resolve().is_relative_to(destination.resolve()): raise ValueError('Owned materialization only')
                    output = target.open('xb')
                try:
                    stream = bundle.extractfile(member)
                    with stream:
                        incoming = Tee(stream, output) if output else stream
                        count, sha = copy_member(incoming, None, shapes.get(relative), member.size, check)
                    if sha != part['members_sha256'][relative] or sha != expected[relative]:
                        raise ValueError('Every decoded archive payload must match its original hash')
                    if output: output.flush()
                finally:
                    if output: output.close()
                seen[relative] = dict(bytes=count, sha256=sha, materialized=selected,
                                      native_shape=list(shapes[relative]) if relative in shapes else None,
                                      native_finite_checked=relative in shapes)
        # Validate the entire gzip trailer/CRC, including trailing tar padding.
        while decoded.read(CHUNK): check()
    tracked.drain()
    if (set(seen) != set(part['members_sha256']) or tracked.count != part['bytes']
        or tracked.hash.hexdigest() != part['sha256']):
        raise ValueError('Complete compressed part identity/membership required')
    return seen


def metadata(commit, job):
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Pinned full remote commit required')
    names = ['archive-index.json', 'export-verification.json', job+'-receipt.json', job+'-launch.json', job+'.log', 'transfer.json']
    return {name: subprocess.check_output(['git','show',commit+':'+PREFIX+job+'/'+name]) for name in names}


def transport(job, commit, report, destination, check):
    if report.exists() or destination.exists(): raise ValueError('Original full or partial return retained; no automatic retry')
    blobs = metadata(commit, job['name']); index, proof, expected = validate_metadata(job, blobs)
    release = json.loads(subprocess.check_output(['gh','release','view',TAG,'--repo','mquan/netflix_challenge','--json','isDraft,assets']))
    if release['isDraft'] is not True: raise ValueError('Transfer release must remain draft')
    assets = {asset['name']: asset for asset in release['assets']}
    transfer = json.loads(blobs['transfer.json'])
    for row in transfer['assets']:
        asset = assets[row['name']]
        if asset['size'] != row['bytes'] or asset.get('digest') != 'sha256:'+row['sha256']:
            raise ValueError('Current own release assets must match full original size/SHA')
    # Retain compact pairs, complete native query arrays, all original code and metadata.
    reserve = len(job['published_columns'])*40*2**20+128*2**20
    if shutil.disk_usage(u.ROOT).free < 18.75*2**30+reserve:
        raise RuntimeError('Complete return storage must fit above unchanged disk floor')
    check(); report.mkdir(parents=True); destination.mkdir(parents=True)
    for name, payload in blobs.items(): (report/name).write_bytes(payload)
    bindings = {str(path.relative_to(u.ROOT)): ops.sha(path) for path in
                [PLAN, Path(__file__), u.ROOT/'scripts/upstream_remote15h_native_stream.py',
                 u.ROOT/'scripts/upstream_remote_weekend.py', u.ROOT/'scripts/upstream_complete_trajectory_ops.py']}
    u.write(report/'plan.json', dict(state='declared_full_weekend_archive_import', job=job['name'], remote_commit=commit,
        source_sha256=bindings, metadata_sha256={name:digest(payload) for name,payload in blobs.items()},
        output_reserve_bytes=reserve, qualifying_labels_read=False, numerical_public_features_read=False))
    seen = {}; parts = []
    with (report/'downloads.log').open('x') as log:
        for part in index['parts']:
            stream = Download(assets[part['file']]['apiUrl'], log)
            try: observed = extract_part(stream, part, job, destination, expected, check)
            finally: stream.close()
            if set(seen)&set(observed): raise ValueError('Duplicate cross-part materialization')
            seen.update(observed); parts.append(dict(file=part['file'], bytes=part['bytes'], sha256=part['sha256']))
            u.write(report/'transport-progress.json', dict(verified_parts=parts, members=seen))
    if set(seen) != set(expected): raise ValueError('All original native/export files required')
    for name, sha in bindings.items():
        if ops.sha(u.ROOT/name) != sha: raise ValueError('Frozen import source changed')
    check()
    u.write(report/'transport-verification.json', dict(state='complete_all_weekend_archive_bytes_members_and_native_finiteness',
        plan_sha256=ops.sha(report/'plan.json'), parts=parts, members=seen,
        all_training_and_factor_outputs_retained_in_original_remote_archives=True,
        qualifying_labels_read=False, independent_projection_and_source_admission_remaining=True))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True); parser.add_argument('--remote-commit', required=True)
    args = parser.parse_args()
    # Heavy processing is only available through the separately reviewed supervisor.
    from upstream_remote_weekend_return_guard import Guard
    guard = Guard(args.job); report = u.ROOT/'experiments/upstream-remote-weekend-return'/args.job
    transport(job_spec(args.job), args.remote_commit, report, u.ROOT/'artifacts/upstream-remote-weekend-return'/args.job, guard.check)


if __name__ == '__main__': main()
