#!/usr/bin/env python3
"""Hash whole own native archive streams; materialize queries and metadata only."""
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import numpy as np
import upstream_repro as u

REPORT = u.ROOT / 'experiments/upstream-remote15h-native-stream'
RUN = u.ROOT / 'artifacts/upstream-remote15h-native-stream'
PREFIX = 'experiments/upstream-remote15h/execution/'
ARCHIVE = 'native-archive-remote15h-assigned-native.json'
JOBS = ['s3-dnn24', 's3-attn32', 's3-rbm200e']
CHUNK = 4*2**20


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError('Exact safe declared native member path required')
    return path


class Parts:
    def __init__(self, parts, opener):
        self.parts = list(parts)
        self.opener = opener
        self.index = 0
        self.current = None
        self.hash = None
        self.count = 0
        self.verified = []

    def read(self, count):
        if not 0 <= count <= CHUNK:
            raise ValueError('Bounded archive stream reads required')
        result = bytearray()
        while len(result) < count and self.index < len(self.parts):
            name, expected = self.parts[self.index]
            if self.current is None:
                self.current = self.opener(name)
                self.hash = hashlib.sha256(); self.count = 0
            block = self.current.read(count-len(result))
            if block:
                self.hash.update(block); self.count += len(block); result.extend(block)
                if self.count > expected['bytes']:
                    raise ValueError('Native release part exceeds declared size')
            else:
                self.current.close(); self.current = None
                if self.count != expected['bytes'] or self.hash.hexdigest() != expected['sha256']:
                    raise ValueError('Complete native release part hash/length differs')
                self.verified.append(dict(name=name, bytes=self.count, sha256=self.hash.hexdigest()))
                self.index += 1
        return bytes(result)

    def drain(self):
        while self.read(CHUNK):
            pass
        if len(self.verified) != len(self.parts):
            raise ValueError('Every complete native archive part required')

    def close(self):
        if self.current is not None:
            self.current.close(); self.current = None


class Download:
    def __init__(self, url, log):
        if not url.startswith('https://api.github.com/repos/mquan/netflix_challenge/releases/assets/'):
            raise ValueError('Expected own repository release asset required')
        self.child = subprocess.Popen(['gh', 'api', url, '-H', 'Accept: application/octet-stream'],
            stdout=subprocess.PIPE, stderr=log)

    def read(self, count):
        return self.child.stdout.read(count)

    def close(self):
        self.child.stdout.close()
        if self.child.poll() is None:
            # EOF normal path: the pipe has no remaining payload.
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.child.terminate(); self.child.wait()
        if self.child.returncode:
            raise RuntimeError('Own native release download failed')


def extract(stream, members, selected, destination, disk_floor=18.75*2**30):
    seen, ignored = {}, []
    with tarfile.open(fileobj=stream, mode='r|gz') as archive:
        for member in archive:
            path = safe_name(member.name)
            if member.name not in members:
                original = str(path.parent / path.name[2:]) if path.name.startswith('._') else None
                if original in members and member.isfile() and member.size < 4096:
                    ignored.append(member.name)
                    continue
                raise ValueError('Undeclared native tar member')
            expected = members[member.name]
            if member.name in seen or not member.isfile() or member.size != expected['bytes']:
                raise ValueError('Exact unique regular native archive members required')
            source = archive.extractfile(member)
            output = None
            if member.name in selected:
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('xb')
            digest = hashlib.sha256(); count = 0
            try:
                while True:
                    block = source.read(CHUNK)
                    if not block:
                        break
                    digest.update(block); count += len(block)
                    if output:
                        output.write(block)
                    if shutil.disk_usage(destination).free < disk_floor:
                        raise RuntimeError('Original CF/native disk floor reached; preserve partial import')
            finally:
                source.close()
                if output:
                    output.close()
            if count != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Every complete plaintext native archive member must match')
            seen[member.name] = dict(bytes=count, sha256=digest.hexdigest(), materialized=member.name in selected)
    if set(seen) != set(members):
        raise ValueError('Complete declared native archive membership required')
    stream.drain()
    return seen, ignored


def main():
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing full or partial native import retained')
    revision = subprocess.check_output(['git', 'rev-parse', 'origin/research/remote15h-handoff'], text=True).strip()
    payload = subprocess.check_output(['git', 'show', revision+':'+PREFIX+ARCHIVE])
    manifest = json.loads(payload)
    if manifest['qualifying_labels_included'] is not False or len(manifest['parts']) != 2:
        raise ValueError('Own exact two-part query archive without recovered labels required')
    evidence = {}; bound_outputs = set()
    original_plan = u.read(u.ROOT/'experiments/upstream-remote15h/plan.json')
    for job in JOBS:
        for filename in [job+'-receipt.json', job+'-launch.json', 'preparation.json', 'verification.json', 'plan.json', job+'.log']:
            key = job+'/'+filename
            evidence[key] = subprocess.check_output(['git', 'show', revision+':'+PREFIX+job+'/'+filename])
        receipt = json.loads(evidence[job+'/'+job+'-receipt.json'])
        remote = json.loads(evidence[job+'/verification.json'])
        plan = json.loads(evidence[job+'/plan.json'])
        launch = json.loads(evidence[job+'/'+job+'-launch.json'])
        prep = json.loads(evidence[job+'/preparation.json'])
        declared = next(r for r in original_plan['jobs'] if r['name'] == job)
        if (plan != original_plan or remote['state'] != 'verified_complete_original_remote15h_producer'
                or remote['source_pin'] != u.PIN or remote['job'] != job
                or not remote['both_full_original_epoch_traces'] or not remote['source_data_binary_rehashed']
                or remote['qualifying_labels_read'] is not False or remote['public_prediction_values_read'] is not False
                or launch['command'] != declared['command'] or launch['environment'] != declared['environment']
                or receipt['launch_sha256'] != sha_bytes(evidence[job+'/'+job+'-launch.json'])
                or receipt['log_sha256'] != sha_bytes(evidence[job+'/'+job+'.log'])
                or prep['plan_sha256'] != sha_bytes(evidence[job+'/plan.json'])
                or launch['preparation_sha256'] != sha_bytes(evidence[job+'/preparation.json'])):
            raise ValueError('Original own source/recipe/full-run metadata differs')
        trace = [int(v) for v in re.findall(r'^Epoch\s+(\d+)\b', evidence[job+'/'+job+'.log'].decode(), re.M)]
        if trace != list(range(1, declared['epochs_per_phase']+1))*2:
            raise ValueError('Both original complete phase traces required')
        for name, digest in receipt['outputs'].items():
            if manifest['members']['source/'+name]['sha256'] != digest:
                raise ValueError('Every native output must retain its original producer receipt hash')
            bound_outputs.add('source/'+name)
        for filename in [job+'-receipt.json', job+'-launch.json', job+'.log']:
            if manifest['members'][filename]['sha256'] != sha_bytes(evidence[job+'/'+filename]):
                raise ValueError('Original producer metadata and native archive differ')
        for name, digest in prep['binary_sha256'].items():
            if manifest['members']['source/'+name]['sha256'] != digest:
                raise ValueError('Original producer executable and complete archive differ')
    if {name for name in manifest['members'] if name.startswith('source/preds_3/')} != bound_outputs:
        raise ValueError('Every native numerical member must belong to an original completed own producer')
    selected = {name for name in manifest['members'] if not name.startswith('source/preds_3/')
                or name.endswith('.probe_3.npy') or name.endswith('.qual.npy')}
    retained_bytes = sum(manifest['members'][name]['bytes'] for name in selected)
    checkpoint_plan = u.ROOT/'experiments/upstream-cf96-disk-resume/recovery-plan.json'
    checkpoint_bytes = max(r['checkpoint_bytes'] for r in u.read(checkpoint_plan)['initial_checkpoints'].values())
    checkpoint_reserve = 2*checkpoint_bytes+100*2**20
    disk_floor = 18.75*2**30+checkpoint_reserve
    if shutil.disk_usage(u.ROOT).free < disk_floor+retained_bytes:
        raise RuntimeError('Full query storage and original dual checkpoint staging reservation required')
    assets = json.loads(subprocess.check_output(['gh', 'release', 'view', manifest['release_tag'], '--json', 'assets']))['assets']
    assets = {a['name']:a for a in assets}
    for name, expected in manifest['parts'].items():
        if assets[name]['size'] != expected['bytes'] or assets[name]['digest'] != 'sha256:'+expected['sha256']:
            raise ValueError('Release asset identity and exact expected native archive part differ')
    REPORT.mkdir(); RUN.mkdir()
    (REPORT / ARCHIVE).write_bytes(payload)
    for name, blob in evidence.items():
        target = REPORT / 'remote' / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(blob)
    u.write(REPORT / 'plan.json', dict(state='declared_complete_native_archive_stream_before_import',
        remote_commit=revision, manifest_sha256=sha_bytes(payload), selected=sorted(selected), retained_bytes=retained_bytes,
        checkpoint_staging_reserve_bytes=checkpoint_reserve,continuous_disk_floor_bytes=disk_floor,
        source_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__), u.ROOT/'REMOTE15H_NATIVE_STREAM_PLAN.md',checkpoint_plan]},
        qualifying_labels_read=False, numerical_public_features_read=False))
    with (REPORT / 'downloads.log').open('x') as log:
        stream = Parts(sorted(manifest['parts'].items()), lambda name:Download(assets[name]['apiUrl'], log))
        try:
            seen, ignored = extract(stream, manifest['members'], selected, RUN, disk_floor)
        finally:
            stream.close()
    for name in selected:
        if name.endswith('.probe_3.npy') or name.endswith('.qual.npy'):
            array = np.load(RUN / name, mmap_mode='r')
            count = 1760494 if name.endswith('.probe_3.npy') else 2817131
            if array.shape != (count,) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('Complete finite native full query arrays required')
            del array
    for name, expected in u.read(REPORT/'plan.json')['source_sha256'].items():
        if u.sha(u.ROOT/name) != expected:
            raise ValueError('Original archive transport code or reservation source changed')
    u.write(REPORT / 'transport-verification.json', dict(state='complete_all_native_archive_part_and_member_hashes_with_materialized_queries',
        observed=dt.datetime.now(dt.timezone.utc).isoformat(), plan_sha256=u.sha(REPORT/'plan.json'),
        parts=stream.verified, members=seen, ignored_appledouble_metadata=ignored,
        recovered_labels_read=False, all_training_predictions_retained_in_original_remote_archive=True,
        separate_native_projection_and_source_admission_remaining=True))
    print('Complete native archive checked;', len(selected), 'query/metadata files retained', flush=True)


if __name__ == '__main__':
    main()
