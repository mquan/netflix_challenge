#!/usr/bin/env python3
"""Restore our pinned local predictions from Git; optionally ingest raw data."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '18390f4bf778e305c940c62756226d1ea792778a'
ARCHIVE_URL = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz'


def restore():
    if subprocess.run(['git', 'cat-file', '-e', SOURCE+'^{commit}'], cwd=ROOT,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git', 'fetch', '--no-tags', 'origin', 'codex/household-personas'], cwd=ROOT, check=True)
    controls = ['-s3.json', '-s3.refit.f32', '-s3.ridge.f32', '-NN2.json',
                '-NN2.nn.f32', '-NN2.avg.f32', '-comparison.json']
    files = ['artifacts/r9/bank', 'artifacts/r4/meta-context.npz']
    files += ['artifacts/r9/R9-LAT-A2h-20261010'+s for s in controls]
    files += [f'artifacts/masked/A2h-refit.refit.e{e}.f32' for e in range(1, 13)]
    run = ROOT/'artifacts/independent-ensemble'
    process = subprocess.Popen(['git', 'archive', SOURCE, *files], cwd=ROOT, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                if name.startswith('artifacts/r9/bank/'):
                    target = run/'bank'/name.removeprefix('artifacts/r9/bank/')
                elif name.startswith('artifacts/r9/'):
                    target = run/'control'/Path(name).name
                elif name.startswith('artifacts/masked/'):
                    epoch = int(re.search(r'\.e(\d+)\.f32$', name).group(1))
                    target = run/'snapshots'/f'A2h-e{epoch:02}.f32'
                else:
                    target = ROOT/name
                target = target.resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError('Archive path outside checkout')
                if target.exists():
                    with archive.extractfile(member) as stream:
                        h = hashlib.sha256()
                        for block in iter(lambda: stream.read(1 << 20), b''):
                            h.update(block)
                    if hashlib.sha256(target.read_bytes()).hexdigest() != h.hexdigest():
                        raise ValueError(f'Existing input differs: {target}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest)
        if process.wait():
            raise RuntimeError('Pinned input archive failed')
    finally:
        if process.poll() is None:
            process.terminate()
    print('Restored pinned locally trained R9 bank, controls and twelve A2h snapshots.', flush=True)


def ingest():
    raw = ROOT/'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw/'nf_prize_dataset.tar.gz'
    subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--output', str(archive), ARCHIVE_URL], check=True)
    h = hashlib.md5()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    if h.hexdigest() != 'a8f23d2d76461211c6b4c0ca6df2547d':
        raise ValueError('Netflix archive checksum mismatch')
    extracted = raw/'extracted/download'
    training = extracted/'training_set/training_set'
    training.mkdir(parents=True, exist_ok=True)
    movies = 0
    with tarfile.open(archive, mode='r|gz') as outer:
        for member in outer:
            if not member.isfile():
                continue
            if Path(member.name).name in ['probe.txt', 'qualifying.txt']:
                with outer.extractfile(member) as f, (extracted/Path(member.name).name).open('wb') as dest:
                    shutil.copyfileobj(f, dest)
            elif Path(member.name).name == 'training_set.tar':
                with tarfile.open(fileobj=outer.extractfile(member), mode='r|') as inner:
                    for movie in inner:
                        if movie.isfile() and re.fullmatch(r'mv_\d{7}\.txt', Path(movie.name).name):
                            with inner.extractfile(movie) as f, (training/Path(movie.name).name).open('wb') as dest:
                                shutil.copyfileobj(f, dest)
                            movies += 1
    if movies != 17770:
        raise ValueError(f'Expected 17770 training files, got {movies}')
    subprocess.run(['make', 'build/netflix'], cwd=ROOT, check=True)
    (ROOT/'data/processed').mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT/'build/netflix'), 'ingest', '--training-dir', str(training),
                    '--probe', str(extracted/'probe.txt'), '--qualifying', str(extracted/'qualifying.txt'),
                    '--output', str(ROOT/'data/processed/netflix.bin')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, help='Existing original Netflix binary; link it into this checkout')
    parser.add_argument('--download-data', action='store_true', help='Download verified raw Netflix archive and ingest it')
    a = parser.parse_args()
    if a.dataset and a.download_data:
        parser.error('Choose an existing dataset or download, not both')
    restore()
    data = ROOT/'data/processed/netflix.bin'
    if a.dataset:
        source = a.dataset.expanduser().resolve(strict=True)
        data.parent.mkdir(parents=True, exist_ok=True)
        if not data.exists():
            data.symlink_to(source)
        elif data.resolve() != source:
            raise ValueError('A different dataset already exists; preserve it and use another checkout')
    if a.download_data:
        if data.exists():
            print('Dataset already exists; skipping download/ingestion.')
        else:
            ingest()
    if not data.exists():
        raise SystemExit('Inputs restored. Supply --dataset /path/to/netflix.bin or run again with --download-data.')
    import json
    reference = json.loads((ROOT/'experiments/LOCAL-ENSEMBLE-20261116-plan.json').read_text())
    expected, = [h for p, h in reference['inputs'].items() if p.endswith('/data/processed/netflix.bin')]
    h = hashlib.sha256()
    with data.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError('Dataset differs from the original R9 dataset; do not train against misaligned columns')
    subprocess.run([os.sys.executable, str(ROOT/'scripts/local_ensemble_run.py'), '--prepare'], cwd=ROOT, check=True)
    print('Ready. Follow REMOTE_ENSEMBLE.md to start the remote worker.')


if __name__ == '__main__':
    main()
