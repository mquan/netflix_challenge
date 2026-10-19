#!/usr/bin/env python3
"""Recover retained NFPBAYS1 recipes and verify serialized posterior averages."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('user', 'item', 'user-bag', 'item-bag', 'day', 'item-bin', 'session', 'drift')


def posterior(path):
    """Read the version-one format written by frozen binary 0c1f29d."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        def read(fmt):
            length = struct.calcsize(fmt)
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError('truncated checkpoint')
            return struct.unpack(fmt, raw)
        magic, fingerprint, rank, sweep, kept = read('<QQIII')
        if magic != 0x4E46504241595331 or not 0 < rank <= 1024 or not 0 < kept <= sweep:
            raise ValueError('unsupported or invalid Bayes checkpoint')
        groups = {}
        for name in GROUPS:
            mode, count = read('<IQ')
            if mode > 2:
                raise ValueError('invalid feature mode')
            groups[name] = dict(mode=mode, size=count)
        bag_power, drift_scale, intercept, precision = read('<ffdd')
        for group in groups.values():
            read('<dd')
            mode, count = group['mode'], group['size']
            expected = [count if mode else 0, count * rank if mode == 2 else 0,
                        rank if mode == 2 else 0, rank if mode == 2 else 0]
            for width, length in zip((4, 4, 8, 8), expected):
                stored, = read('<Q')
                if stored != length or stream.tell() + stored * width > size:
                    raise ValueError('invalid parameter vector')
                stream.seek(stored * width, 1)
        count, = read('<Q')
        if stream.tell() + count * 8 > size:
            raise ValueError('invalid posterior vector')
        summed = np.fromfile(stream, dtype='<f8', count=count)
        rng_length, = read('<Q')
        if rng_length > 2**20 or len(stream.read(rng_length)) != rng_length or stream.read(1):
            raise ValueError('invalid RNG state or trailing checkpoint data')
    pred = (summed / kept).astype('<f4')
    if not np.isfinite(pred).all():
        raise ValueError('nonfinite posterior predictions')
    return dict(rank=rank, sweep=sweep, kept=kept, fingerprint=fingerprint, groups=groups,
                bag_power=bag_power, drift_scale=drift_scale, intercept=intercept,
                precision=precision, prediction_count=count), pred


def main():
    inventory = json.loads((ROOT / 'experiments/production-bank-inventory.json').read_text())
    bank = {Path(c['path']).name: c for c in inventory['columns']}
    binary = ROOT / 'artifacts/r8/bin/netflix-bayes-0c1f29d'
    binary_sha = digest(binary)
    recipes = []
    for name, budget in [('B2-full-r64', 128), ('B3-full-r128', 160), ('B4-full-r128', 160)]:
        column = bank[f'{name}.s{budget:04d}.f32']
        checkpoint = ROOT / 'artifacts/bayes' / (name + '.bin')
        info, pred = posterior(checkpoint)
        assert info['sweep'] == budget and info['kept'] == budget - 16
        assert digest(column['path']) == column['sha256']
        assert hashlib.sha256(pred.tobytes()).hexdigest() == column['sha256']
        record_path = ROOT / 'experiments' / (name + '.json')
        initial = json.loads(record_path.read_text())
        assert initial['binary_sha256'] == binary_sha
        cmd = list(initial['command'])
        cmd[0] = '{BINARY}'
        for flag, placeholder in [('--data', '{DATASET}'), ('--model', '{MODEL}'), ('--predictions', '{PREDICTIONS}')]:
            cmd[cmd.index(flag) + 1] = placeholder
        stages = [dict(command_template=cmd)]
        records = {str(record_path): digest(record_path)}
        if name.startswith('B4'):
            resume_path = ROOT / 'experiments/B4r-full-r128.json'
            resume = json.loads(resume_path.read_text())
            assert resume['binary_sha256'] == binary_sha and resume['exit_code'] == 0
            seed_path = ROOT / 'artifacts/bayes/B4-full-r128.sweep32.bin'
            seed_sha = resume['inputs']['artifacts/bayes/B4-full-r128.sweep32.bin']['sha256']
            assert digest(seed_path) == seed_sha
            seed_info, _ = posterior(seed_path)
            assert seed_info['sweep'] == 32 and seed_info['kept'] == 16
            assert seed_info['groups'] == info['groups'] and seed_info['rank'] == info['rank']
            cmd[cmd.index('--samples') + 1] = '32'
            cmd[cmd.index('--model') + 1] = '{MODEL_SWEEP32}'
            resume_cmd = list(resume['command'])
            resume_cmd[0] = '{BINARY}'
            for flag, placeholder in [('--data', '{DATASET}'), ('--model', '{MODEL}'),
                                      ('--predictions', '{PREDICTIONS}'), ('--init-model', '{MODEL_SWEEP32}')]:
                resume_cmd[resume_cmd.index(flag) + 1] = placeholder
            stages.append(dict(command_template=resume_cmd, historical_checkpoint_sha256=seed_sha))
            records[str(resume_path)] = digest(resume_path)
        recipes.append(dict(column=column['path'], column_sha256=column['sha256'],
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), checkpoint_metadata=info,
            posterior_mean_byte_identical=True, burn_in=16, stages=stages, records=records))
    report = dict(binary=str(binary), binary_sha256=binary_sha, recipes=recipes,
        source_sha256=digest(__file__),
        limitation='Stored checkpoint posterior means reproduce retained columns exactly. This is not a full retraining replay. Production must initialize on its own dataset; B4 stage two must use the newly produced sweep-32 checkpoint, never the historical development checkpoint. Native query RMSE diagnostics on zero sentinels are not scores.')
    path = ROOT / 'experiments/production-bayes-recipes.json'
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(verified_posterior_columns=len(recipes), binary_sha256=binary_sha)))


if __name__ == '__main__':
    main()
