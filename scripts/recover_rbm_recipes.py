#!/usr/bin/env python3
"""Recover retained RBM checkpoint budgets and resume chains; optionally replay inference."""
import argparse
import datetime
import json
from pathlib import Path
import re
import struct
import subprocess

from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    'R1-rbm100', 'R2-rbm100-fast', 'R3-rbm200-long', 'R3R-rbm200-resume',
    'P3-rbm200-seed7', 'P4-rbm300-wc001', 'R4D-rbm200-condbias-x3',
    'R4C-rbm200-condbias', 'P2-rbm200-condbias-scratch', 'R2C-rbm100-condbias',
    'P3C-rbm200-seed7-condbias', 'P4C-rbm300-wc001-condbias',
)


def argument(command, flag, default=None):
    return command[command.index(flag) + 1] if flag in command else default


def replace(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay-dir', type=Path)
    parser.add_argument('--replay-stage', action='append', help='replay only named retained stages; default all')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/netflix')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() or (args.replay_dir and args.replay_dir.exists()):
        parser.error('choose new output paths')
    if args.replay_stage and not args.replay_dir:
        parser.error('--replay-stage requires --replay-dir')
    inventory = json.loads((ROOT / 'experiments/production-bank-inventory.json').read_text())
    bank = {}
    for column in inventory['columns']:
        bank.setdefault(column['sha256'], []).append(column)
    models, stages, retained = {}, [], []
    for name in STAGES:
        record_path = ROOT / 'experiments' / (name + '.json')
        record = json.loads(record_path.read_text())
        command = record['command']
        model = ROOT / argument(command, '--model')
        prediction = ROOT / argument(command, '--predictions')
        model_sha = digest(model)
        if record.get('model_artifact'):
            assert model_sha == record['model_artifact']['sha256'], name
        with model.open('rb') as stream:
            magic, fingerprint, hidden, epoch = struct.unpack('<QQII', stream.read(24))
        assert magic in (0x4E4650524253314D, 0x4E4650524253324D), name
        assert hidden == int(argument(command, '--hidden')) and epoch > 0, name
        assert epoch <= int(argument(command, '--iterations')), name
        log = ROOT / 'experiments' / (name + '.log')
        matches = re.findall(r'^epoch=' + str(epoch) + r'\s.*$', log.read_text(), re.M)
        assert len(matches) == 1, (name, epoch)
        init = argument(command, '--init-model')
        dependency = None
        if init:
            dependency = models[str((ROOT / init).resolve())]
            assert digest(ROOT / init) == record['inputs'][init]['sha256'], name
            assert int(argument(command, '--start-epoch')) == dependency['fixed_final_epoch'] + 1, name
        template = list(command)
        template[0] = '{BINARY_WITH_REQUIRED_SOURCE_AND_FIXED_EPOCH}'
        replace(template, '--data', '{PRODUCTION_DATASET}')
        replace(template, '--model', '{OUTPUT_DIRECTORY}/' + name + '.bin')
        replace(template, '--predictions', '{OUTPUT_DIRECTORY}/' + name + '.f32')
        replace(template, '--iterations', epoch)
        replace(template, '--patience', 0)
        if init:
            replace(template, '--init-model', '{OUTPUT_DIRECTORY}/' + dependency['name'] + '.bin')
        if '--fixed-epoch' not in template:
            template.append('--fixed-epoch')
        stage = dict(name=name, record=str(record_path), record_sha256=digest(record_path),
                     log_sha256=digest(log), epoch_log_line=matches[0], model=str(model),
                     model_sha256=model_sha, historical_model_digest_present=bool(record.get('model_artifact')),
                     hidden=hidden, fixed_final_epoch=epoch, dataset_fingerprint=fingerprint,
                     start_epoch=int(argument(command, '--start-epoch', 1)),
                     dependency=dependency['name'] if dependency else None,
                     required_historical_sources=record['sources'],
                     historical_binary_sha256=record.get('binary_sha256'),
                     production_command_template=template)
        models[str(model.resolve())] = stage
        stages.append(stage)
        # Reference-bank files may have been copied to a different path.
        columns = bank.get(digest(prediction), [])
        for column in columns:
            assert digest(column['path']) == column['sha256'], name
            historical = record.get('predictions_artifact')
            if historical:
                assert historical['sha256'] == column['sha256'], name
            retained.append(dict(stage=name, column=column['path'], original_prediction=str(prediction), sha256=column['sha256'],
                                 historical_prediction_digest_present=bool(historical)))
    assert len(retained) == 9
    if args.replay_stage and not set(args.replay_stage) <= {c['stage'] for c in retained}:
        parser.error('unknown retained replay stage')
    replay = []
    if args.replay_dir:
        args.replay_dir.mkdir(parents=True)
        binary_sha = digest(args.binary)
        for column in retained:
            if args.replay_stage and column['stage'] not in args.replay_stage:
                continue
            stage = next(s for s in stages if s['name'] == column['stage'])
            prefix = args.replay_dir / stage['name']
            prediction = prefix.with_suffix('.f32')
            command = [str(args.binary.resolve()), 'eval-rbm', '--data',
                       str(ROOT / 'data/processed/netflix.bin'), '--model', stage['model'],
                       '--predictions', str(prediction), '--threads', '1']
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
            result.check_returncode()
            actual = digest(prediction)
            evidence = dict(stage=stage['name'], command=command, binary_sha256=binary_sha,
                            prediction_sha256=actual, expected_sha256=column['sha256'],
                            byte_identical=actual == column['sha256'])
            replay.append(evidence)
            print(json.dumps(evidence), flush=True)
            assert evidence['byte_identical'], stage['name']
        assert digest(args.binary) == binary_sha
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  stages=stages, retained_columns=retained, inference_replays=replay,
                  limitations=[
                      'Fixed budgets come from retained checkpoints, not original maximum iteration flags.',
                      'Resume boundaries restart momentum and user-order RNG; preserve each stage and its original thread count.',
                      'Production dependencies must be newly trained production checkpoints, never these development checkpoints.',
                      'Historical source hashes are recorded, but matching fixed-epoch builds and full retraining are not verified here.',
                      'Inference replays, when requested, verify checkpoint-to-column identity, not production retraining or hidden-test accuracy.',
                  ])
    destination.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
