#!/usr/bin/env python3
"""Recover joint/integrated checkpoint budgets and verify retained prediction aliases."""
import argparse
import datetime
import json
from pathlib import Path
import re
import struct
import subprocess

from recover_rbm_recipes import argument, replace
from train_holdout import digest

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    'J1-joint64', 'J2-joint128-reg02', 'J3-joint128-reg005', 'J4-joint128-day128',
    'J6-joint256-reg01', 'T2-f64-lr005-reg015-d97', 'T3-f64-lr008-reg008',
    'T4-f64-lr012-reg015-d93', 'T6-f256-lr012-reg015-d93', 'T6R-f256-resume',
    'I2-integrated-f64-koren', 'I2R-integrated-f64-resume',
)
LEGACY_HASH_MISMATCHES = set(STAGES[:4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay-dir', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/netflix')
    args = parser.parse_args()
    if args.output.exists() or (args.replay_dir and args.replay_dir.exists()):
        parser.error('choose new output paths')
    inventory = json.loads((ROOT / 'experiments/production-bank-inventory.json').read_text())
    bank = {}
    for column in inventory['columns']:
        bank.setdefault(column['sha256'], []).append(column)
    stages, by_model = [], {}
    data = ROOT / 'data/processed/netflix.bin'
    data_sha = digest(data)
    for name in STAGES:
        record_path = ROOT / 'experiments' / (name + '.json')
        record = json.loads(record_path.read_text())
        command = record['command']
        family = command[1]
        assert family in ('joint', 'integrated')
        model = ROOT / argument(command, '--model')
        prediction = ROOT / argument(command, '--predictions')
        model_sha, prediction_sha = digest(model), digest(prediction)
        historical_checks = {}
        for field, expected in [('model_artifact', model_sha), ('predictions_artifact', prediction_sha)]:
            if record.get(field):
                matches = record[field]['sha256'] == expected
                historical_checks[field] = dict(recorded=record[field]['sha256'], current=expected, matches=matches)
                assert matches or name in LEGACY_HASH_MISMATCHES, (name, field)
        with model.open('rb') as stream:
            magic, fingerprint, rank, epoch = struct.unpack('<QQII', stream.read(24))
        assert magic in (0x4E46504A4F494E31, 0x4E46504A4F494E32, 0x4E46504A4F494E33), name
        assert rank == int(argument(command, '--factors')) and 0 < epoch <= int(argument(command, '--iterations')), name
        log = ROOT / 'experiments' / (name + '.log')
        lines = re.findall(r'^epoch=' + str(epoch) + r'\s.*$', log.read_text(), re.M)
        assert len(lines) == 1, (name, epoch)
        columns = bank.get(prediction_sha, [])
        assert columns, name
        for column in columns:
            assert digest(column['path']) == prediction_sha, column['path']
        recorded_inputs = record.get('inputs', {})
        for path, value in recorded_inputs.items():
            assert digest(ROOT / path) == value['sha256'], (name, path)
        init = argument(command, '--init-model')
        dependency = by_model[str((ROOT / init).resolve())] if init else None
        start_epoch = int(argument(command, '--start-epoch', 1))
        if init and '--start-epoch' in command:
            assert start_epoch == dependency['fixed_final_epoch'] + 1, name
        template = list(command)
        template[0] = '{BINARY_WITH_REQUIRED_SOURCE_AND_FIXED_EPOCH}'
        for flag, value in [('--data', '{PRODUCTION_DATASET}'),
                            ('--model', '{OUTPUT_DIRECTORY}/' + name + '.bin'),
                            ('--predictions', '{OUTPUT_DIRECTORY}/' + name + '.f32'),
                            ('--iterations', epoch)]:
            replace(template, flag, value)
        if '--patience' in template:
            replace(template, '--patience', 0)
        else:
            template.extend(['--patience', '0'])
        template.append('--fixed-epoch')
        if init:
            replace(template, '--init-model', '{OUTPUT_DIRECTORY}/' + dependency['name'] + '.bin')
        graph = argument(command, '--graph')
        sidecars = {}
        if graph:
            replace(template, '--graph', '{PRODUCTION_GRAPH64}')
            sidecars[str(ROOT / graph)] = digest(ROOT / graph)
            sidecars[str(model) + '.nbr'] = digest(str(model) + '.nbr')
            if init:
                sidecars[str(ROOT / init) + '.nbr'] = digest(str(ROOT / init) + '.nbr')
        stage = dict(name=name, family=family, record=str(record_path), record_sha256=digest(record_path),
                     log_sha256=digest(log), epoch_log_line=lines[0], model=str(model), model_sha256=model_sha,
                     historical_model_digest_present=bool(record.get('model_artifact')),
                     historical_prediction_digest_present=bool(record.get('predictions_artifact')),
                     historical_artifact_checks=historical_checks,
                     recipe_provenance=('historical_hash_mismatch_requires_retraining_check'
                                        if any(not c['matches'] for c in historical_checks.values())
                                        else 'recorded_command_and_current_checkpoint_budget'),
                     original_prediction=str(prediction), prediction_sha256=prediction_sha,
                     retained_columns=[c['path'] for c in columns], rank=rank, dataset_fingerprint=fingerprint,
                     fixed_final_epoch=epoch, start_epoch=start_epoch,
                     dependency=dependency['name'] if dependency else None,
                     historical_initial_model_digest_present=bool(init and init in recorded_inputs),
                     graph=str(ROOT / graph) if graph else None, graph_and_neighbor_sidecars=sidecars,
                     required_historical_sources=record['sources'], historical_binary_sha256=record.get('binary_sha256'),
                     production_command_template=template)
        stages.append(stage)
        by_model[str(model.resolve())] = stage
    assert len(stages) == 12 and sum(len(s['retained_columns']) for s in stages) == 13
    replay = []
    if args.replay_dir:
        args.replay_dir.mkdir(parents=True)
        binary_sha = digest(args.binary)
        for stage in stages:
            prefix = args.replay_dir / stage['name']
            prediction = prefix.with_suffix('.f32')
            command = [str(args.binary.resolve()), 'eval-' + stage['family'], '--data', str(data),
                       '--model', stage['model'], '--predictions', str(prediction), '--threads', '1']
            if stage['graph']:
                command.extend(['--graph', stage['graph']])
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
            result.check_returncode()
            actual = digest(prediction)
            evidence = dict(stage=stage['name'], command=command, binary_sha256=binary_sha,
                            prediction_sha256=actual, expected_sha256=stage['prediction_sha256'],
                            retained_columns=stage['retained_columns'], byte_identical=actual == stage['prediction_sha256'])
            replay.append(evidence)
            print(json.dumps(evidence), flush=True)
            assert evidence['byte_identical'], stage['name']
        assert digest(args.binary) == binary_sha
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  data_sha256=data_sha, stages=stages, inference_replays=replay,
                  limitations=[
                      'Replays verify retained checkpoints and aliases, not full production retraining or score improvement.',
                      'J4 deliberately restarts epoch numbering at 1 when adding day factors to J3; preserve this warm-start boundary.',
                      'Integrated models require both the .bin and .bin.nbr files and a newly built production graph.',
                      'Current sidecar hashes and inference identity do not create missing historical sidecar digests.',
                      'Historical source hashes are recorded; matching fixed-epoch training builds remain to be verified.',
                      'The first four legacy models differ from their original producer hashes. Their commands are candidates pending retraining checks, even if current checkpoint inference matches the bank.',
                      'Every production predecessor must be retrained on production data, never reused from development.',
                  ])
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
