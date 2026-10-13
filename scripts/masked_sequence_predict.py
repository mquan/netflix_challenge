#!/usr/bin/env python3
"""Predict a dataset's probe rows with a saved masked-sequence (A2) checkpoint.

The sequence analogue of masked_history_predict.py, for the Stage C rehearsal.
--source is the scripts directory holding the trainer that wrote the checkpoint
(the A2 worktree); its masked_sequence.py and masked_history.py hashes are
recorded so the report can be matched against the training run's
source_sha256 / base_source_sha256.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if pathlib.Path(a.output).exists():
        raise FileExistsError(a.output)
    source = pathlib.Path(a.source).resolve()
    sys.path.insert(0, str(source))
    import dataclasses
    import masked_sequence as ms
    from gbdt_blend import load
    from masked_history import digest, write_f32
    torch.set_num_threads(a.threads)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    c = dataclasses.replace(ms.Config(**saved['config']), device=a.device)
    model = ms.SequenceModel(saved['items'], c, saved['weeks'])
    model.load_state_dict(saved['state'])
    model.to(a.device)
    d = load(a.data)
    if ms.weeks(d) != saved['weeks'] or d['items'] != saved['items']:
        raise ValueError('dataset calendar or item count differs from the checkpoint')
    pred = ms.predict(model, d)
    write_f32(a.output, pred)
    y = d['probe']['rating'].astype(np.float64)
    report = dict(checkpoint=a.checkpoint, checkpoint_sha256=digest(a.checkpoint), epoch=saved['epoch'],
                  source_sha256=digest(source / 'masked_sequence.py'),
                  base_source_sha256=digest(source / 'masked_history.py'),
                  data=a.data, data_sha256=digest(a.data), output=a.output, output_sha256=digest(a.output),
                  rmse=float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - y)**2))))
    pathlib.Path(a.output + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
