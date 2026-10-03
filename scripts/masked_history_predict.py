#!/usr/bin/env python3
"""Predict a dataset's probe rows with a saved masked-history checkpoint.

For the Stage C rehearsal: a selection checkpoint trained on a training-only
holdout dataset is scored on that holdout and on a dataset with identical
training arrays whose probe is the original probe (scripts/rehearsal_view.py).
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from gbdt_blend import load
from masked_history import Config, HistoryModel, digest, predict, write_f32


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if pathlib.Path(a.output).exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(a.threads)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    c = Config(**saved['config'])
    c.device = 'cpu'
    model = HistoryModel(saved['items'], c)
    model.load_state_dict(saved['state'])
    d = load(a.data)
    with torch.no_grad():
        pred = predict(model, d)
    write_f32(a.output, pred)
    y = d['probe']['rating'].astype(np.float64)
    report = dict(checkpoint=a.checkpoint, checkpoint_sha256=digest(a.checkpoint), epoch=saved['epoch'],
                  data=a.data, data_sha256=digest(a.data), output=a.output, output_sha256=digest(a.output),
                  rmse=float(np.sqrt(np.mean((np.clip(pred.astype(np.float64), 1, 5) - y)**2))))
    pathlib.Path(a.output + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
