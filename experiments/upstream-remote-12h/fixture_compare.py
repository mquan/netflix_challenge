"""Post-freeze diagnostic: compare exported fresh ProbeX vectors with released fixtures, if present."""
import json, sys
from pathlib import Path
import numpy as np
job = sys.argv[1]
B = Path('artifacts/upstream-repro')
y = np.load(B/'fresh/data/probex/ratings.npy').astype(np.float64)
out = {}
for npz in sorted(Path('experiments/upstream-remote-12h/outputs').glob(f'{job}[._]*.npz')):
    stem = npz.name[:-4]
    ref = B/'reference/preds_new'/(stem+'.npy')
    if not ref.exists():
        out[stem] = 'no released fixture'; continue
    f = np.load(npz)['prediction'].astype(np.float64); r = np.load(ref).astype(np.float64)
    out[stem] = dict(fresh_rmse=float(np.sqrt(np.mean((f-y)**2))), released_rmse=float(np.sqrt(np.mean((r-y)**2))),
                     rms_diff=float(np.sqrt(np.mean((f-r)**2))), corr=float(np.corrcoef(f, r)[0, 1]))
out['note'] = 'Diagnostic after freezing fresh exports; released vectors never used as training inputs.'
Path(f'experiments/upstream-remote-12h/{job}/fixture-comparison.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=1))
