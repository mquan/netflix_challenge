"""Refit the supplementary linear model in a temporary directory and compare its weights and predictions."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import own_schema_branch as branch
import own_schema_receipt as receipt
import own_stable_subspace as stable


def main():
    actual=stable.OUT
    reference=receipt.read(actual/'freeze.json')
    expected=reference['qualifying_hashes']['selected']
    assert receipt.streamed_sha(actual/'selected.f32')==expected
    with TemporaryDirectory(prefix='netflix-own-replay-') as temporary:
        trial=Path(temporary);stable.OUT=trial
        try:stable.main()
        finally:stable.OUT=actual
        replay=receipt.read(trial/'freeze.json')
        assert reference['selected']==replay['selected'] and reference['raw_inputs_sha256']==replay['raw_inputs_sha256']
        a=np.memmap(actual/'selected.f32','<f4','r');b=np.memmap(trial/'selected.f32','<f4','r')
        unequal=int(np.count_nonzero(a!=b));assert unequal==0 and replay['qualifying_hashes']['selected']==expected
        with np.load(actual/'fit.npz') as x,np.load(trial/'fit.npz') as y:
            assert x.files==y.files
            errors={k:float(np.max(np.abs(x[k]-y[k]))) for k in x.files}
        assert all(v<1e-10 for v in errors.values()),errors
        assert receipt.read(actual/'audit.json')==receipt.read(trial/'audit.json')
        result=dict(time=datetime.now().astimezone().isoformat(),rows=len(a),different_float32_predictions=unequal,
            sha256=expected,replayed_sha256=receipt.streamed_sha(trial/'selected.f32'),maximum_fit_array_differences=errors,
            held_user_metrics_identical=reference['known']==replay['known'],recovered_scores_identical=True,
            raw_forecast_manifest_sha256=reference['raw_inputs_sha256'],fit_implementation_sha256=receipt.streamed_sha(stable.__file__),
            verifier_sha256=receipt.streamed_sha(__file__))
        branch.write(actual/'independent-refit.json',result);print(result,flush=True)


if __name__=='__main__':main()
