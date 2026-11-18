# Fresh-only nonlinear blend progress

Branch: `research/fresh-nonlinear-blend`. Started 2026-11-18 03:11 UTC
(November 17 Pacific). Runner PID 96583; sleep-prevention PID 96584.

The frozen control reproduced at **0.8662864618486446**. All 35 upstream
checks pass, including actual-backend outer-label poisoning and independent
verification checks. Input/code hashes and fixed recipes are in
`experiments/upstream-nonlinear/manifest.json`.

At 03:16 UTC both workers had completed fold 0 and advanced to fold 1:

| Fold 0 arm | RMSE | Change versus its matched control |
| --- | ---: | ---: |
| Contextual ridge control | 0.8657325558 | — |
| Three-seed tree correction | 0.8638341205 | -0.0018984353 |
| Three-seed neural correction | 0.8639569160 | -0.0017756398 |

These are partial development results, not overall five-fold scores. No recipe
changes follow these observations. All 226,746 fold-0 tree predictions replayed
bit for bit from saved model files. Full replay and independent scoring remain
required after both five-fold runs and the fixed equal average complete.

Live machine-readable status: `experiments/upstream-nonlinear/status.json`.
Family logs: `tree.log` and `nn.log` in that directory. Fold receipts are under
`tree/` and `nn/`. No downloaded prediction columns or qualifying outcomes enter
these fits. Base recipes have historical development-selection exposure.
