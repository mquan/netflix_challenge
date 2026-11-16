# Second-computer progress (M3 Max, 96 GB)

Auto-updated by `scripts/report_remote_progress.sh`. Last update: 2026-11-15 23:11 PST.

- State: **running**, device `mps`, arms ['combined']
- Completed: 8/10
- Active: combined-enn-7
- Runner heartbeat (UTC): 2026-11-16T07:11:33.380963+00:00

| Stage | Family | State | Elapsed |
|---|---|---|---|
| baseline-tree | tree | done | 0.39 h |
| baseline-nn | neural | done | 0.17 h |
| combined-tree | tree | done | 0.41 h |
| combined-nn | neural | done | 0.07 h |
| baseline-enn-2026 | neural | done | 0.27 h |
| baseline-enn-7 | neural | done | 0.19 h |
| baseline-enn-13 | neural | done | 0.14 h |
| combined-enn-2026 | neural | done | 0.12 h |
| combined-enn-7 | neural | running |  |
| combined-enn-13 | neural | queued |  |

## comparison.json

```json
{
  "results": [
    {
      "name": "baseline",
      "rmse": 0.8668672078045607,
      "delta_rmse": 0.0,
      "fold_delta": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "sha256": "f8ddc6281bdf7e0853ff5b8222d63149d8127c2c4cfa3ae46c5a9e60de3bca07"
    },
    {
      "name": "combined",
      "rmse": 0.8669229732749426,
      "delta_rmse": 5.576547038188551e-05,
      "fold_delta": [
        -2.8533923075446666e-06,
        8.778074013771064e-05,
        6.451255547990087e-06,
        0.00018257600250348727,
        4.690619361924497e-06
      ],
      "sha256": "e7ecb91ff55fd9296dea908e8abf5a198586e9d3855a64f55d2a932a5ea7f71a"
    },
    {
      "name": "baseline-ENN-three-seed",
      "rmse": 0.8672046100659176,
      "delta_rmse": 0.00033740226135692275,
      "fold_delta": [
        0.0003844879285829217,
        0.0003375927558650904,
        0.0003087323921361351,
        0.0003091661799833334,
        0.000347052718584262
      ],
      "sha256": "3899b88e2e3d75f846ff21c55303d0746deb407d9636b01bc3b0a241299ae5e9"
    }
  ],
  "limitation": "Matched same-machine/backend original-Probe development comparisons; base checkpoints were historically Probe-selected; no qualifying result"
}
```
