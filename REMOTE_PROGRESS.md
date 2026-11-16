# Second-computer progress (M3 Max, 96 GB)

Auto-updated by `scripts/report_remote_progress.sh`. Last update: 2026-11-15 22:23 PST.

- State: **running**, device `mps`, arms ['combined']
- Completed: 2/10
- Active: combined-tree, baseline-enn-2026
- Runner heartbeat (UTC): 2026-11-16T06:23:17.853805+00:00

| Stage | Family | State | Elapsed |
|---|---|---|---|
| baseline-tree | tree | done | 0.39 h |
| baseline-nn | neural | done | 0.17 h |
| combined-tree | tree | running |  |
| combined-nn | neural | queued |  |
| baseline-enn-2026 | neural | running |  |
| baseline-enn-7 | neural | queued |  |
| baseline-enn-13 | neural | queued |  |
| combined-enn-2026 | neural | queued |  |
| combined-enn-7 | neural | queued |  |
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
    }
  ],
  "limitation": "Matched same-machine/backend original-Probe development comparisons; base checkpoints were historically Probe-selected; no qualifying result"
}
```
