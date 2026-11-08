# Production retrain: schedule review (Claude, 2026-11-08)

Read-only review requested by the user after Codex's ETA of Nov 15–19.
No Codex process, plan, guard or frozen file was changed. Recommendations
below need Codex (owner) and the user to adopt them.

## Finding

The remaining production work is about **61 h on the queue order as frozen**, plus
1–2 h of guard overhead and roughly 1–3 h of final fit, two inferences and export.
That lands about **Nov 11**, not Nov 15–19, if nothing fails. The critical path is

    CPU-neural (4 h left) → RBM (22 h) → Bayes (33–36 h) → kernel (0.2 h) → user-kNN (1 h)

and almost all of that chain is **ordering guards, not data dependencies**:

| Guard | Real input needed? |
|---|---|
| RBM waits for all 7 CPU-neural audits | No: no RBM stage reads a neural output |
| Bayes waits for all 12 RBM audits + CPU-neural exit | No |
| Kernel waits for 4 Bayes + 28 kNN audits | Only the J3 joint model |
| User-kNN waits for 4 kernel audits | No: K1 trains from scratch |
| Joint waits for all 4 DEPS audits | Only I2/I2R read graph64 |
| AutoRec waits for all 7 GPU audits | No (GPU sharing only) |

Real data chains: B4 stage0 → stage1 (16–19 h, the longest), the RBM resume
chains (R3 → R3R → R4C/R4D ≈ 6–8 h; P3 → P3C, P4 → P4C ≈ 5–6 h each), and
DEPS → joint → kNN → kernel (≈ 7 h).

## Remaining work by family (historical runtimes of the same recipes)

| Family | Left (h, serial) | Threads | Evidence |
|---|---|---|---|
| GPU (A2f…MT9) + AutoRec | ≈ 17 + 2.5 (upper ≈ 35) | MPS + 1 | production A2h/A2f logs; `artifacts/masked/*-refit.json` |
| CPU-neural (MT1…MT6) | ≈ 4 | 2–3 | production progress; historical refit reports |
| RBM (12 stages) | ≈ 22 | 4–8 | `experiments/<name>.log` per-epoch seconds |
| Bayes B2, B3, B4×2 | ≈ 33–36 | 4–5 | `experiments/B{2,3,4,4r}-full-*.log` |
| DEPS + joint + kNN + kernel + user-kNN | ≈ 8 (upper ≈ 13) | 1 (K1: 6) | PR5–PR12 full-data replays |

Total remaining compute is about 300 thread-hours. With about 12 usable
threads on this 14-core machine (Docker and other user processes take
about 2), the **throughput floor is about 25–30 h**.

## Options (biggest first)

1. **Run guards on real data dependencies plus a thread budget, not family
   order.** Start the B4 chain and B3 now. Run RBM chains in parallel with
   Bayes once S0 finishes (about 1–3 h from now), with B2 and the
   single-thread lane filling gaps. Keep every stage's frozen recipe, thread
   count and audit, so outputs are unaffected; only start times change.
   Estimate **≈ 30–36 h** to the final fit (≈ Nov 10), versus ≈ 61 h.
   Risk: CPU contention stretches individual stages, which is why this isn't 25 h.
2. **Optional, costs score: skip B4 and substitute a duplicate of B2.** S0's
   audited `quality-bayes` arm already measures this exact swap: +0.000106
   dev CV (0.8669768 vs 0.8668706), about 2.6% of the 0.0041 gap. It removes
   the 16–19 h B4 chain, leaving ≈ 20–25 h, co-limited by the GPU lane. Not
   recommended while the goal is score; listed for a deadline.
3. **Not worth reopening plans for:** the duplicate "replay" runs (kNN,
   kernel, DEPS, user-kNN: ≈ 0.6 h in total) and per-stage checksums of 279
   inputs (minutes). The GPU lane is not critical. The `pad_row_tokens` fix
   from the moonshot (A2-family memory ≈ 60 GB → 7 GB) would free unified
   memory for CPU overlap, but it changes the frozen GPU recipe, so it isn't
   proposed.

## Uncertainty

Low-confidence estimates: B4 stage0 (3.8–6.9 h), joint J6–I2R (possibly 2×),
the AutoRec MPS parity check, and the final fit (no runtime record). A2h ran
1.9× its historical time while sharing the machine (including Claude's short
M0 pilots); A2f is at 1.0×.
