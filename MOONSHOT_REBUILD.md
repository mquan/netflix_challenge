# Moonshot: one large sequence model ("M")

Started 2026-10-31 at the user's request ("Let's go for the moonshot"). Owner:
Claude. The production retrain (STRUCTURAL_PLAN lane 4) is assigned to
**Codex** by the user and is not part of this effort.

## Honest framing

Reaching quiz 0.8554 (≈0.8628 dev CV, gap 0.0041) this way is unlikely:
- A2-family standalone gains have converted into blend gain at about 0.028 per unit.
- The headroom map found no untapped information source.
- The best published single models are about 0.879 probe-equivalent.

This is a deliberate high-variance bet. Its success condition is a model whose
errors differ *structurally* from the bank, not just a lower standalone RMSE.

## Design

Start from the strongest existing sequence model: N1, which is A2h plus
unrated-choice tokens (choice_sequence.py in the honest-residual-chains
worktree). Copy it into a new trainer `scripts/moonshot_sequence.py`;
Codex's and SO1's frozen files are not edited. Then change four things:

1. **Full history.** Lift the 1536-token cap with query-chunked attention, so
   the attention matrix never exceeds MPS's INT_MAX limit (the A2g failure),
   plus length-bucketed batches with a token budget. Only 5.3% of probe rows
   change context at 3072, so this is a necessary step, not the bet itself.
2. **Scale with a re-tuned optimiser.** d_model 128 → 256 and layers 4 → 6,
   with warmup plus cosine decay and the learning rate and dropout re-tuned per
   size on short P2 pilots. The A2c test changed size without re-tuning and
   compared at early epochs, so it was not a scaling study.
3. **A wider user code.** Today a 64-dimensional tanh code feeds a dot-product
   head. Replace it with a two-layer MLP head over (user code, item embedding,
   date features), so user–item interactions need not be bilinear. This is
   the change most likely to alter error *structure*.
4. **Distribution loss:** MSE plus λ·CE, at whatever λ E3 measures on P2 (or
   MSE only if E3 shows no gain).

## Stages and gates (fixed before any score)

| Stage | Work | Gate |
|---|---|---|
| M0 | Engineering on CPU and tiny fixtures: chunked attention equal to full attention, plus throughput and memory pilots | Exact or near equality (≤1e-5) on fixtures; a memory plan that fits 96 GB |
| M1 | P2 pilot at d128 L4, full history, choice tokens, MLP head | **Standalone P2 ≥0.003 better than N1**, and **≥0.0010 beyond bank11 + N1** (shared P2 gate) |
| M2 | Scale to d256 L6, only if M1 passes on both counts | Better than M1 on P2 by ≥0.002 standalone; the gain beyond the bank must not shrink |
| M3 | Dev run on T (predict probe) and full-bank test | ≥0.0005 dev CV, all five folds → promote, and hand the recipe to Codex's production pipeline |

**Stop rule:** if M1 fails both gates, close the moonshot.

## Compute and coordination

- **GPU order:** Codex's E3 → Codex's production GPU retrains (≈30–35 h) →
  M1. M0 runs on CPU meanwhile, so it doesn't break E3's launch condition
  (no other training live).
- **Estimates on the M3 Max:** M0 3–5 days of code; M1 ≈2–3 days GPU
  including pilots; M2 ≈1 week; M3 ≈1 week. **Total ≈3–5 weeks.** A d256 run
  may need the GPU alone (memory).

## Progress

**M0 step 1 — fork and long-context attention (2026-10-31).**
`scripts/moonshot_sequence.py` is forked byte for byte from N1's
`choice_sequence.py` (sha256 prefix 9dc0d0ce36e53dab); the helper modules it
imports are identical to the main repo's. It adds `attn_chunk` (default 0 =
N1 behaviour), which runs query-chunked attention: each chunk attends over all
keys, so the score tensor is n·heads·chunk·L instead of n·heads·L², the MPS
INT_MAX failure mode at 3072 tokens. `tests/test_moonshot_sequence.py` (4
tests, CPU) covers:
- chunked = full attention outputs within 1e-6, for chunk sizes 1–64,
  including padded rows and a length-1 row;
- gradients agree to ~1e-7 relative (float32 summation order);
- the default fork reproduces N1's predictions **byte for byte** on a fixture;
- the new field is off by default and absent from N1.

**M0 step 2 — MLP head and LR schedule (2026-11-07).** Both opt-in; the
defaults still reproduce N1 byte for byte.
- `head_mlp` (hidden width): a two-layer GELU MLP over (user vector, a
  dedicated item embedding, date features) is *added* to the bilinear
  output. Its last layer is zero-initialized and its parameters are created
  last, so at initialization predictions equal the bilinear model exactly.
  Requires `--context session`.
- `warmup_steps`, `cosine_epochs`, `lr_floor`: linear warmup, then cosine
  decay to `lr_floor`·lr over a **fixed** horizon of `cosine_epochs`. The
  horizon does not depend on the run length, so a refit for the selected
  epoch count follows the selection run's schedule exactly.
- Distribution loss dropped: E3 failed both P2 gates, so per the design M
  trains with MSE only (the existing `--loss` flag remains).
- Tests: 10 pass (`tests/test_moonshot_sequence.py`, run with
  `~/.cache/netflix-venv/bin/python -m unittest`). New ones cover: zero-init
  head equals the base model byte for byte; the head trains; head without
  session context is rejected; schedule endpoints and monotonicity; a 1-epoch
  scheduled refit equals epoch 1 of a 2-epoch run byte for byte, and differs
  from the unscheduled run.

**M0 step 3 — MPS memory/throughput pilot (2026-11-07, user go-ahead). M0 gate met.**
`scripts/moonshot_pilot.py` trains 40 steps per config on P2 data
(`artifacts/validation/train-matched-20260927.bin`): the 5 widest batches
first, then random ones. There is no prediction pass and no score. MPS was
capped at 19.4 GB (`PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=0.25/0.2`), with a
watchdog on memory pressure. It ran beside Codex's A2h production trainer.

Two engineering findings, each fixed with an opt-in flag and a test (13 tests pass):
- **`pad_row_tokens`.** N1's `pad_rows=64` pads every batch to 64 rows, so a
  13-user batch at width 1536 runs attention over 64×1536². That is why N1
  used ~57 GB driver memory, and it OOMed even the N1 config under the cap.
  The new flag pads rows to the largest power of two ≤ pad_row_tokens/width
  (≤ pad_rows). Predictions are unchanged (≤1e-6).
- **`checkpoint`.** Query chunking bounds the forward pass, but backward kept
  every layer's scores and dropout masks: M1's widest batch reached 16.6 GB,
  then OOMed. Per-block activation checkpointing (dropout RNG preserved;
  training is identical on CPU, ≤1e-6) fixes it.

| Config | Params | Peak driver memory | Widest batch | Throughput | Est. epoch (P2, GPU shared) |
|---|---|---|---|---|---|
| N1 recipe + pad_row_tokens 16384 | 25.9 M | **7.1 GB** (vs ~57 GB) | 1536: 0.93 s | 46.3 k tok/s | 0.6 h |
| M1: d128 L4 ff512, seq 3072, chunk 512, MLP head 256, pad_row_tokens, checkpoint | 38.1 M | **7.3 GB** | 3072: 2.6 s | 22.3 k tok/s | 1.3 h |

Evidence: `experiments/M0-pilot-20261107-a.{json,log}` (N1p; its M1 row is the OOM),
`experiments/M0-pilot-20261107-b.{json,log}` (M1 with checkpointing).

**Contention:** during the pilots Codex's A2h trainer slowed from ~23 s to
30–65 s per 100 batches. A multi-day M1 beside it would delay production
correspondingly.

**M1 plan:** selection run on P2 with patience, then the shared P2 gates.
About 12 epochs × 1.3 h ≈ 16 h per run if the GPU is shared; less alone.

**M1 result (2026-11-10). Moonshot closed.** P2 selection run (plan
`experiments/M1-20261109-plan.json`, 14 epochs, ≈34 min each with the GPU
alone): best validation RMSE **0.893387** at epoch 13 against N1's 0.868338.
That is 0.025 *worse*, so the standalone gate (≤0.865338) fails and M2 is not
reached. The curve was flat from epoch 10 (0.8941 → 0.8934). Per the stop rule
there is no re-tuning after reading P2. The bank gate was not scored: a model
0.025 behind N1 standalone has no realistic path to +0.0010 beyond bank11+N1.
A gap this large probably comes from one of the new pieces (3072-token
context, MLP head, or the warmup/cosine schedule with lr_floor 0.05)
rather than from tuning; no ablation was run to find out which.
Evidence: `artifacts/moonshot/M1-select.{json,log}`, `M1-select.select.progress.json`.
