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

**Next M0 step:** the MPS memory and throughput pilot (`--max-batches`) for
full history (`--seq-len` ≈3072, `--attn-chunk`) with the MLP head. It needs
the GPU, which stays with Codex's production queue first.
