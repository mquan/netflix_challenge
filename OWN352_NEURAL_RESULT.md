# Complete temporal NN improves the development best

The full579-input study scores **0.8590361082610746** for the neural model
and **0.8589160570135321** for its predeclared fixed half with tree154.
The latter improves the preceding development best0.8589474951024617 by
0.0000314381 on the same1,128,164 known rows and original five user folds.
This is a small, independently verified improvement.

All352 scalar predictors and90 learned context components remain, with
354 signal and225 context positions, H128/rank32, three seeds, max40 epochs,
patience4, batch8192 and original AdamW. All15 selected outer refits completed.

The original independent verifier reached the compressor guard after
fitting completed. Its12.4GiB peak stayed below the16GiB worker cap, but
system compressed memory exceeded5GiB. After all15 models and five receipts
were rehashed, only that suspended read-only verifier was retired. No model
was retrained, shortened or changed. Original code, failure and watchdog
evidence remain intact.

The separate bounded verifier reproduces complete eligible percentiles and
original ordered float64 mean/std reductions with small scratch buffers.
Every statistic exactly matches the actual saved scalers; every normalized
input and all15 saved-model held prediction vectors replay. Separate final
arithmetic reconstructs each fixed-half float32 value and the full canonical
score. All original memory, compressor, swap and disk limits remain.

This is adaptive known-development CV, not official or untouched Test.
The later own352 fold-bag qualifying primary scores0.8604024389/0.8613852654;
the [complete13 comparison](OWN13_QUALIFYING_RESULT.md) improves both slightly
to0.8603920746/0.8613737959. The strict0.85725 target remains unmet.
[Original frozen qualifying fold-bag declaration](OWN352_FOLD_BAG_PRODUCTION_PLAN.md).

[Complete actual pipeline replay](experiments/upstream-own352-bounded-replay/independent-verification.json) ·
[Separate canonical score](experiments/upstream-own352-bounded-replay/independent-canonical-verification.json) ·
[Guard and recovery plan](OWN352_BOUNDED_REPLAY_PLAN.md).
