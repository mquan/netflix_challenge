# Intermediate progress after October 11

These are completed original-Probe development measurements, not official Netflix scores. Unpromoted screens appear faded on both graphs. They are not joined into the retained score frontier.

| When (Pacific) | Experiment | RMSE | Status / reason |
| --- | --- | ---: | --- |
| Oct 18, 12:11 | [HR2a: Honest A2h residual blend](experiments/journey-history/HR2a-blend.status.json) | 0.8668444071 | Two folds worsened and the mean gain was below the continuation threshold. |
| Oct 18, 18:13 | [HR2b: Honest Bayesian residual blend](experiments/journey-history/HR2b-blend.status.json) | 0.8668187324 | The standalone correction helped, but the full blend failed two folds and was not promoted. |
| Oct 19, 00:28 | [LC6: Training-trajectory snapshots](experiments/journey-history/LC6-final-verification.json) | 0.8668264382 | The verified epochs 4/8 snapshot blend improved the mean slightly, but one fold worsened; not promoted. |
| Oct 31, 05:26 | [N1 full: Choice-token transformer blend](experiments/journey-history/N1-full-independent-verification-20261031.json) | 0.8667006277 | Independent original-Probe scoring reproduced a 0.0001700 gain; fold 1 worsened, so the full-bank candidate was not promoted. |
| Oct 31, 09:24 | [H8 full: Matched headroom-feature blend](experiments/journey-history/H8-full-independent-comparison-20261031.json) | 0.8664848395 | All folds improved, but the gain versus its matched control was 0.0002203, below the fixed 0.0003 gate. It was not promoted. |
| Nov 15, 10:34 | [PUB201 calibration: Nested movie calibration](experiments/journey-history/item-calibration.json) | 0.8659848000 | A user-disjoint inner split chose the per-movie correction penalty; all folds improved against the matched ridge. This is a separate screen. |
| Nov 15, 10:51 | [PUB466 ridge: 466-base contextual ridge](experiments/journey-history/PUB3-union-ridge-screen.json) | 0.8655424521 | Adding strict-training-subset public predictions improved the ridge screen; the subsequent 466-base NN2 was lower. |
| Nov 15, 10:52 | [PUB201 NN2: 201-base local/public NN2](experiments/journey-history/PUB1-union-NN2-20261115.events.jsonl) | 0.8651051433 | The first public-bank neural blend improved the development mean; no formal promotion is asserted. |
| Nov 15, 10:54 | [PUB201 tree: 201-base residual tree](experiments/journey-history/PUB1-union-tree-20261115.events.jsonl) | 0.8653223453 | Completed nested user-fold tree comparison on the original Probe; a measured development candidate, not a promoted checkpoint. |

October records were recovered from available Git research history. Their original JSON reports are published unchanged under `experiments/journey-history/`; [provenance.json](experiments/journey-history/provenance.json) records source commits, paths, hashes and date bases. Recording dates for N1/H8 are upper bounds, not exact training completion times.

The October 31 H8 screen reached 0.8664848395 but failed its predeclared matched-control gain threshold. R9 therefore remained the retained promoted reference.

SO1, E3, E5, M1 and related P2-only pilot scores use a different development population and are excluded from this Probe chart. Production refits without scored outcomes are not plotted as score progress. The November 8 S0 report was a bank-size/quality diagnostic, not a new promoted checkpoint.

The separate recovered qualifying audit is **0.8571903481**, below **0.85725**. Its different query population and production training prevent treating it as a continuation of the Probe CV curve. See [BREAKTHROUGH.md](BREAKTHROUGH.md).
