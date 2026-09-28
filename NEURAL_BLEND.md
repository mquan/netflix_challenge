# Nonlinear blending and predictor diversity

## What the linked accounts imply

[Amatriain's retrospective](https://amatria.in/blog/on-the-usefulness-of-the-netflix-prize-403d360aaf2/)
describes 104 predictors and a shallow neural blend. The Quora page was not
accessible through the research tool; this is the author's accessible account.

[Netflix's 2012 post](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
reports SVD at 0.8914, RBM at 0.8990, and their linear combination at 0.88.
It distinguishes a 107-algorithm first Progress Prize entry from the later
grand-prize ensemble of hundreds of models. Those historical competition
scores must not be compared directly to our original-probe development CV.

The [BigChaos 2009 technical report](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/TocherGrandPrize2009_BPC_BigChaos.pdf),
sections 6–8, describes nonlinear blends, small neural networks on predictor
subsets, and a final linear combination. It also describes selecting model
settings by ensemble improvement. Thus a single network over exactly 104
columns is not a complete technical specification of the winning system.
The [2008 report](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/ProgressPrize2008_BigChaos.pdf)
reports a 0.0020 neural-over-linear blending gain on its predictor bank.

**Decision:** test the combination rule and family diversity separately.
Keep the existing bank frozen when testing a new blender, then add a completed
RBM column to measure family diversity. Counting predictors is not the objective.

## NN1: nonlinear correction to the existing contextual blend

Use exactly the 12 raw predictor columns in the previous best ensemble. A
contextual ridge model is the matched control. A small neural network learns a
residual correction to it, with one hidden layer of 16 tanh units and a linear
output. Inputs are the mean prediction, each model's deviation from that mean,
and the same four label-free context variables used by the C++ ensemble.
These inputs emphasize disagreements among models, while preserving the full
prediction information. This is a controlled adaptation, not a reproduction
of the historical network.

Every outer user fold independently:

1. Fits a ridge baseline on an inner subset of its training users.
2. Fits normalizers and the network on that subset; chooses epoch count on
   separate inner validation users, up to 60 epochs with patience six.
3. Refits the baseline and network on all outer training users for the selected
   duration, then predicts the outer held-out users.

The neural head starts at zero, and zero epochs is an allowed outcome if inner
validation never improves. AdamW uses learning rate .001 and weight decay .01.
Batch size is 8192. Five folds match the existing user hash. The Python ridge
control is checked against the C++ control to make comparisons interpretable.
The outer held-out labels never select duration or normalization parameters.
A synthetic test changes those labels and checks that predictions and the
chosen epoch count for that fold remain exactly unchanged.

Raw base-model checkpoint selection still used the probe, so these scores
remain development estimates. Never feed previous cross-fitted ensemble columns
into this script: their training histories may include labels from the new
outer holdout. Any later ensemble of neural blends must preserve nesting.

## NN2: introduce a completed RBM family

The original branch's R2 training completed successfully. Copy its prediction
column only after verifying the exact SHA-256 against its completed experiment
manifest. The source manifest is archived at `experiments/NN2-rbm-source.json`.
AutoRec columns with incomplete manifests were excluded. No trainer or original
artifact was changed.

The RBM was trained by the agent working on `score-iteration-2`; this branch
claims only the blending experiment. Its trainer is not included here. Repeating
NN2 requires the completed RBM prediction column identified by that manifest.

Compare the same contextual ridge blender with and without the RBM column;
then apply the same neural-blending protocol to the expanded bank. This isolates
predictor-family diversity from the change of blending method.

## Results

All comparisons use the same 1,408,395 probe rows and five user folds. Lower
RMSE is better. The table scores the saved float32 predictions.

| Raw prediction bank | Contextual ridge | Ridge + neural correction |
| --- | ---: | ---: |
| Existing 12 columns | 0.8841835475 | 0.8835714199 |
| Same 12 + completed RBM | 0.8811258732 | **0.8803262843** |

The RBM's standalone RMSE is 0.9126515614, yet it improves the contextual
ensemble by 0.0030576744. Neural blending adds 0.0007995889 on the expanded
bank. Both gains occur in all five folds. The neural gain's paired user-cluster
95% interval for change in MSE is [-0.00154062, -0.00127626]; this is a
descriptive interval conditional on the selected models, not protection against
repeated development on the probe. The `outside_pilot` rows in the paired logs
refer to the earlier GP pilot and are not an independent neural validation set.

The final score improves by 0.0057708709 against this branch's frozen six-column
starting control (0.8860971552). That crosses the original 0.005 research target
in development CV, with a substantial contribution from the other agent's RBM.
It is not a measured improvement on Netflix's hidden test set.

The Python ridge control agrees with the existing C++ implementation within
2e-10 RMSE on both banks. NN1 selects epochs [37, 55, 37, 36, 31]; NN2 selects
[32, 46, 38, 38, 38]. Summed fold fitting/scoring time is approximately 135 and
112 seconds respectively, excluding input/context construction. These are
single-seed experiments. The strengthened synthetic test passes with a nonzero
neural correction and exact invariance to changed outer-fold labels.

**Next bet:** expand structurally distinct raw predictors and judge each by its
incremental ensemble gain. Completed AutoRec variants are candidates once their
artifacts are stable. Small neural blends over predictor subsets are a later
experiment; their combination needs nested fitting to avoid label leakage.

The complete configurations, hashes and fold results are in
`experiments/NN-round-manifest.json`; paired comparisons are in
`experiments/NN1-paired.json` and `experiments/NN2-paired.json`.

## Running the experiment

```sh
export MOONSHOT_PYTHON=/path/to/python-with-numpy-and-pytorch
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$MOONSHOT_PYTHON" tests/test_moonshot_nn.py
nice -n 15 bash scripts/moonshot_nn_bet.sh
nice -n 15 bash scripts/moonshot_nn_bet.sh NN2-blend rbm
```

The script is CPU-only and limits PyTorch and BLAS to one thread. Prediction
artifacts remain local. JSON output records the exact input-column hashes,
configuration, selected epochs, fold scores and aggregate scores. The script
emits cross-fitted predictions and the matched ridge predictions; it does not
export a model for serving or qualifying-set inference.
