# Next own-trained model: all available warm predictors and complete AE50 contexts

Declared before the new NN-pool qualifying audit. This candidate is selected
from completed own source coverage and fixed architecture compatibility,
without using a new qualifying outcome.

Use the already verified own166/latent90 development bank, which includes
all 59 captured own native snapshot additions, rather than dropping those
features after their earlier standalone study. Append the independently
completed own item autoencoder's final epoch50 raw predictor, all64 existing
own item PCA contexts, and all64 verified final-AE50 user contexts. The
result has 167 raw predictors and 218 learned context columns.

Keep the existing rank32/hidden128 MPS context architecture, five outer
user folds and original inner partitions, seeds2026/7/13, full data,
batch8192, AdamW settings, maximum40 epochs and patience4. Each seed's
stopping is selected on inner users and refitted freshly on all outer
training users. Preserve the actual saved model and preprocessing state
and independently replay every outer-fold prediction.

Feature order is fixed: retain the first302 warm166 nonlinear columns;
insert raw AE50 clipped0..6 minus float32(3.6); retain the old90 learned
contexts; append existing64 item contexts and final-AE50 user64 contexts.
This produces521 nonlinear columns plus the ridge baseline,522 NN inputs.
Insert the raw predictor before the old ridge intercept, producing243
ridge columns. Existing values remain bitwise unchanged. All64 user
contexts use the same trainx-derived user coordinate tables for both
queries; fulltrain AE50 is used only for the raw qualifying predictor.

Build complete C-order fingerprints and every ridge group statistic from
streamed source blocks. Use RAM assembly and exact bounded normalization;
do not create another multi-gigabyte persistent cache. Release ridge and
source mappings before training. Numerical execution changes require
bitwise comparison with the complete original transform/statistics on
actual rows; never reduce data, capacity, epochs or seeds to fit resources.
Use an exclusive GPU compute lock, sustained24GiB total research RSS,
14GiB own-worker limit,16GiB disk floor and current compressor/swap admission.

Compare the new NN with own107/item154 NN0.8593837349 and its own166/90
predecessor0.8594144161 on identical row/user folds. Predeclare one fixed
half with the already saved tree154 for a global development comparison.
Keep all results even if negative. No full refit or qualifying audit follows
unless the known-only evidence earns eligibility. A matched NN improvement
is reported separately from any global ensemble improvement.

The current NNpool production/refit and independent admission/audit chain
takes priority. This next model starts after those GPU replay jobs release
their slot. The first implementation milestone is full source/column and
exact normalization verification, followed by actual complete nested fits.
