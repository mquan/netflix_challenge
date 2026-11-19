# Own107 predictors plus original RBM50 context

Train the original rank16, hidden64 context residual neural blender on our
107 prediction columns plus all 50 original learned RBM user components.
Retain the same five nested user folds, seeds, optimizer and stopping rules.
This is 107 raw predictors, not 157 raw predictors: the ridge has 183 columns,
while the nonlinear cache has 293 features and receives the ridge prediction
as input 294. The signal/context partition contains 109 and 185 entries.

Our original RBM50 completed its first 20-epoch phase. Original public predfeat
uses that trainx user-factor table for both known and qualifying queries.
Every gathered component on every row of both queries was independently
checked bitwise. These non-rating features enter unchanged, without clipping
or rating centering. Both full 20-epoch native phases continue separately;
this support receipt does not claim whole-model completion. Original Tx20lat
remains queued afterward; its 40 components are not ready or included.

The raw107 cache retains own105 and appends our complete CF96 epoch10 pair
and original completed23/23 temporal60 pair. Ridge, labels, users and folds
are byte-identical to the already verified raw107 cache. Adding context does
not imply a new ridge solve. All retained and new nonlinear columns were
checked; the existing independent ridge replay remains bound.

Controller98487 started actual worker98536 after shared-lock and measured
memory admission. A small actual fit/save/reload test passed bitwise before
launch; it is integration evidence, not an accuracy result. Full five-fold
scoring and all saved-model replays are required before any promotion. Compare
standalone NN and the fixed half mean separately against NN103's verified
0.8594692563654073. A lower aggregate score is eligible without an all-fold or
minimum-gain veto. Qualifying labels do not enter fitting, stopping or selection.

If development improves, prepare a separately verified paired bank, refit only
on known labels and freeze/replay the selected candidate before recovered-label
scoring. Preserve original tree91's bank/ridge if reusing that family. Prior
nine-column development/production realization differences remain disclosed.
The qualifying target is below0.85725 on both groups and is not yet achieved.
