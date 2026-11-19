# Add the completed CFNADE-12 signal to the context neural blend

The completed fresh CFNADE-12 model improved the five-fold ridge diagnostic by
0.00026658 RMSE, with every fold improving. Test its contribution to the context
neural blend now, while the fixed six-model native extension continues.

`scripts/upstream_cf12_context.py` requires the full original paired CFNADE-12
receipt, verifies all receipt output hashes, finite float32 paired predictions,
native row identities and terminal/epoch-40 equality. It verifies the existing
85-predictor parent tree's saved-model replay and fold artifacts, and binds the
preceding context blend's independently verified result and prediction vector.

Append CFNADE-12 to the parent ridge design before its intercept and to the neural
features after the original 221 columns. The resulting design has 162 columns;
neural inputs have 223 columns including their newly fitted ridge prediction.
The new CFNADE and ridge signals enter the signal branch. The five original
context features and 130 voting features keep their original branch and order.
Use the same rank-16 interaction, optimizer, seeds, nested user folds, scalers,
budget selection and refitting rules as the existing context experiment.

Fit only the new neural component. Reuse the original 85-predictor tree's actual
OOF predictions, with explicit provenance and no new tree-fit receipts. Evaluate
the fixed equal mean of that tree and the new neural predictions against the
preceding context mean's independently verified RMSE, 0.860216195766495. The tree
keeps its original ridge; the new NN uses the 86-predictor ridge. They are separate
absolute prediction components. Replay all saved neural models bit-for-bit and
independently score the stored final vectors and fixed average.

The ten-core machine currently has capacity for a two-thread low-priority
worker. The runner counts other native and workspace Python process groups,
admits below 17 GiB other RSS, and stops only its own child after sustained total
RSS above 24 GiB or free disk below 22 GiB. Completed folds remain receipted;
incomplete artifacts are preserved for inspection. Existing native fits retain
their original settings and jobs.

Three focused tests verify the 223-column signal/context partition, actual
training and saved-model replay, and scoring-label exclusion through the actual
fold fitter. The last test changes held-user labels to 100 and verifies bytewise
identical ridge, scaler and prediction artifacts and identical selected budgets.

This is an adaptive known-ProbeX experiment, not a qualifying result. It does not
change the frozen 85-column qualifying attempts or the fixed six-model extension.
If it improves development, production integration still requires its matching
86-predictor NN inputs, the verified 85-predictor tree component, full-data refits
and replay, frozen qualifying predictions, and independent checks of both
recovered groups. No qualifying outcomes enter preparation, fitting or selection.
