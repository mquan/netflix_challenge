# Conditional full refit of the own103 context blend

`scripts/upstream_cf48_103_production.py` waits on the shared fitting lock held
by the current nested context experiment. It proceeds only after five-fold
evaluation, independent scoring and saved-model replay are complete, and the
fixed equal tree/NN combination improves the previous verified development best.
There is no minimum-gain or all-fold-win requirement. A non-improving result
ends this queue without a production refit or qualifying audit.

The full-known NN uses the verified paired103 bank: 179 ridge columns,
239 features and a ridge prediction give 240 neural inputs. Its three epoch
budgets are per-seed medians from the five actual nested training receipts.
All ridge weights, scalers and neural weights fit known ProbeX only.

The tree family is the already trained original tree91. Reuse requires the
exact original paired91 bank, 167-column ridge, 227-feature schema, source,
environment, budgets, model files, prediction vector and actual saved-model
replay receipt to match their hashes. The NN103 bank and ridge never enter tree
inference. Revalidating this unchanged tree's existing actual replay avoids
repeating its approximately 20-minute inference pass; this iteration does not
claim to rerun that pass.

The neural family is refitted and independently reloaded for an actual bitwise
query replay. Publish the exact tree vector with explicit reuse provenance,
then freeze the equal float32 family mean. Only after freezing does the generic
runner audit community-recovered qualifying labels. Run the separate immutable
frozen-mean checker through its adapter API afterward. These adaptive recovered
groups are not an official or untouched Netflix Test.

The actual generic neural production worker passed a synthetic end-to-end
fit/save/reload/bitwise-query-replay check using 240 inputs and all three
original seed values. That fixture checks integration only and supplies no
accuracy evidence. Existing experiments and native jobs were not modified.

No score promotion follows from preparing or queuing this work. The target
remains RMSE strictly below 0.85725 on both recovered qualifying groups, and
the best qualifying result remains the previous verified tree91/context86
combination until an actual better result is checked.
