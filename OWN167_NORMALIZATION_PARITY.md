# Complete normalization parity for the next own model

The own167/context218 source tensors were reconstructed independently on
all1,128,164 known rows. Every243-column ridge and521-column nonlinear
input bit matched, and all ten ridge group statistics matched the unchanged
original reference implementation bitwise.

The subsequent normalization proof preserves all522 NN input columns and
the exact original eligible row order for outerfold0:

| Statistical population | Complete eligible rows |
| --- | ---: |
| Inner training | 721,484 |
| Full outer training | 901,418 |

Both populations' complete percentile bounds, means and population standard
deviations match the original NumPy reference bitwise. Every normalized
float32 value also matches the original transform. These are actual source
matrices, not a sampled statistical test. The scope is the two populations
for fold0; no claim is made that later models or other folds already ran.

To bound memory, construct exactly the matrix that the original row mask
would select, with every selected value checked against the original full
block conversion. Compute the unchanged percentile/clipping operations,
release that input buffer before NumPy computes variance, then reconstruct
the same eligible matrix and check its complete fingerprint before testing
the bounded implementation. This changes buffer lifetime, not data,
statistical dimensions, rank, model capacity or epoch budget.

The supervisor reserves8GiB of measured temporary footprint and enforces
24GiB total research footprint including compressed/swapped anonymous
pages, alongside14GiB own RSS/footprint,24GiB total RSS and16GiB free disk.
This avoids treating the native jobs' low RSS as their full memory usage.
The actual complete parity run finished under this guard.

Proofs are in `experiments/upstream-own167-context218-features/` and
`experiments/upstream-own167-context218-normalization/`. No NN has been
fitted on this candidate yet. Next is the declared complete five-fold,
three-seed nested study, actual serialized-model/scaler replay and
independent scoring, with no qualifying-based model selection.
