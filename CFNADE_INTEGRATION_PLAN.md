# Integrate the original public CFNADE columns after complete training

The independent CFNADE workers are additional fresh source-model fits. They are
not yet production inputs. The primary scheduler's successor holds these jobs
to avoid duplicating that training; it does not mark them reproduced.

The pinned registry's final input list specifies these four additional columns:

| Column | Independent producer | Completion required |
| --- | --- | --- |
| `cfnade-48_ep25` | Original 160-hidden / 48-factor model | Both original 95-epoch phases |
| `cfnade-96_ep10` | Original 320-hidden / 96-factor model | Both original 95-epoch phases |
| `cfnade-96_ep95` | Same 96-factor training trajectory | Both original 95-epoch phases |
| `cfnade-96r_ep95` | Raw SGD export of that verified trajectory | Completed 96-factor phases and successful export |

The fifth public CFNADE column, `cfnade-12_ep40`, is already represented by the
terminal 40-epoch CFNADE-12 predictor in the fixed six-model source extension.
Do not append it twice. Use the four public snapshot choices above without
searching other epochs or subsets against qualifying outcomes.

Once the native jobs finish, the next implementation must:

1. Verify the two explicit independent-execution receipts, their frozen plans,
   actual commands, source/data/proof/binary hashes and positive completion
   evidence. Rehash the final epoch-95 checkpoint manifests and all 84 tensors
   per phase, including the 96-factor model whose native finalizer only checks
   terminal prediction shape and hash.
2. Validate every selected paired prediction's recorded hash, finite float32
   contents and row count (1,128,164 ProbeX / 2,817,131 qualifying). Preserve
   explicit source-driver provenance and the shared raw-SGD trajectory. Do not
   fabricate execution of the original command in the primary tree or count
   raw export as a second independent fit.
3. Append all four columns to the verified six-model extension, retaining its
   transforms, ridge intercept and development identities. This gives 95 rating
   predictors, 171 ridge-design columns and 231 nonlinear features. Qualifying
   outcomes stay zero. Use the original nested user folds and known-label budget
   selection, and independently replay and score the new development models.
4. Require the fixed equal tree/NN development average to beat the best verified
   preceding fixed-average development result. Only then assemble the matching
   paired production bank, fit on known ProbeX, replay all saved models, freeze
   the final qualifying vector and independently audit both recovered groups.

The existing 85-column production attempt, context-blend queue and six-model
extension continue under their frozen plans. This document does not launch a
new fitter, admit partial native outputs or establish a score improvement.
The goal remains below 0.85725 on both recovered groups, with no official or
untouched Test claim.
