# Final item-autoencoder user context

Require independent completion of BOTH original own50-epoch item-autoencoder
phases and all saved-query replay evidence before extraction. Current early
snapshot predictors are weak and their earlier5/10 neural addition failed.
This experiment measures their learned representation rather than treating
source completion as an accuracy improvement.

Use final trainx model weights only: all480189 user columns of the128-unit
encoder, transposed to users x128, and all480189 decoder user rows.
For each matrix, center all users, compute the complete128x128 covariance
and its top32 eigenvectors, fix signs by the largest absolute loading and
scale scores by population standard deviation. Append both32-column tables.
All qualifying queries use the SAME trainx user table as known queries, so
context coordinates stay identical. No qualifying labels or original public
numerical output enter extraction. No rank/epoch/side search is planned.

Independent verification recomputes covariance and top spectrum, checks
orthogonality/eigenpair residuals/signs/scales, replays every projected
user and every known/full query gather, and rehashes all source/model outputs.
No score is claimed by extraction.

Planned follow-up keeps own107/item154 as the fixed parent. Add final50
raw predictor and all64 new contexts: own108,218 support columns,184 ridge
columns,462 nonlinear features and463 neural inputs. Keep the existing
H128/rank32,five nested user folds,three seeds,batch8192,max40 selection
epochs,patience4 and selected full refits. Use exact memory operators and
full input/scaler proofs; compare both fixed tree50/tree154 half means and
replay all15 saved neural models before any promotion. No qualifying refit
without an independently improved development result.

Extraction is queued behind actual complete50/50 evidence and full neural
slot headroom. It must preserve the current raw source trajectory.
