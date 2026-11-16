# Dataset enrichment investigation

Branch: `research/dataset-enrichment`. Research accessed 2026-09-27.
Status: literature review and local feasibility audit complete; no enriched
model trained and no improvement claimed.

## Recommendation

Run a small, controlled metadata experiment before building a large external
catalog. Start with the supplied release year and verified series/package
relationships, then test structured external metadata and personalized content
features. Judge each addition against the frozen R5 ensemble, whose recorded
development RMSE is **0.8748094089**, rather than against a weak standalone
content model. This is a research recommendation, not a predicted gain.

The most directly relevant literature is cautious about external movie content.
One apparently positive side-information paper actually obtained its gains from
rating history and dates. Modern papers reporting better Netflix ranking results
do not establish an improvement on our original-probe rating RMSE.

## Literature and what it means here

| Source | Evidence | Implication for this repository |
|---|---|---|
| Lees-Miller, Anderson, Hoehn, Greiner, **Does Wikipedia Information Help Netflix Predictions?**, ICMLA 2008 ([paper](https://citeseerx.ist.psu.edu/document?doi=d02d7fd21b50a28be3d41254704e241f8f177829&repid=rep1&type=pdf), [DOI](https://doi.org/10.1109/ICMLA.2008.121)) | Wikipedia text/link similarity fed kNN and factorization, then blending. The authors found no significant overall accuracy improvement. Their broader mapping covered 83% of titles and 96% of probe ratings, with estimated 93% matching correctness. | Content similarity is worth an ablation, but matching quality and incremental ensemble value must be measured. Catalog coverage and rating-weighted coverage are different. |
| Pilászy and Tikk, **Recommending New Movies: Even a Few Ratings Are More Valuable Than Metadata**, RecSys 2009 ([paper text](https://www.researchgate.net/publication/221141047_Recommending_new_movies_Even_a_few_ratings_are_more_valuable_than_metadata), [DOI](https://doi.org/10.1145/1639714.1639731)) | Projects sparse metadata into collaborative latent factors. In its new-movie experiment, a simple bias model using ten movie ratings beats its metadata-only predictor. | A useful approach for new items, but neither a universal ten-rating rule nor a test of metadata's marginal value in our ensemble. Our probe has no completely unseen items. |
| Porteous, Asuncion, Welling, **Bayesian Matrix Factorization with Side Information and Dirichlet Process Mixtures**, AAAI 2010 ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/7686/7547)) | At 45 dimensions, Table 2 reports 0.8970 to 0.8866 with side information on their probe subset. The experiments section identifies dates, a date-change flag, previous ratings and similar-movie ratings as the successful inputs. Wikipedia director/actor/language features did **not** improve RMSE. They trained on 90% of the original probe. | Evidence for behavioral enrichment, not a demonstrated external-content win. Their scores are not directly comparable to our full-probe holdout. Much of this signal is already represented by our temporal and neighborhood models. |
| Tu, Wu, Yu, **Netflix Movie Rating Prediction using Enriched Movie Metadata**, Stanford CS229 project, 2008 ([report](https://cs229.stanford.edu/proj2008/TuWuYu-NetflixMovieRatingPredictionUsingEnrichedMovieMetadata.pdf)) | IMDb features and actor clustering gave 0.95957 baseline, 0.95881 unclustered and 0.95797 clustered on a 1,000-user experiment. The report describes title, translation and DVD-versus-film mapping problems. | Small positive evidence from a course project with a different protocol; useful engineering lessons, not strong evidence of a gain over R5. |
| Rendle, **Factorization Machines**, ICDM 2010, and **Factorization Machines with libFM**, 2012 ([author's reference page](https://www.libfm.org/)) | A general way to estimate interactions among sparse categorical features using shared latent factors. | A suitable second-stage model for user × genre/director/series interactions. Methodological support, not an empirical Netflix enrichment result. |
| **Geometric Collaborative Filtering with Convergence**, 2024 preprint ([paper, v1](https://arxiv.org/html/2410.03064v1)) | Uses item geometry and reports Netflix Recall/nDCG gains under a thresholded interaction task and user splits. | Supports exploring semantic representations, but its target and split differ from Netflix Prize RMSE. Do not compare its leaderboard claim with 0.8748094. |

## What the local data supports

The reproducible [audit](experiments/enrichment/local-audit.json) hashes both
inputs and reads item IDs without using rating values. It counted all
99,072,112 training rows and 1,408,395 probe rows, covering 17,770 movie IDs.

| Local observation | Count / probe coverage | Interpretation |
|---|---:|---|
| Supplied year present | 17,763 items; 99.9985% of probe rows | Cheapest metadata experiment; no external join needed |
| Year range | 1896–2005 | Missing years need their own flag |
| Titles containing season/volume/disc/collection/box-set markers | 1,174 items; 5.5350% of probe rows | Material package-matching workload; lexical markers are not verified series labels |
| Duplicate normalized title + year keys | 9 pairs; 376 probe rows | Even local title/year keys are not unique identities |
| Items with zero training ratings | 0 | This is not an item cold-start benchmark |
| Items with fewer than 100 training ratings | 1,102 items; 0.3284% of probe rows | Sparse-item gains alone affect relatively little aggregate RMSE |
| Items with fewer than 500 training ratings | 8,559 items; 3.5750% of probe rows | Report item coverage and weighted coverage together |

The supplied raw README explicitly says year can refer to DVD release rather
than theatrical release, and movie IDs are not IMDb IDs. Titles may contain
unquoted commas and Latin-1 characters. The audit handles both.

R5's `movie_age` in `scripts/round4_boost.py` means days since the first observed
implicit interaction for that item. It does **not** use the supplied release
year. A release-year feature is therefore distinct, though it may be redundant
in practice. Existing user means, rating extremes, popularity, activity and
item standard deviation already cover much of the simplest behavioral metadata.

## Candidate enrichment, in order

1. **Local year and package information.** Year, missing-year flag, rating
   calendar year minus supplied year, explicit season/volume markers, and
   reviewed series membership. Keep DVD year separate from externally verified
   theatrical year. A title prefix is only a candidate relationship: do not
   merge unrelated works or editions automatically.
2. **Structured movie metadata.** Genre, original language/country, runtime,
   media type, directors and a limited cast. Prefer a frozen Wikidata snapshot
   for reusable research data; its structured data is
   [CC0](https://www.wikidata.org/wiki/Wikidata:Licensing). IMDb's
   [non-commercial interface](https://www.imdb.com/interfaces/) is another
   research source, with title aliases and basics; its use conditions differ.
   Language of an alternative title must not be mistaken for the movie's
   original language. No external catalog has been downloaded or matched yet.
3. **Personalized content features.** For each user, derive genre/director/series
   exposure and shrinkage estimates of preference from training ratings only.
   Interact those with target metadata. Try an FM or an additive content term
   in item factors if tree features show promise. Global genre flags alone
   may mostly duplicate existing item biases.
4. **Plot embeddings and content neighbors.** Freeze the text revision,
   embedding model/version and vectors. Fit compression on training items and
   construct a separate content-neighbor predictor for ensemble diversity.
   Compare it with structured features and a shuffled-item-feature control.
   Current summaries or pretrained embeddings can encode later knowledge;
   label this a modern enrichment experiment, not a strict 2006 reconstruction.

For a catalog join, retain Netflix ID, external ID, raw and normalized titles,
both years, entity type, relation (same work / season / collection), candidate
count, match method, review status, source revision and retrieval timestamp.
Unique normalized title/year matches are candidates, not proof. Review a fixed
random sample and all ambiguous package mappings; freeze the mapping before
looking at residuals. Missing/ambiguous matches remain missing. Report accuracy
from review as well as item and probe-weighted coverage. This avoids selecting
matches because they happen to improve evaluation scores.

Exclude present-day aggregate ratings, votes, popularity and post-release awards
from the first experiment: their availability dates differ from the original
rating task. Enrich movie entities; this approach does not require resolving
anonymous users to external identities.

## Controlled experiment specification

Freeze these arms before fitting: **E0** reproduce R5 unchanged; **E1** E0 plus
local year/package fields; **E2** E1 plus verified structured metadata; **E3** E2
plus personalized content summaries. Plot embeddings are contingent on these
results. Keep the predictor bank, seeds, regularization and inner stopping
protocol fixed so added features are the main difference.

Use `round4_boost.py --extra-support` for float32 feature columns in exact
processed probe order. Store source hashes, row-order/dataset hash, feature
definitions and mapping provenance alongside them. Content columns are tree
features, not additional prediction columns for contextual ridge. Add alignment
checks before running a blend. When scoring training holdouts, recompute user
content statistics from that split's training CSR; target ratings must never
enter their own features or neighbor aggregates.

For the first development comparison, use the existing five user folds and
inner user split. Report overall/fold RMSE, user-cluster paired error intervals,
coverage, low-user-history and low-item-support slices, package slices and
unmatched rows. Do not tune only on matched rows. A missingness-only arm and a
fixed shuffled-feature arm help distinguish content from catalog-selection
effects or added model capacity.

Proposed continuation threshold: at least **0.0001 aggregate RMSE reduction**
over the matched control, improvement in at least four of five folds, and no
large unmatched-row regression. This is a practical research gate, not a
statistical guarantee. Confirm promising additions on training-only validation
as in `HOLDOUT.md`, recognizing that its distribution differs from the probe.
All original-probe results remain development estimates after repeated model
selection; neither user folds nor metadata make that probe a fresh test set.

The R5 artifact bank is not available at the paths documented in ROUND5.md in
this checkout (`artifacts/r4` is absent). Recover and verify the exact inputs
before claiming a matched R5 ablation. This investigation has not run that
training experiment.

## Reproduce the completed audit

```sh
python3 scripts/enrichment_audit.py \
  --output experiments/enrichment/local-audit-rerun.json
python3 -m unittest discover -s tests -p test_enrichment_audit.py
```

The fixture verifies comma/accent parsing, binary alignment, duplicate keys,
support counts, truncated-input rejection and invariance to changed rating
values. Input hashes intentionally change when ratings change; audit statistics
do not. The script requires only the Python standard library.
