# Original split3 attention pilot

This executes the first independent native pilot from the 263 split3 raw
features identified in `READY_OUTPUT_COVERAGE.md`. The target remains RMSE
strictly below 0.85725 on both recovered qualifying groups. No new score is
claimed by preparing or launching this job.

The fixed model is the original `s3-attn16`: attention dimension16, MF
dimension192, pool32, seed11,12 epochs per phase, original single model thread,
with every epoch prediction saved. Both train_3/probe_3 and fulltrain/qual
phases must finish. The native command and environment are recorded verbatim
in `experiments/upstream-split3-pilot/plan.json`.

The isolated source tree contains pinned source/configuration files only.
Public predictions and the bundled recovered-label payload are excluded.
Verified own train/probe/fulltrain/qual arrays are symlinked as inputs; the
upstream ingest job is never invoked. The original fit executable matches its
pinned build receipt, and the unchanged native split_3 builder is compiled
separately. All source, actual inputs and executables are hash-bound.

An actual preflight verified that all1,128,164 known ProbeX rows form a unique
exact `(user,item,date)` subset of the1,408,395 original Probe rows, with equal
known ratings. Its mapping and source hashes are recorded in
`original-holdout-projection.json`. The generated probe_3 mapping must be
checked again, and every evaluation identity must be absent from generated
train_3, before the model runs. Tests reject duplicate/missing identities,
changed known labels, out-of-range IDs and evaluation/training overlap.

The controller waits for two stable memory samples before each native stage.
It can admit the small pilot alongside the exact already allocated tsvdx5-60
process identity if total owned native footprint stays below20GiB, compressor
below6GiB and swap growth below32MiB per30seconds. Otherwise it requires the
primary scheduler absent and owned footprint below12GiB. FourGiB is an
estimated pilot allowance, not a measured peak. The CF96 admission policy is
unchanged. A controller originally waiting for primary idle was stopped only
after verifying it had no native child; its original plan/code/status are
preserved in `before-memory-admission-update/`.

Controller PID40527 was launched at11:02:08UTC on November19. It owns no old
native jobs. Once a native stage starts, observation timeouts never kill or
restart it. The existing tsvdx5-60 and1,000-factor temporal processes continue
their complete original schedules; tsvdx5-60 has23 epochs per phase.

The native split_3 data stage completed at approximately11:04UTC:98,720,013
training rows and1,760,494 validation rows. The controller verified the actual
mapping for all1,128,164 known rows and checked their absence from all generated
training rows, then froze all generated array hashes. At11:05UTC it was waiting
for memory admission for the attention fit; the data stage is complete, while
model training is not yet claimed to have started. See `data-verification.json`
and `startup-verification.json` for the observed evidence.

At11:08:33UTC, native PID42926 started the original two-phase attention fit.
Its actual command is recorded in `model-launch.json`, and its initial log
confirms the original16/192/12/seed11 configuration. The prior waiting snapshot
remains historical evidence, not the current model state. No completion or score
is claimed before the full native receipt and separate verification exist.

On successful completion, retain every native output and the complete actual
two-phase receipt. Check all declared outputs, finite float32 query vectors,
terminal/epoch12 equality, and project the fixed terminal and published epoch8
columns onto known ProbeX. Keep paired qualifying outputs in the isolated
tree. Independent completion verification must precede admission into a blend.
Candidate fitting uses known labels only; recovered-group audits remain
post-freeze evaluations and are not official or untouched Test claims.
