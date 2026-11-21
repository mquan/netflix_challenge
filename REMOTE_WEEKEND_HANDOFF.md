# Weekend assignment: three substantial fresh original producers

Coordinator branch: `research/full-upstream-reproduction`.
Handoff branch: `research/remote-weekend-handoff`.
Remote execution branch: `research/remote-weekend-execution`.

Execute this assignment; no additional planning round is needed. Use the
previous Apple M3 Max/96GiB laptop, after confirming actual hardware,
free disk and live processes. Preserve all old outputs, checkouts and jobs.
Our goal is below0.85725 on BOTH community-recovered groups using freshly
trained own predictions and known-only blend weights. Current recovered
scores remain0.8603920746/0.8613737959. This training assignment promises
new independent model coverage, not a particular score improvement.

## Assignment

Run these producers serially, retaining their complete original source,
settings, data, seeds, threads and epochs. Each must complete BOTH
`train_3→probe_3` and `fulltrain→qual` before any snapshot is eligible.
No completed receipt for these three exists in the coordinator inventory;
still check this laptop before launching a duplicate. An existing complete
producer requires an independently verifiable original source/data/run
receipt, not a prediction filename alone.

| Order | Original producer | Full epochs | Distinct method | Planning estimate |
|---|---|---|---|---|
|1|`s3-rbm400pf`|15+15|400-unit bipolar RBM, internally trained rank50 factorization|3–8h, uncertain|
|2|`s3-tx300r`|24+24|300-factor temporal model,100 neighbors, epoch12 reset|15–22h, uncertain|
|3|`s3-rbm800p`|25+25|800-unit bipolar RBM|14–24h, uncertain|

The combined estimate is32–54h plus preparation/verification/transfer.
Measured related remote runs: RBM200e25+25 took3h32m; TX60o25+25 took3h37m.
These scaling estimates are not measurements of the assigned recipes.
The machine-readable exact original commands/environments/registry objects
are frozen in `experiments/upstream-remote-weekend/plan.json`.
No `LM`, smaller rank, lazy-day, alternative data or thread overrides.
Factorization parameters are initialized and trained from scratch;
`NFACT=50` never means reading downloaded pretrained factors.

Return ALL terminal/epoch pairs:16+25+26=67 named pairs,64 epoch pairs
before terminal/final aliases. Return every complete native holdout query,
qualifying query, training-prediction array, factor output and source/binary
receipt as well as the canonical known projections. No source/epoch is
selected using an individual error or qualifying outcome.

Do not duplicate coordinator original1200, completed1600, CF96 trajectories
or the previous remote15 producers/extensions. Do not run original ingestion
or download numerical prediction banks. Qualifying targets must be zero.

## Isolate and prepare

Use the existing remote research checkout/environment; adjust paths only
to match this machine. Do not reset, clean or switch a dirty/live checkout.

```sh
cd ~/workspace/netflix_upstream_repro
git fetch origin research/remote-weekend-handoff
git worktree add ~/workspace/netflix_remote_weekend -b research/remote-weekend-execution origin/research/remote-weekend-handoff
cd ~/workspace/netflix_remote_weekend
```

Create `experiments/upstream-remote-weekend/execution/start.json` with UTC
arrival, actual hardware, paths, disk/live-process observations and fixed
UTC training/return deadlines. Ask the owner to correct the assumed deadline
if their laptop availability differs. The owner confirmed availability until Monday Nov23 at08:00 Pacific.
Train until Monday at06:00 Pacific (`2026-11-23T14:00:00Z`), finish return
by08:00 Pacific (`2026-11-23T16:00:00Z`). Never silently extend this deadline. The helper
requires10/24/24h remaining respectively before admission; skip a producer
that cannot plausibly finish in full. Keep every completed earlier producer.

Use `~/workspace/netflix_upstream_repro/.venv/bin/python`. Prepare a new
owned native tree, using only the hash-verified existing original code/data:

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_weekend.py prepare \
  --source-tree ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh \
  --data-root ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data \
  --tree ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/source
```

The helper verifies every original source/data hash, builds unchanged `fit`
and `split_3`, reproduces all split hashes, recomputes the complete known-row
projection, and checks known ratings are absent from training. It never
invokes ingestion. No raw ratings are uploaded. If an earlier matching split
exists, leave it intact; the new isolated preparation keeps its own receipt.
Do not rerun prepare over a partial directory; preserve it and report why.

The96GiB runner reserves64GiB per native job,80GiB research physical/RSS,
12GiB compressor (admission below10),128MiB original swap growth,32GiB disk
at admission and20GiB continuously. Three stable admission samples required.
A bootstrap launch gate records the exact process before model execution,
with a live watchdog already running. It cannot dispatch the native model
without both the live owner and watchdog. A guard or owner loss preserves
the same actual process in RAM, with no
retry or recipe reduction. Report it promptly rather than restarting.
Only the explicit availability deadline terminates an incomplete producer;
such a producer is ineligible and must remain labelled incomplete.

## Run, export and archive each completed producer immediately

Use tmux, AC power and `caffeinate -i` around the session. Run one producer
at a time, first `s3-rbm400pf`, then `s3-tx300r`, then `s3-rbm800p` if admitted.
The command below uses the owner-confirmed deadline. Record that same
deadline at arrival; change it only if the owner explicitly changes availability.

```sh
~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_weekend.py run-job \
  --tree ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/source \
  --job s3-rbm400pf --deadline-utc 2026-11-23T14:00:00Z

~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_weekend.py export \
  --tree ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/source \
  --job s3-rbm400pf \
  --output ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/exports/s3-rbm400pf

~/workspace/netflix_upstream_repro/.venv/bin/python scripts/upstream_remote_weekend.py archive \
  --tree ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/source \
  --job s3-rbm400pf \
  --output ~/workspace/netflix_remote_weekend/artifacts/upstream-remote-weekend/archives/s3-rbm400pf
```

Repeat all three commands with the next producer name. Export independently
rehashes source/data/binary/run/all native outputs, checks both full epoch
traces, exact full training/query/factor shapes and finite arrays, terminal/final identity and every compressed known/
qualifying pair bit. It retains full native query and side arrays via owned
hardlinks, without changing their bytes. Archive parts contain at most1GiB
of original file payload each, with every decoded member rehashed before
return. No original output/export is deleted. Do not put numeric arrays or
archive parts in git.

After EACH producer, upload its archive parts and `index.json` to a new
transfer-only draft GitHub release in `mquan/netflix_challenge`, tagged
`remote-weekend-20261120`. Keep it draft; do not publish it. Use a separate
asset prefix per producer, including its index (e.g. `s3-rbm400pf-index.json`)
to avoid collisions. Create the draft once, targeting the handoff branch.
Never overwrite an existing differing asset. If an identical asset already
exists, verify its expected size/SHA before reusing it. If upload fails,
retain local artifacts and report the exact state; do not discard training.

Commit/push only small execution metadata to `research/remote-weekend-execution`:
arrival/deadlines, observations, preparation/launch/completion receipts,
logs, export verification, archive indexes and a transfer manifest containing
release tag/asset names/sizes/fullSHA256. Push a startup checkpoint, after
each producer, and at least every two hours while training. Never amend or
rewrite coordinator/handoff history. Include actual PID/start/state/elapsed
CPU, latest full epoch, memory and disk in status; a quiet log is not failure.
No automatic restarts. Do not stop a live model merely because a polling
command times out. Verify the same handle again.

Coordinator will independently admit all returned outputs, preserve complete
paired aliases, fit weights using known ratings only, and perform the next
frozen recovered audit locally. Do not read recovered qualifying labels,
fit blends or claim a leaderboard score on the remote laptop.
