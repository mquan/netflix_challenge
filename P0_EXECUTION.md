# P0 production execution runbook

The user assigned the retained-bank production retrain to Codex on 2026-10-31.
This document holds protocol and verification details; read [STATUS.md](STATUS.md)
for live state. Prepared plans and engineering checks do not prove completed
production training or achievement of the quiz target.

## Managed process handles — recovered 2026-11-07

Prior handles are historical launch evidence: all were absent at recovery.
Current detached runners and trainer handles are recorded in
`experiments/recovery-20261107/launch.json` and STATUS.md.

| Queue | PID |
|---|---:|
| Primary GPU | 71725 (A2f trainer 66571; A2h audited) |
| CPU neural | 71728 |
| RBM | 71730 |
| Dependencies | 71732 |
| Joint/integrated | 71734 |
| Item-kNN | 71736 |
| AutoRec GPU handoff | 71738 |
| Bayes | 71740 |
| GP/local | 71742 |
| User-kNN | 71744 |
| Final bank/blender/export | 71746 |
| Idle-sleep assertion bound to final runner | 71758 |

Recovery verified 621 distinct frozen inputs. Interrupted files/logs are
preserved in `artifacts/production/P0-20261031/interrupted-20261107/`.
A2h restarts the exact original 12-epoch recipe; the old epoch-5 checkpoint
has weights but no optimizer/RNG state, so exact trajectory continuation is
unavailable. S0 resumes with unchanged interrupted n16/n32 recipes and remaining
lane commands. CPU/report copies replace stale numeric PID guards with original
lane process-name signatures. No frozen sources, budgets, audits or gates changed.

These handles are not perpetual proof of liveness. Inspect current `ps`, logs
and audits before acting. Do not duplicate queues on an observation timeout.
Full trainers remain capped at three with about ten CPU threads maximum.
`caffeinate -i -w 71746` prevents idle sleep while final runner lives.

Original launch records remain preserved as historical evidence. The frozen
phase details below describe original protocols; references to original handles
are superseded by the recovery record and STATUS.md.

## Frozen phases and evidence

All bare `P0-*.json` references below live in `experiments/`.

Monitor the live production queues and audit every completed fixed-budget stage.
   All 79 columns/supports and the final fitted blender remain required.
   Managed queues launched and waiting behind these guards:
   - CPU neural: seven fixed recipes, waits for a free S0 slot and normal memory.
     `P0-CPU-neural-20261031-plan.json`.
   - RBM: 12 stages / nine retained columns, waits for both S0 lanes and all
     seven audited CPU neural stages. `P0-RBM-20261031-plan.json`.
   - Dependencies: baseline, graph256, exact graph64 prefix, learned neighborhood;
     waits for both S0 lanes. One thread; combined planned maximum is ten.
     `P0-DEPS-20261031-plan.json`: 263 inputs / 264 checksum checks pass;
     five tests and eight actual native fixture handoffs pass, including query-label isolation.
   - Joint/integrated: 12 stages / 13 columns, one thread; waits for both S0 lanes
     and all four dependency-stage audits. `P0-JOINT-20261031-plan.json`:
     297 inputs / 298 checksum checks pass; six audit/resource tests pass.
     J4 restarts epoch numbering; integrated resumes require audited parent sidecars.
   - Item-kNN: 28 recipes / 29 columns and six support aliases; waits for all
     16 joint/dependency audits. `P0-KNN-20261031-plan.json`: 278 inputs /
     279 checksum checks pass; ten audit/resource/handoff tests and all 28 native
     label-isolation fixtures pass. Ordered alias manifest waits for all outputs.
   - AutoRec: two MPS trajectories, epochs 1 and 44 with snapshot14; requires all
     seven primary GPU audits, primary process exit and a successful MPS parity check.
     `P0-AUTOREC-20261031-plan.json`: 255 inputs / 256 checksum checks pass.
     Both budgets and 512-unit/two-layer architecture pass tiny CPU training/reload audits;
     actual MPS prepare rejects the absent MPS-specific receipt. Waiting for primary GPU completion.
   - Bayes: four stages / three columns, waits for all 12 RBM audits and S0,
     CPU neural and RBM process exit, plus normal memory. Exact four/five-thread
     settings preserved; a tiny historical-thread test must pass before training.
     `P0-BAYES-20261031-plan.json`: 249 inputs / 250 checksum checks pass;
     five audit/resource tests, eight single-thread native fixture handoffs and
     all three full historical checkpoint scans pass. B4 resumes the newly
     produced sweep-32 parent, retaining 144 samples at sweep 160. Queue waiting;
     historical-thread runtime test remains pending. `P0-BAYES-preflight-20261031.json`.
   - GP/local: four invocations / six columns, waits for all 32 Bayes/item-kNN
     audits, their process exit and normal memory; consumes newly audited J3.
     `P0-KERNEL-20261031-plan.json`: 252 inputs / 253 checksum checks pass;
     five audit/resource tests and eight actual kernel fixture handoffs pass.
     Complete GP exports and all six local epochs repeat exactly; query-label
     isolation passes. Missing J3 audit rejects prepare before creating output.
     Queue waiting. `P0-KERNEL-preflight-20261031.json`.
   - User-kNN: new K1 rank256/41 epochs at six threads, then original retained
     inference binary at three threads; waits for all four kernel audits and CPU
     process exit. One prediction plus two support aliases. `P0-USERKNN-20261031-plan.json`:
     250 inputs / 251 checksum checks and five audit/resource tests pass. Ten
     native fixture commands cover full one-thread K1 budgets, exact original/
     rebuilt inference, label isolation and nonzero support/corrections. Queue
     waiting. `P0-USERKNN-preflight-20261031.json`.
   Frozen plans cover all **79/79 columns plus eight support aliases**, verified
   in original blender order by `P0-all-family-planned-coverage-20261031.json`.
   Latest runtime (2026-11-08): **3/82 required production-stage audits exist** (A2h, MH3, MT0), independently checked against required metadata and output hashes.
   Full outputs and final fitted blender/export remain outstanding. All nine
   family runners are live at guards; only primary GPU and both S0 lanes are
   currently training. Prelaunch verified 486 distinct frozen inputs and an
   acyclic runner-dependency graph. PIDs/sessions and initial waiting state are
   recorded in `P0-managed-queues-launch-20261031.json`; never launch duplicates.
   The separate latent basis remains frozen.
   Final bank resolver now requires all 82 matching audits and all bound output
   hashes, preserves original order/settings/ready latent+metadata files, and
   fills exactly 79 predictions/eight support aliases. Five synthetic complete-
   bank contract tests pass; real incomplete production bank rejects without
   creating `resolved-bank.json`. Engineering only, not production completion.
   `P0-bank-resolver-verification-20261031.json`. Final fit/export runner is launched and guarded.
   Submission serialization passes five tests, including text/float32 roundtrip
   and rejection of changed identities, labels, counts or bounds. Full real-data
   order check verifies 2,817,131 queries and 17,470 movie headers against raw
   qualifying IDs/dates, dense-user mapping and production rows. No submission
   written; the tested model/inference handoff is queued behind all production audits.
   `P0-submission-order-verification-20261031.json`.
   Final handoff now binds bank/model hashes before each command, audits fixed
   budgets and outputs afterward, requires two exact inference runs, then binds
   raw-order text export. Four tests pass with real tiny CLI fitting/inference/
   export and synthetic production receipts. Real final fit rejects the absent
   resolved bank before creating started metadata. Full fixed-budget final
   runner is frozen and launched as a waiting bash process. `P0-FINAL-20261031-plan.json`: 500
   inputs / 501 checksum checks pass; one resource test plus the prior 14
   resolver/handoff/serialization tests pass. Preserves 79/64/5 inputs and
   tree 904/969/1106, NN 15/14/14 budgets with seeds 2026/7/13. Guard requires
   all 82 audits, queue/stage process exit and stable normal memory, then runs
   resolve → fit → infer → replay → export. `P0-FINAL-preflight-20261031.json`.
   PID 75657/session 29645 verified live at its guard; all five final outputs
   absent and still 0/82 production-stage audits. `P0-FINAL-launch-20261031.json`.
   AutoRec's historical provenance limitations remain explicit; MPS engineering verification
   is still pending. Recent logs and active workers confirm all three live jobs continue.
   Full production outputs remain required; the development gap is unchanged.

## Completion requirements

- All 82 real fixed-budget stage audits must be complete, matching their frozen
  plans, datasets and consumed production parents. Engineering fixtures do not
  satisfy this condition.
- Resolve all 79 prediction occurrences and eight support aliases in the
  retained blender order, preserving separately frozen metadata and latent basis.
- Fit only original development labels at the frozen median budgets.
- Audit both complete qualifying inference runs against the same bank and model;
  all four inference output arrays must repeat exactly.
- Verify all 2,817,131 qualifying identities/dates and raw movie-header order,
  finite bounded ratings, submission text and complete provenance.
- A verified production export alone does not prove quiz 0.8554 or development
  CV 0.8628. No qualifying labels or measured hidden score are available.

## Runtime review — 2026-11-08

The earlier 1–1.5 week estimate was an uncalibrated planning allowance, not a
measured critical path. Read-only review bound 90 source files, verified retained
recipe record hashes, and recovered historical timings. Full evidence and
explicit assumptions: `experiments/runtime-audit-20261108/evidence.json`.
No running job, frozen plan, thread count, budget, output or audit was changed.

| Work | Evidence-based scale | Limitation |
|---|---:|---|
| Bayes B2/B3/B4 | 4.22 / 12.47 / 19.15 h; 35.84 h serial | Original-data runs, mixed prior contention; production unmeasured |
| RBM 12 stages | 21.56 h | Sum of fixed epoch counts × retained-epoch timings, not measured stage totals |
| Joint/integrated | 3.86 h | Same epoch extrapolation; resource overlap matters |
| Item-kNN 28 unique runs | 0.254 h historical; ≈1.02 h for double queries and two passes | Assumes linear query scaling |
| Remaining GPU neural | About 20 h at historical runtimes, before slowdown allowance and AutoRec | Actual A2h was 9.18 h vs historical 4.89 h; A2f ≈17 min/epoch currently |
| Final fit/replay/export | 2 h scenario allowance | Not benchmarked end to end |

The simple remaining CPU-path scenario is about **63.9 h**: assumed CPU tail
3 h → RBM 21.56 h → Bayes 35.84 h → user-kNN tail allowance 1.5 h → final
allowance 2 h. Dependency/joint/item-kNN work overlaps this path in the existing
schedule. GPU is a parallel path and must be accounted for using the maximum,
not added to CPU time. Thermal slowdown, memory pressure, MPS parity failure and
final fitting remain uncertainties. **Rough current-schedule range: 3–5 days;
optimized target: 48–72 h, unvalidated.** This is full production completion,
not reaching the quiz target.

Recommended changes, in priority order:

1. **Concurrent independent Bayes chains.** Bayes consumes production data and
   its own audited B4 parent, not RBM outputs. Its wait for all RBM audits is a
   resource policy. Put B4's 32+128 sweep chain on one lane, B3 then B2 on the
   other, retaining their exact four/five-thread recipes. Ideal Bayes wall time
   falls from 35.84 to max(19.15, 12.47+4.22)=19.15 h: **16.69 h saving**.
   Two four-thread jobs plus GPU host need nine nominal CPU threads; the
   five-thread B4 resume plus four-thread B3 and GPU host need ten. This uses
   all three full-job slots; native/CPU neural jobs must then wait. No assumed
   linear speedup: if simultaneous runs slow by 50%, Bayes becomes ≈28.7 h
   and the saving is only ≈7.1 h. Measure before retaining overlap.
2. **Stage-level resource scheduling.** Overlap four-thread RBM stages with
   one four/five-thread Bayes job plus the GPU host (nine/ten threads, three
   jobs). Other RBM stages use six/eight threads and cannot be indiscriminately
   added. Reordering preserves every original parent edge and each stage's
   command. Possible additional savings of several hours are not credited in
   the 47.2 h candidate scenario.
3. **Release actual data dependencies.** GP/local audits require J3, not Bayes
   or item-kNN products. K1 can train from production data, and user-kNN only
   requires the new audited K1; neither consumes GP/local output. Prefer early
   J3 and fill suitable resource slots with these tasks. Resource reservations
   must replace the released family barriers; simply deleting waits is unsafe.
4. **Keep one GPU trainer and moonshot on hold.** Avoid known GPU contention.
   Do not apply new padding/checkpointing to the frozen retained recipes in
   flight: reduced memory is promising but padded dropout/RNG layout and
   training trajectory need separate compatibility evidence. No runtime saving
   from that code is counted here.

The conservative candidate arithmetic, with only ideal Bayes overlap credited,
is **47.2 h remaining**, including explicit 3/1.5/2 h allowances. This is a
scenario, not a delivery commitment. Peak memory, effective throughput and the
GPU branch must support it. Maintain 79 columns/eight aliases, all 82 audits,
all fixed budgets, both exact inference passes and full raw-order export.
Repeated 3.32 GiB hash checks are not the multi-day bottleneck; keep them.

Compatibility and verification:

- Scheduling-only proposals preserve bank coverage, fixed commands, thread
  counts, parent audits, ordering at final resolution, and final export.
  Training is already nondeterministic for historical Hogwild recipes; a timing
  change does not imply bit-identical retraining. No change has been deployed.
- Before rollout, freeze a separate scheduler/launch manifest, bind every
  original command and audit, verify the graph is acyclic, and test the resource
  scheduler against simultaneous readiness, absent/invalid receipts, existing
  output directories, producer liveness and warning memory pressure. Ensure
  no more than three full jobs or ten nominal CPU threads.
- Handoff only at stage boundaries; replace only verified idle own waiters.
  Active trainers and other-session files/jobs remain untouched. Original
  frozen plans stay immutable so their receipt hashes remain valid.
- Measure per-sweep/per-epoch rates and process peak memory. Admit a second
  Bayes process only with measured headroom, normal memory and a complete
  resource reservation. Retain overlap only when aggregate progress improves.
- Keep existing tiny historical-thread checks and parent/reload/label-isolation
  audits. Validate stage command equality and no duplicate output destinations.
  All original final completion requirements still apply.

Skill retro: no updates to the optimize-performance skill are suggested.
