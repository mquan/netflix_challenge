# P0 production execution runbook

The user assigned the retained-bank production retrain to Codex on 2026-10-31.
This document holds protocol and verification details; read [STATUS.md](STATUS.md)
for live state. Prepared plans and engineering checks do not prove completed
production training or achievement of the quiz target.

## Managed process handles

| Queue | PID | Session |
|---|---:|---:|
| Primary GPU | 32318 (trainer 32347) | 91805 |
| CPU neural | 66919 | 50042 |
| RBM | 66932 | 98248 |
| Dependencies | 66954 | 2271 |
| Joint/integrated | 66974 | 57900 |
| Item-kNN | 66999 | 19384 |
| AutoRec GPU handoff | 67017 | 3812 |
| Bayes | 67037 | 24506 |
| GP/local | 67055 | 16829 |
| User-kNN | 67069 | 35145 |
| Final bank/blender/export | 75657 | 29645 |

These are launch handles, not perpetual proof of liveness. Inspect current `ps`,
logs and audits before acting. Do not duplicate or restart a queue because an
observation times out. Full trainers are capped at three and nominal own CPU
threads at ten. Guarded bash waiters do not represent active training jobs.

Launch records: `experiments/P0-GPU-launch-20261031.json`,
`experiments/P0-managed-queues-launch-20261031.json`, and
`experiments/P0-FINAL-launch-20261031.json`.

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
   This is planning coverage: **0/82 required production-stage audit files exist**.
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
