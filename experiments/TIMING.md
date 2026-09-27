# Score timeline and activity time

Refresh the plot-ready data with no extra dependencies:

```sh
python3 scripts/score_timeline.py
```

Output: `experiments/score_timeline.csv`. Use `timestamp_utc` or
`hours_since_reference` for X, `score` for Y, and `metric` for separate series.
`best_so_far_in_metric` gives the descending frontier within each metric.
Lower RMSE is better. Strategies and source references accompany every row.

## Timing evidence

The available conversation text has no per-message timestamps. We do not
invent those timestamps or evenly distribute milestones along the clock.

| `timestamp_basis` | Meaning |
|---|---|
| `git_commit_upper_bound` | The score was already documented in this commit. It was achieved no later than that time, possibly substantially earlier. Several early milestones share a timestamp. |
| `log_mtime_proxy_upper_bound` | The old completed run had no per-line timestamps. Its last log-write time is a filesystem-based proxy for when its best result was already available, not the precise winning epoch time. This evidence is frozen in `timing_backfill.json` and bound to the log hash; copying files can otherwise change mtimes. |
| `stdout_observed` | The updated experiment runner recorded UTC when it received this score line. Stdout buffering can delay receipt; it is not a claim about the exact instant the model computed the score. |

For runs, `earliest_utc` gives launch/start evidence. Historical commit rows
have no known lower bound. The relative-time origin is the **first documented
milestone**, not the start of the project, data download, or total engineering
effort. Treat approximate X coordinates as bounds, not precise speed claims.
File-time proxies cannot recover the old exact timing even if displayed with
microseconds. A chart should identify these approximate points explicitly.

For future runs, `scripts/experiment.py` writes timestamped `.events.jsonl`
files plus manifest `launched_utc` and `finished_utc`. The exporter includes
each new best score from those events. For old runs it exports only the best
score after completion, not fabricated epoch times. Interrupted completed
logs may be included with `stopped_or_failed` status; that does not certify a
valid saved checkpoint. Graph construction has no RMSE and is omitted.

Keep `single_model_probe`, `blend_probe_fit` and `blend_probe_user_cv` separate.
All are development scores; even user-fold blending uses base models whose
epochs were selected on the full probe. None is an untouched test score.

## Subtracting idle and rate-limit time

`activity_intervals.csv` is the evidence ledger. It starts empty because we
cannot reliably reconstruct historical work/idle intervals. Add intervals
only when observed or imported from timestamped runtime/transcript events:

```csv
start_utc,end_utc,state,source
2026-09-28T10:00:00+00:00,2026-09-28T10:05:00+00:00,working,example training process trace
2026-09-28T10:05:00+00:00,2026-09-28T10:07:00+00:00,rate_limit,example platform pause event with no background work
```

These are format examples, **not actual recorded intervals**. Every row needs
timezone-aware start/end timestamps and an evidence/source description. States:

- `working`: observed useful work, including training while the assistant waits.
- `idle`: no useful work, such as waiting for a user to resume.
- `rate_limit`: an observed platform rate-limit pause.
- `timeout`: an observed stalled/timeout period, not a guess based on silence.

The exporter unions overlapping intervals, clips them to the plotted window,
and gives working intervals precedence over simultaneous pause intervals.
Thus two parallel training runs do not count twice, and an assistant rate
limit is not subtracted when another recorded process is doing useful work.
Work intervals must exclude known host suspension/stalls; process lifetime
alone does not establish continuous work. This is elapsed useful-work time,
not summed CPU/GPU time or API cost.

CSV time fields:

- `calendar_elapsed_seconds`: unadjusted elapsed wall-clock time.
- `recorded_work_seconds`: union of recorded useful-work intervals.
- `recorded_pause_seconds`: union of recorded pauses, excluding overlapping work.
- `unclassified_seconds`: time with no work/pause evidence.
- `elapsed_minus_recorded_pauses_seconds`: calendar time minus known pauses.
- `activity_coverage`: `complete` only if no unclassified time remains.

Until coverage is complete, adjusted elapsed time **includes unknown time**;
it must not be presented as measured active-work duration. With full coverage,
adjusted elapsed equals recorded work. Timestamp uncertainty is separate from
coverage: complete activity records do not make an old milestone's date exact.

The local experiment runner cannot observe Codex platform rate limits, user
inactivity, or every computer sleep event. These need explicit runtime events
or manual/imported intervals. Existing long log gaps are not automatically
classified as pauses. `elapsed_seconds` in old run manifests is not enough
to infer them, and concurrent run durations must not simply be added.
