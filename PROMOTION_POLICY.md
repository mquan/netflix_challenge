# Development promotion policy, revised November 18, 2026

A completed candidate can be promoted when its overall held-row RMSE improves
its comparable documented control. Improving every fold and clearing a fixed
0.0003 RMSE gain are no longer mandatory. Fold changes, uncertainty, evaluation
rows and verification strength remain visible. A small measured gain is not a
claim that future unseen performance must improve.

This is a retrospective change requested after reviewing the earlier policy.
Original experiment reports retain their original decisions. The dated amendment
in `experiments/promotion-policy-revision-20261118.json` records the revised
labels and binds each historical source by hash.

| Candidate | Recorded overall RMSE | Revised status |
| --- | ---: | --- |
| R10 B5 | 0.8668212 | Promoted; documented original-Probe record at that time |
| HR2a | 0.8668444071 | Promoted candidate; improves its R9 control |
| HR2b | 0.8668187324 | Promoted; new original-Probe record at that time |
| LC6 | 0.8668264382 | Promoted candidate; improves its R9 control |
| N1 full | 0.8667006277 | Promoted; independently scored original-Probe result |
| H8 full | 0.8664848395 | Promoted; independently scored matched comparison |

LC6 follows HR2b in the recorded chronology and has a slightly higher RMSE, so
it is a promoted candidate rather than a new overall record. HR2a similarly
does not beat the preceding B5 result. The plots show both candidates and draw
the record line only through successive lower scores. B5's score is reported
to seven decimal places in its source; this amendment does not add precision.

Scores on original Probe, the current known-ProbeX reproduction, and recovered
qualifying groups remain distinct evaluation series. Their gains cannot be
added or interpreted as improvements on one common test set. These retrospective
labels do not establish independent Test performance or satisfy the active
fresh-only qualifying target. No qualifying outcomes were used for this revision.
