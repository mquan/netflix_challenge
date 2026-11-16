# Reproduction coordination

Local machine: {"complete": true, "state": "assigned-work-complete", "active": null}.

Upstream worker: {"state": "complete", "active": null, "completed": 8, "commit": "db394c1125068de57b317f99b762bbb14c0b42f2", "error": null}.

Fresh-only development blends: complete.

The 0.856167 released-prediction reference is not a fresh-training result.

| Model | Fresh development RMSE | Released fixture RMSE |
| --- | ---: | ---: |
| tsvdx4-60 | 0.879210416 | 0.879210415 |
| tsvdx4-60__knns | 0.879095602 | 0.878952653 |
| rbmx2-40 | 0.921512211 | unavailable |
| rbmx2-40__knn3 | 0.903717708 | 0.903673503 |
| dnn-24 | 0.917593471 | 0.917593471 |
| dnn-24__knn3 | 0.907716749 | 0.907716749 |
| dnn-24__dnn-16p50 | 0.907173046 | 0.907173046 |
| dnn-24__dnn-16p50__knn3 | 0.896837159 | 0.896837159 |
