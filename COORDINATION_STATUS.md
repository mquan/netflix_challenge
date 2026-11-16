# Reproduction coordination

Local machine: {"complete": true, "state": "assigned-work-complete", "active": null}.

Upstream worker: {"state": "running", "active": null, "completed": 5, "commit": "bf526a45bf0c89004f8b99f8df3fc18db2ccf5b9", "error": null}.

Fresh-only development blends: waiting for all eight fresh model artifacts.

The 0.856167 released-prediction reference is not a fresh-training result.

| Model | Fresh development RMSE | Released fixture RMSE |
| --- | ---: | ---: |
| tsvdx4-60 | 0.879210416 | 0.879210415 |
| tsvdx4-60__knns | 0.879095602 | 0.878952653 |
| rbmx2-40 | 0.921512211 | unavailable |
| rbmx2-40__knn3 | 0.903717708 | 0.903673503 |
| dnn-24 | 0.917593471 | 0.917593471 |
