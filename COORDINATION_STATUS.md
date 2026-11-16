# Reproduction coordination

Local machine: {"complete": true, "state": "assigned-work-complete", "active": null}.

Upstream worker: {"state": "running", "active": null, "completed": 4, "commit": "01f3f84fb6d080f5086a7c49c7df509b4c0ad173", "error": null}.

Fresh-only development blends: waiting for all eight fresh model artifacts.

The 0.856167 released-prediction reference is not a fresh-training result.

| Model | Fresh development RMSE | Released fixture RMSE |
| --- | ---: | ---: |
| tsvdx4-60 | 0.879210416 | 0.879210415 |
| tsvdx4-60__knns | 0.879095602 | 0.878952653 |
| rbmx2-40 | 0.921512211 | unavailable |
| rbmx2-40__knn3 | 0.903717708 | 0.903673503 |
