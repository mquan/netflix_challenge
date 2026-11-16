# Reproduction coordination

Local machine: {"complete": true, "state": "assigned-work-complete", "active": null}.

Upstream worker: {"state": "running", "active": null, "completed": 7, "commit": "c408d3998a9c15986f6ffb57679b2c13b9db4e7e", "error": null}.

Fresh-only development blends: waiting for all eight fresh model artifacts.

The 0.856167 released-prediction reference is not a fresh-training result.

| Model | Fresh development RMSE | Released fixture RMSE |
| --- | ---: | ---: |
| tsvdx4-60 | 0.879210416 | 0.879210415 |
| tsvdx4-60__knns | 0.879095602 | 0.878952653 |
| rbmx2-40 | 0.921512211 | unavailable |
| rbmx2-40__knn3 | 0.903717708 | 0.903673503 |
| dnn-24 | 0.917593471 | 0.917593471 |
| dnn-24__dnn-16p50 | 0.907173046 | 0.907173046 |
| dnn-24__dnn-16p50__knn3 | 0.896837159 | 0.896837159 |
