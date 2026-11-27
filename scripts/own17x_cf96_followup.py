#!/usr/bin/env python3
"""Select and freeze a ProbeX residual transfer, then optionally audit frozen outputs."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from audit_qualifying import scores, sha

KNOWN_ROWS = 1_128_164
QUAL_ROWS = 2_817_131
LABEL_SHA = "f52ba6e3d5a86818502d374e2a03bf95cf9b756f82d5c644450ee5171d592ec5"
ARMS = {
    "context_shift": ("context",),
    "tree_shift": ("tree_total",),
    "separate_context_and_tree": ("context", "tree_increment"),
}


def user_folds(users):
    x = users.astype(np.uint64) + np.uint64(20261116)
    x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
    return (x ^ (x >> 31)) % 5


def manifest(root, filename):
    return {
        name.removeprefix("./"): digest
        for digest, name in (line.split(maxsplit=1) for line in (root / filename).read_text().splitlines())
    }


def load_vectors(root, split, rows):
    cf = manifest(root, "cf-members.sha256")
    tree = manifest(root, "tree-members.sha256")

    def load(folder, relative, expected):
        path = root / folder / relative
        if sha(path) != expected[relative]:
            raise ValueError(f"Checksum mismatch for {path}")
        values = np.fromfile(path, dtype="<f4").astype(np.float64)
        if values.shape != (rows,) or not np.all(np.isfinite(values) & (values >= 1) & (values <= 5)):
            raise ValueError(f"Invalid prediction vector {path}")
        return values

    control, cf96, native = (
        load("cf", f"arms/{split}-{arm}.f32", cf) for arm in ("X0", "X3", "N0")
    )
    context = load("treepkg", f"blend/{split}-context.f32", tree)
    native_tree = load("treepkg", f"tree/combined-final/{split}-tree.f32", tree)
    return control, cf96, {
        "context": context - native,
        "tree_total": native_tree - native,
        "tree_increment": native_tree - context,
    }


def clipped(p):
    return np.clip(p, 1.05, 4.95)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y)**2)))


def fit(blend, residual, mask):
    fit_result = lsq_linear(blend[mask], residual[mask], bounds=(0, 1), method="bvls")
    if not fit_result.success:
        raise ValueError(fit_result.message)
    return fit_result.x


def freeze(root, probex, out):
    y = np.load(probex / "ratings.npy", allow_pickle=False).astype(np.float64)
    users = np.load(probex / "user_idxs.npy", allow_pickle=False)
    if y.shape != (KNOWN_ROWS,) or users.shape != y.shape or not np.isin(y, [1, 2, 3, 4, 5]).all():
        raise ValueError("Expected complete native ProbeX labels and user IDs")
    folds = user_folds(users)
    control, cf96, directions = load_vectors(root, "known", KNOWN_ROWS)
    masks = [folds == f for f in range(5)]
    summary = json.loads((root / "cf/arms/summary.json").read_text())

    def metrics(values):
        return {"pooled": rmse(values, y), "folds": [rmse(values[m], y[m]) for m in masks]}

    results = {"own17x": metrics(control), "cf96": metrics(cf96)}
    for name, arm in (("own17x", "X0"), ("cf96", "X3")):
        np.testing.assert_allclose(results[name]["folds"], summary[arm]["folds"], atol=4e-8, rtol=0)

    for name, keys in ARMS.items():
        design = np.column_stack([directions[key] for key in keys])
        held = np.empty_like(y)
        weights = []
        for mask in masks:
            w = fit(design, y - cf96, ~mask)
            held[mask] = clipped(cf96[mask] + design[mask] @ w)
            weights.append(w.tolist())
        result = metrics(held)
        result["fold_weights"] = weights
        result["folds_improved"] = sum(a < b for a, b in zip(result["folds"], results["cf96"]["folds"]))
        results[name] = result

    eligible = [name for name in ARMS if results[name]["folds_improved"] >= 4
                and results[name]["pooled"] < results["cf96"]["pooled"]]
    selected = min(eligible, key=lambda name: results[name]["pooled"]) if eligible else "cf96"

    qualifier_control, qualifier_cf96, qualifier_dirs = load_vectors(root, "qual", QUAL_ROWS)
    vectors = {"own17x": qualifier_control, "cf96": qualifier_cf96}
    if selected in ARMS:
        keys = ARMS[selected]
        design = np.column_stack([directions[key] for key in keys])
        weights = fit(design, y - cf96, np.ones(KNOWN_ROWS, dtype=bool))
        results[selected]["final_weights"] = dict(zip(keys, weights.tolist()))
        vectors[selected] = clipped(qualifier_cf96 + np.column_stack([qualifier_dirs[key] for key in keys]) @ weights)

    out.mkdir()
    prediction_hashes = {}
    for name, values in vectors.items():
        prediction = out / f"{name}.f32"
        values.astype("<f4").tofile(prediction)
        prediction_hashes[name] = sha(prediction)
    report = {
        "known_rows": KNOWN_ROWS,
        "fold_rows": [int(mask.sum()) for mask in masks],
        "known_ratings_sha256": sha(probex / "ratings.npy"),
        "known_users_sha256": sha(probex / "user_idxs.npy"),
        "known_comparison_is_nested": False,
        "selection_rule": "Lowest pooled RMSE among arms improving on CF96 in at least four folds and pooled, otherwise CF96",
        "selected": selected,
        "known": results,
        "qualifying_hashes": prediction_hashes,
        "qualifying_labels_used_for_fitting": False,
    }
    (out / "freeze.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def audit(out, labels_path):
    frozen = json.loads((out / "freeze.json").read_text())
    predictions = {}
    for name, expected in frozen["qualifying_hashes"].items():
        path = out / f"{name}.f32"
        if sha(path) != expected:
            raise ValueError(f"Frozen prediction changed: {path}")
        predictions[name] = np.fromfile(path, dtype="<f4").astype(np.float64)

    if sha(labels_path) != LABEL_SHA:
        raise ValueError("Unexpected recovered-label checksum")
    with gzip.open(labels_path, "rt") as stream:
        if stream.readline().strip() != "rating,is_test":
            raise ValueError("Unexpected recovered-label schema")
        labels = np.loadtxt(stream, delimiter=",", dtype=np.int16)
    if labels.shape != (QUAL_ROWS, 2) or np.bincount(labels[:, 1]).tolist() != [1_408_789, 1_408_342]:
        raise ValueError("Unexpected recovered group sizes")
    report = {}
    for name, values in predictions.items():
        result = scores(values, labels[:, 0], labels[:, 1])
        report[name] = {"source0": result["quiz"], "source1": result["test"]}
    with (out / "audit.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    mode = parser.add_subparsers(dest="mode", required=True)
    freeze_args = mode.add_parser("freeze")
    freeze_args.add_argument("root", type=Path)
    freeze_args.add_argument("probex", type=Path)
    audit_args = mode.add_parser("audit")
    audit_args.add_argument("labels", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.root, args.probex, args.out)
    else:
        audit(args.out, args.labels)


if __name__ == "__main__":
    main()
