"""Prove splits and the pretraining list are disjoint; exit non-zero loudly on any violation."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data

TWIN_FIG = Path("results/rotated_twins.png")
TWIN_JSON = Path("results/rotated_twins.json")
STRATA_D8 = 40   # near / clean test strata threshold (grouping used 24)
SPLITS = ("train", "val", "test")
CHECK_COLUMNS = ["path", "md5", "group_id"]
PHASH_MAX_DIST = 24


def report(name, count, results):
    """Print one check line; a check passes only when count is zero."""
    status = "PASS" if count == 0 else "FAIL"
    print(f"  {status}  {name:<46} {count}")
    results.append({"check": name, "count": int(count), "status": status})


def check_key_disjoint(splits, key, results):
    """No value of key (path, md5, group_id) may appear in two splits."""
    for a, b in itertools.combinations(SPLITS, 2):
        shared = set(splits[a][key]) & set(splits[b][key])
        report(f"{key} shared {a}/{b}", len(shared), results)


def check_phash_disjoint(splits, results):
    """Zero cross-split pairs at 256-bit pHash Hamming <= 24 under any of 8 orientations (splits/hashes.npz)."""
    cache = np.load(data.HASH_CACHE, allow_pickle=False)
    row_of = {p: i for i, p in enumerate(cache["paths"])}
    missing = sum(p not in row_of for s in SPLITS for p in splits[s]["path"])
    report("split paths missing from hashes.npz", missing, results)
    d8 = {s: cache["h256_d8"][[row_of[p] for p in splits[s]["path"] if p in row_of]] for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        dist = data.d8_distance(d8[a], d8[b])        # min over 8 orientations (rotations and mirrors)
        report(f"8-orient pHash pairs <= {PHASH_MAX_DIST} {a}/{b}", int((dist <= PHASH_MAX_DIST).sum()), results)


def check_pretrain(pretrain, splits, results):
    """The pretraining list must be a subset of train.csv with no val or test path."""
    paths = set(pretrain["path"])
    print(f"  pretrain list: {len(paths)} paths")
    report("pretrain paths not in train", len(paths - set(splits["train"]["path"])), results)
    report("pretrain paths in val", len(paths & set(splits["val"]["path"])), results)
    report("pretrain paths in test", len(paths & set(splits["test"]["path"])), results)
    check_pretrain_vs_test_phash(sorted(paths), splits["test"]["path"], results)


def check_pretrain_vs_test_phash(pretrain_paths, test_paths, results):
    """Direct check: pretrain list vs test at 8-orient pHash <= 24, not inferred through train.csv."""
    cache = np.load(data.HASH_CACHE, allow_pickle=False)
    row_of = {p: i for i, p in enumerate(cache["paths"])}
    missing = [p for p in pretrain_paths if p not in row_of]
    report("pretrain paths missing from hashes.npz", len(missing), results)   # unhashed = unchecked
    pre = cache["h256_d8"][[row_of[p] for p in pretrain_paths if p in row_of]]
    test = cache["h256_d8"][[row_of[p] for p in test_paths]]
    dist = data.d8_distance(pre, test)
    report(f"8-orient pHash pairs <= {PHASH_MAX_DIST} pretrain/test", int((dist <= PHASH_MAX_DIST).sum()), results)


def rotated_twin_matches(splits, max_dist=PHASH_MAX_DIST):
    """For each val/test image: min Hamming over its 8 orientations to any train image (identity hash)."""
    cache = np.load(data.HASH_CACHE, allow_pickle=False)
    row_of = {p: i for i, p in enumerate(cache["paths"])}
    d8 = cache["h256_d8"]
    train_paths = list(splits["train"]["path"])
    train = d8[[row_of[p] for p in train_paths], 0].astype(np.int32)          # train at identity
    rows = []
    for split in ("val", "test"):
        for p in splits[split]["path"]:
            q = d8[row_of[p]].astype(np.int32)                                 # (8, 256)
            dist = q @ (1 - train).T + (1 - q) @ train.T                       # (8, n_train)
            o, t = np.unravel_index(np.argmin(dist), dist.shape)               # ties -> lowest orientation
            rows.append({"split": split, "path": p, "twin": train_paths[t], "orientation": int(o),
                         "dist": int(dist[o, t]), "dist_identity": int(dist[0].min())})
    df = pd.DataFrame(rows)
    df["twin_found"] = df["dist"] <= max_dist
    return df


def save_twin_figure(matches, path=TWIN_FIG, n_pairs=3):
    """Val/train pairs only (test pixels are never loaded here): image, twin, twin rotated to match."""
    pairs = (matches[(matches["split"] == "val") & matches["twin_found"] & (matches["orientation"] > 0)]
             .sort_values(["dist", "path"]).drop_duplicates("twin").head(n_pairs))
    if pairs.empty:
        print("  no val/train rotated twins to plot")
        return
    fig, axes = plt.subplots(len(pairs), 3, figsize=(10, 3.4 * len(pairs)), squeeze=False)
    for r, row in enumerate(pairs.itertuples()):
        val_img, twin = data.resize_paths([row.path, row.twin], px=256)
        k, m = data.ORIENTATIONS[row.orientation]
        # val oriented by (k, m) matches the twin, so the twin un-oriented matches the val image
        tiles = [(val_img, f"VAL {Path(row.path).stem}"),
                 (twin, f"TRAIN twin {Path(row.twin).stem}"),
                 (data.unorient(twin, k, m), f"twin, undo {data.ORIENTATION_NAMES[row.orientation]}\n"
                                             f"Hamming {row.dist} (identity {row.dist_identity})")]
        for c, (img, title) in enumerate(tiles):
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(title, fontsize=12)
            axes[r, c].axis("off")
    fig.suptitle("Rotated twins across splits (256-bit pHash <= 24 after rotation/mirror)", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path} ({len(pairs)} val/train pairs)")


def report_rotated_twins():
    """Count val/test images with a rotated or mirrored train twin, by matching orientation. Report only."""
    splits = {s: data.load_split(s, allow_test=True, columns=CHECK_COLUMNS) for s in SPLITS}
    m = rotated_twin_matches(splits)
    found = m[m["twin_found"]]
    table = (found.assign(orientation=[data.ORIENTATION_NAMES[o] for o in found["orientation"]])
             .pivot_table(index="orientation", columns="split", values="path", aggfunc="count", fill_value=0)
             .reindex(index=data.ORIENTATION_NAMES, columns=["val", "test"], fill_value=0))   # zero twins -> zeros
    table.loc["TOTAL with twin"] = table.sum()
    table.loc["split size"] = m.groupby("split").size()
    print(f"\nrotated twins: val/test images with min Hamming <= {PHASH_MAX_DIST} to any train image "
          f"over 8 orientations (current split)")
    print(table.to_string())
    print(f"  distinct train images acting as twins: {found['twin'].nunique()}")
    summary = {"max_hamming": PHASH_MAX_DIST, "orientations": data.ORIENTATION_NAMES,
               "counts": {s: {o: int(table.loc[o, s]) if s in table else 0 for o in table.index}
                          for s in ("val", "test")},
               "distinct_train_twins": int(found["twin"].nunique())}
    TWIN_JSON.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {TWIN_JSON}")
    save_twin_figure(m)
    return m


def preregister_test_strata(threshold=STRATA_D8, choices_path=Path("results/frozen_choices.json")):
    """Store the near / clean test-strata RULE only; eval.py computes the lists before reading any test label."""
    strata = {
        "rule": ("near = a test image whose nearest train image by random-init feature cosine is within "
                 f"8-orientation 256-bit pHash distance <= {threshold}; clean = every other test image."),
        "threshold": threshold,
        "encoder": "resnet18(weights=None), torch.manual_seed(42), fc = Identity, eval mode",
        "features": "512-d encoder output, 96 px grayscale, fixed normalization (0.5, 0.5), no augmentation",
        "nearest_train": "argmax cosine similarity over ALL train.csv images (866), not a label budget",
        "distance": "min Hamming over 8 orientations, symmetric (data.d8_distance), from splits/hashes.npz",
        "when": ("eval.py computes the lists from test images and train features BEFORE it reads any test "
                 "label, and saves them to results/test_strata.json"),
        "validation_evidence": ("VAL, same rule: near 35 images at 1-NN acc 1.000, clean 139 at 0.799. "
                                "Replaced the hash-only rule (min distance to any train image), which did not "
                                "separate val: near 124 at 0.847, clean 50 at 0.820."),
    }
    choices = json.loads(choices_path.read_text())
    choices["test_strata"] = strata          # replaces any earlier strata, including the hash-only lists
    choices_path.write_text(json.dumps(choices, indent=2))
    print(f"test strata rule pre-registered in {choices_path} (threshold {threshold}); no test lists stored")


def main():
    """Run all checks, print counts, exit 1 on any FAIL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", default="splits/pretrain.csv", help="pretraining list CSV (path column)")
    parser.add_argument("--rotations", action="store_true", help="report rotated/mirrored twins across splits")
    parser.add_argument("--strata", action="store_true", help="pre-register the near/clean test-strata rule (text only)")
    args = parser.parse_args()
    if args.rotations:
        report_rotated_twins()
        return
    if args.strata:
        preregister_test_strata()
        return

    # Per CLAUDE.md, only these columns are read from any split, so no label or pixel of test is touched.
    splits = {s: data.load_split(s, allow_test=True, columns=CHECK_COLUMNS) for s in SPLITS}
    print("split sizes: " + ", ".join(f"{s}={len(splits[s])}" for s in SPLITS))
    results = []
    for key in ("path", "md5", "group_id"):
        check_key_disjoint(splits, key, results)
    check_phash_disjoint(splits, results)
    check_pretrain(pd.read_csv(args.pretrain), splits, results)

    n_fail = sum(r["status"] == "FAIL" for r in results)
    # Saved for report.ipynb (which reads only results/): one file per pretraining list checked.
    stem = Path(args.pretrain).stem
    out = Path("results") / ("check_splits.json" if stem == "pretrain" else f"check_splits_{stem}.json")
    out.write_text(json.dumps({"pretrain_list": str(args.pretrain), "split_sizes": {s: len(splits[s]) for s in SPLITS},
                               "n_checks": len(results), "n_fail": int(n_fail), "checks": results}, indent=2))
    if n_fail:
        print(f"\nFAIL: {n_fail} of {len(results)} checks failed. Pretraining list {args.pretrain} is NOT clean.")
        sys.exit(1)
    print(f"\nPASS: all {len(results)} checks passed for {args.pretrain}.")


if __name__ == "__main__":
    main()
