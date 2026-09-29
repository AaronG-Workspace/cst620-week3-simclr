"""Score the frozen encoders once on the clean test split (or on val with --dry-run)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_score

import baselines as bl
import data

MANIFEST = Path("results/frozen_manifest.json")
CHOICES = Path("results/frozen_choices.json")
LOCK = Path("results/eval.lock")
CONTAM_WEIGHTS = Path("weights/ssl_encoder_contaminated.pt")   # CONTAMINATED control only, not in the manifest
BUDGETS = ("10", "25", "50", "all")
N_BOOT, BOOT_SEED, GROUP_SEED = 1000, 42, 42
METHODS = ("SSL k-NN", "SSL probe", "random k-NN", "random probe", "supervised")
ENC_NAME = {"ssl": "SSL", "random": "random"}


# ---------- 1. integrity ----------

def verify_manifest():
    """Recompute every SHA-256 in frozen_manifest.json; refuse on any missing file or mismatch."""
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for name, entry in manifest["files"].items():
        path = Path(entry["path"])
        if not path.exists():
            bad.append(f"{name}: missing {path}")
        elif bl.sha256_of(path) != entry["sha256"]:
            bad.append(f"{name}: sha256 changed for {path}")
    if bad:
        raise SystemExit("REFUSING TO RUN: frozen artifacts changed since the freeze\n  " + "\n  ".join(bad))
    print(f"manifest OK: {len(manifest['files'])} files match the freeze of {manifest['frozen_at_utc']}")
    return manifest


# ---------- 2. features and strata (no labels read) ----------

def load_target(split):
    """Paths, group ids and 96 px images of the target split; the label column is NOT read here."""
    df = data.load_split(split, allow_test=(split == "test"), columns=["path", "group_id"])
    return df["path"].to_numpy(), df["group_id"].to_numpy(), data.resize_paths(df["path"])


def supervised_model(path, trunk_only=False):
    """Frozen supervised ResNet-18; trunk_only drops the 2-class head to expose 512-d features."""
    model = bl.build_supervised()
    model.load_state_dict(torch.load(path, weights_only=True))
    if trunk_only:
        model.fc = nn.Identity()
    return model.eval()


def extract_all(target_u8, ssl_path=bl.SSL_WEIGHTS):
    """Frozen train and target features for the SSL and random-init encoders (train images from train.csv)."""
    train = data.load_split("train")
    train_u8 = data.resize_paths(train["path"])
    feats = {}
    for name, enc in (("ssl", bl.ssl_encoder(ssl_path)), ("random", bl.random_init_encoder(seed=42))):
        feats[name] = {"train": bl.extract_features(enc, train_u8), "target": bl.extract_features(enc, target_u8)}
    return feats, train


def compute_strata(feats, train_paths, target_paths, rule):
    """near = nearest train image by random-init cosine is within 8-orient pHash <= threshold; clean = rest."""
    nn_idx = bl.cosine_sim(feats["random"]["target"], feats["random"]["train"]).argmax(axis=1)
    dist, _ = bl.train_val_d8(list(train_paths), list(target_paths))
    d_nn = dist[np.arange(len(nn_idx)), nn_idx]
    near = d_nn <= rule["threshold"]
    return near, {"rule": rule["rule"], "threshold": rule["threshold"],
                  "near": sorted(target_paths[near].tolist()), "clean": sorted(target_paths[~near].tolist()),
                  "n_near": int(near.sum()), "n_clean": int((~near).sum())}


# ---------- 3. metrics ----------

def group_bootstrap_auc(y, s, groups, n=N_BOOT, seed=BOOT_SEED):
    """95% CI for AUROC, resampling whole groups (group_id) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ug]
    aucs = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


def class_metrics(y, pred):
    """Accuracy and per-class recall."""
    return {"acc": float((pred == y).mean()),
            "recall_ok": float((pred[y == 0] == 0).mean()), "recall_defect": float((pred[y == 1] == 1).mean())}


def strata_acc(y, pred, near):
    """Accuracy on the near and clean strata."""
    return {"acc_near": float((pred[near] == y[near]).mean()) if near.any() else None,
            "acc_clean": float((pred[~near] == y[~near]).mean()) if (~near).any() else None}


def auc_or_none(y, s):
    """AUROC, or None when only one class is present."""
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def score_frozen_draw(enc, choice, budget_paths, feats, train, y, groups, near):
    """k-NN, probe and anomaly score for one encoder and one budget draw, all with the frozen choices."""
    row_of = {p: i for i, p in enumerate(train["path"])}
    idx = np.array([row_of[p] for p in budget_paths])
    X, yl = feats[enc]["train"][idx], train["label"].to_numpy()[idx]
    Xt = feats[enc]["target"]
    out = {"encoder": enc, "budget": str(choice["budget"]), "draw": choice["draw"]}

    knn = bl.knn_predict(X, yl, Xt, choice["knn_k"])
    out["knn"] = {"k": choice["knn_k"], **class_metrics(y, knn), **strata_acc(y, knn, near)}
    probe = bl.fit_probe(X, yl, choice["probe_C"]).predict(Xt)   # scaler fit on the labeled budget only
    out["probe"] = {"C": choice["probe_C"], **class_metrics(y, probe), **strata_acc(y, probe, near)}

    scores = bl.anomaly_option_scores(X[yl == 0], Xt, choice["anomaly_option"])
    flagged = scores >= choice["threshold"]
    out["anomaly"] = {
        "option": choice["anomaly_option"], "threshold": choice["threshold"],
        "auroc": auc_or_none(y, scores), "auroc_ci95_group": group_bootstrap_auc(y, scores, groups),
        "auroc_near": auc_or_none(y[near], scores[near]), "auroc_clean": auc_or_none(y[~near], scores[~near]),
        "missed_defects": int((~flagged & (y == 1)).sum()), "false_rejects": int((flagged & (y == 0)).sum()),
        "recall_defect": float(flagged[y == 1].mean()), "recall_ok": float((~flagged)[y == 0].mean())}
    return out


def score_supervised(manifest, target_x, y, near):
    """Accuracy, per-class recall and strata accuracy for every frozen supervised model."""
    rows = []
    for name, entry in manifest["files"].items():
        if not name.startswith("sup_"):
            continue
        _, budget, draw = name.split("_")
        pred = bl.predict(supervised_model(entry["path"]), target_x).numpy()
        rows.append({"budget": budget, "draw": int(draw), "lr": entry["lr"],
                     **class_metrics(y, pred), **strata_acc(y, pred, near)})
    return rows


def embedding_quality(emb_by_encoder, y, groups):
    """Silhouette and Davies-Bouldin on one image per group (seed 42), L2-normalized features."""
    rng = np.random.default_rng(GROUP_SEED)
    pick = np.array([rng.choice(np.flatnonzero(groups == g)) for g in np.unique(groups)])
    out = {}
    for name, f in emb_by_encoder.items():
        z = f[pick] / np.linalg.norm(f[pick], axis=1, keepdims=True)
        out[name] = {"silhouette": round(float(silhouette_score(z, y[pick])), 4),
                     "davies_bouldin": round(float(davies_bouldin_score(z, y[pick])), 4), "n": int(len(pick))}
    return out


# ---------- 4. summaries and outputs ----------

def summarize(frozen_rows, sup_rows):
    """Mean and std over draws for every method x budget (accuracy, recalls, strata, anomaly)."""
    recs = []
    for r in frozen_rows:
        for clf, key in (("k-NN", "knn"), ("probe", "probe")):
            recs.append({"method": f"{ENC_NAME[r['encoder']]} {clf}", "budget": r["budget"], **r[key]})
    for r in sup_rows:
        recs.append({"method": "supervised", **r})
    df = pd.DataFrame(recs)
    cols = ["acc", "recall_ok", "recall_defect", "acc_near", "acc_clean"]
    acc = df.groupby(["method", "budget"])[cols].agg(["mean", "std", "count"])

    an = pd.DataFrame([{"encoder": r["encoder"], "budget": r["budget"], "draw": r["draw"], **r["anomaly"]}
                       for r in frozen_rows])
    return acc, an


def fmt(m, s, n):
    """mean +/- std, or just the mean for a single draw."""
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m:.3f}" if n <= 1 else f"{m:.3f} +/- {s:.3f}"


def flags_from(acc):
    """Probe beats k-NN by > 10 points, or k-NN beats probe (per encoder and budget)."""
    out = []
    for enc in ("SSL", "random"):
        for b in BUDGETS:
            knn, probe = acc.loc[(f"{enc} k-NN", b), ("acc", "mean")], acc.loc[(f"{enc} probe", b), ("acc", "mean")]
            gap = 100 * (probe - knn)
            if gap > 10:
                out.append(f"{enc}, budget {b}: probe beats k-NN by {gap:.1f} points (> 10)")
            elif gap < 0:
                out.append(f"{enc}, budget {b}: k-NN beats probe by {-gap:.1f} points")
    return out


def plot_budget_curve(acc, majority, label, n, path):
    """Accuracy vs labels per class, mean with std bars, majority-class dashed line."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(BUDGETS))
    for i, (method, (color, style, marker)) in enumerate(bl.VAL_STYLES.items()):
        m = [acc.loc[(method, b), ("acc", "mean")] for b in BUDGETS]
        s = [np.nan_to_num(acc.loc[(method, b), ("acc", "std")]) for b in BUDGETS]
        xs = x + (i - 2) * 0.05
        ax.errorbar(xs, m, yerr=s, color=color, ls=style, marker=marker, ms=8, lw=2, capsize=4, label=method)
        ax.annotate(f"{method} {m[-1]:.3f}", (xs[-1], m[-1]), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=12, color="#333333")
    ax.axhline(majority, color="#888888", ls="--", lw=1.5)
    ax.text(x[0] - 0.25, majority + 0.006, f"majority class {majority:.1%}", fontsize=12, color="#555555")
    ax.set_xticks(x, list(BUDGETS), fontsize=13)
    ax.set_xlim(-0.35, len(x) - 1 + 1.25)
    ax.set_xlabel("labeled images per class ('all' = every train label)", fontsize=14)
    ax.set_ylabel(f"{label} accuracy ({n} images)", fontsize=14)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    ax.set_title(f"{label}: accuracy vs label budget, frozen choices\nmean +/- std over 3 draws; 'all' is 1 draw",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(emb_by_encoder, y, label, path):
    """3-panel t-SNE (SSL, random-init, supervised-all) colored by label."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (name, f) in zip(axes, emb_by_encoder.items()):
        z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(
            f / np.linalg.norm(f, axis=1, keepdims=True))
        for cls, color, lab in ((0, "#2a78d6", "ok"), (1, "#eb6834", "defect")):
            ax.scatter(z[y == cls, 0], z[y == cls, 1], s=14, c=color, label=lab, alpha=0.8)
        ax.set_title(name, fontsize=15)
        ax.set_xticks([]), ax.set_yticks([])
        ax.legend(fontsize=12, frameon=False)
    fig.suptitle(f"t-SNE of frozen encoder features on {label} (a picture, not proof)", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_comparison(path, label, counts, manifest, strata, acc, an, flags, quality):
    """results/comparison.md: every method at every budget, anomaly, per-class recall, strata, embeddings."""
    L = [f"# {label} results (frozen choices)", ""]
    if label.startswith("CONTAMINATED"):
        L += ["**CONTAMINATED. The SSL encoder was pretrained on the train pool PLUS the TEST images.** "
              "Leakage control only, never a result. Same frozen k, C, anomaly option and threshold as the clean "
              "run; random-init and supervised rows are unchanged from the clean run. "
              "The leakage estimate is in leakage_gap.json (CONTAMINATED minus clean).", ""]
    elif label != "TEST":
        L += ["**DRY RUN on VAL. Not the reported result.** Val chose these settings, so these numbers are optimistic.", ""]
    L += [f"Frozen at {manifest['frozen_at_utc']}; every hash verified before scoring.",
          f"{counts['n']} images: {counts['ok']} ok, {counts['defect']} defect. Majority class: "
          f"{counts['majority']:.1%}. One image = {100 / counts['n']:.2f} points.",
          f"Strata: near {strata['n_near']}, clean {strata['n_clean']} ({strata['rule']})", "",
          "## Accuracy (mean +/- std over 3 draws; 'all' = 1 draw)", "",
          "| method | " + " | ".join(BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(fmt(acc.loc[(m, b), ("acc", "mean")], acc.loc[(m, b), ("acc", "std")],
                                              acc.loc[(m, b), ("acc", "count")]) for b in BUDGETS) + " |")
    L += ["", "## Probe vs k-NN flags", ""] + ([f"- {f}" for f in flags] or ["- none"])
    L += ["", "## Per-class recall (mean over draws): ok / defect", "",
          "| method | " + " | ".join(BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{acc.loc[(m, b), ('recall_ok', 'mean')]:.3f} / "
                                          f"{acc.loc[(m, b), ('recall_defect', 'mean')]:.3f}" for b in BUDGETS) + " |")
    L += ["", "## Anomaly score (frozen option and threshold; 95% CI = group bootstrap, 1000 resamples, seed 42)", "",
          "| encoder | budget | draw | option | AUROC [95% CI] | missed defects | false rejects | "
          "defect recall | ok recall | AUROC near / clean |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in an.itertuples():
        near = "n/a" if r.auroc_near is None else f"{r.auroc_near:.3f}"
        clean = "n/a" if r.auroc_clean is None else f"{r.auroc_clean:.3f}"
        L.append(f"| {ENC_NAME[r.encoder]} | {r.budget} | {r.draw} | {r.option} | {r.auroc:.3f} "
                 f"[{r.auroc_ci95_group[0]:.3f}, {r.auroc_ci95_group[1]:.3f}] | {r.missed_defects} | "
                 f"{r.false_rejects} | {r.recall_defect:.3f} | {r.recall_ok:.3f} | {near} / {clean} |")
    L += ["", f"## Strata: accuracy near ({strata['n_near']}) / clean ({strata['n_clean']}), mean over draws", "",
          "| method | " + " | ".join(BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{acc.loc[(m, b), ('acc_near', 'mean')]:.3f} / "
                                          f"{acc.loc[(m, b), ('acc_clean', 'mean')]:.3f}" for b in BUDGETS) + " |")
    L += ["", "## Embedding quality (one image per group, seed 42; L2-normalized)", "",
          "| encoder | silhouette | Davies-Bouldin | n |", "|---|---|---|---|"]
    L += [f"| {k} | {v['silhouette']:.4f} | {v['davies_bouldin']:.4f} | {v['n']} |" for k, v in quality.items()]
    L += ["", "Higher silhouette and lower Davies-Bouldin = cleaner ok/defect separation. The t-SNE plot "
          "(tsne.png) is a picture, not proof; the k-NN, probe and anomaly numbers are the proof.", ""]
    path.write_text("\n".join(L), encoding="ascii")


def main():
    """Verify, compute strata, read labels, score everything once, write outputs (and the lock)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run everything on VAL into results/dry_run/, no lock")
    parser.add_argument("--contaminated", action="store_true",
                        help="score the CONTAMINATED SSL encoder on test into results/contaminated/; never locks")
    args = parser.parse_args()
    ssl_path = bl.SSL_WEIGHTS
    if args.dry_run:
        split, label, out = "val", "DRY RUN (VAL)", Path("results/dry_run")
    elif args.contaminated:
        split, label, out = "test", "CONTAMINATED TEST", Path("results/contaminated")
        ssl_path = CONTAM_WEIGHTS
        if not LOCK.exists() or json.loads(LOCK.read_text())["status"] != "complete":
            raise SystemExit("REFUSING: the clean test run must be complete (results/eval.lock) before CONTAMINATED")
        if not ssl_path.exists():
            raise SystemExit(f"REFUSING: {ssl_path} missing; run python contaminated.py --pretrain first")
    else:
        split, label, out = "test", "TEST", Path("results")
        if LOCK.exists():
            raise SystemExit(f"REFUSING TO RUN: {LOCK} exists, test has already been scored once.")
    out.mkdir(parents=True, exist_ok=True)
    locking = not (args.dry_run or args.contaminated)   # only the one clean test run writes eval.lock

    manifest = verify_manifest()
    choices = json.loads(CHOICES.read_text())
    budgets = {(str(d["budget"]), d["draw"]): d["paths"] for d in json.loads(bl.BUDGET_PATH.read_text())["draws"]}

    # --- no labels of the target split are read before this block ends ---
    paths, groups, target_u8 = load_target(split)
    feats, train = extract_all(target_u8, ssl_path)
    near, strata = compute_strata(feats, train["path"].to_numpy(), paths, choices["test_strata"])
    strata_name = "val_strata.json" if args.dry_run else "test_strata.json"
    (out / strata_name).write_text(json.dumps(strata, indent=2))
    print(f"strata written before any label was read: near {strata['n_near']}, clean {strata['n_clean']} "
          f"-> {out / strata_name}")
    if locking:   # lock before the first test label is read, so a crash cannot allow a second attempt
        LOCK.write_text(json.dumps({"status": "started", "at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"), "manifest_frozen_at": manifest["frozen_at_utc"]}, indent=2))

    y = data.load_split(split, allow_test=(split == "test"), columns=["label"])["label"].to_numpy()
    counts = {"n": len(y), "ok": int((y == 0).sum()), "defect": int((y == 1).sum())}
    counts["majority"] = max(counts["ok"], counts["defect"]) / counts["n"]

    frozen_rows = [score_frozen_draw(enc, c, budgets[(str(c["budget"]), c["draw"])], feats, train, y, groups, near)
                   for enc, draws in choices["encoders"].items() for c in draws]
    target_x = bl.simclr.to_float(target_u8)
    sup_rows = score_supervised(manifest, target_x, y, near)
    acc, an = summarize(frozen_rows, sup_rows)
    flags = flags_from(acc)

    sup_all = bl.extract_features(supervised_model(manifest["files"]["sup_all_0"]["path"], trunk_only=True), target_u8)
    emb = {"SSL": feats["ssl"]["target"], "random-init": feats["random"]["target"], "supervised (all)": sup_all}
    quality = embedding_quality(emb, y, groups)

    metrics = {"label": label, "split": split, "counts": counts, "manifest_frozen_at": manifest["frozen_at_utc"],
               "strata": {k: strata[k] for k in ("rule", "threshold", "n_near", "n_clean")},
               "frozen_draws": frozen_rows, "supervised": sup_rows, "flags": flags, "embedding_quality": quality,
               "bootstrap": {"resamples": N_BOOT, "seed": BOOT_SEED, "unit": "group_id"},
               "ssl_encoder": {"path": str(ssl_path), "sha256": bl.sha256_of(ssl_path)}}
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_comparison(out / "comparison.md", label, counts, manifest, strata, acc, an, flags, quality)
    plot_budget_curve(acc, counts["majority"], label, counts["n"], out / "test_budget_curve.png")
    plot_tsne(emb, y, label, out / "tsne.png")
    if locking:
        LOCK.write_text(json.dumps({"status": "complete", "at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"), "manifest_frozen_at": manifest["frozen_at_utc"]}, indent=2))
    print(f"wrote {out}/test_metrics.json, comparison.md, test_budget_curve.png, tsne.png"
          + (f"; locked {LOCK}" if locking else ""))
    print("\n" + (out / "comparison.md").read_text())


if __name__ == "__main__":
    main()
