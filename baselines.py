"""Label budgets, random-init floor, and the supervised-from-scratch baseline."""

import copy
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import simclr

BUDGETS = [10, 25, 50]   # per class, plus "all"
DRAW_SEEDS = [0, 1, 2]
BUDGET_PATH = Path("splits/budgets.json")
SSL_WEIGHTS = Path("weights/ssl_encoder.pt")
FEATURE_SPLITS = ("train", "val")   # test features are extracted only by eval.py
ENCODERS = ("ssl", "random")
K_GRID = (1, 5, 10, 20)
C_GRID = (0.01, 0.1, 1, 10)
DEFECT_RECALL = 0.95
SUP_BUDGETS = ("10", "25", "50", "all")
MANIFEST_PATH = Path("results/frozen_manifest.json")
RESIDUAL_D8 = 40   # 8-orient pHash distance splitting "near" from "clean" (grouping used 24)
# Global anomaly scores, added after seeing val: with all ok labels, k <= 20 only measures local
# distance, and every val image (ok or defect) has a near ok neighbor, so local AUROC fell to chance.
ANOMALY_GLOBAL = ("all_ok", "centroid")
ANOMALY_GRID_NOTE = ("Anomaly grid widened AFTER seeing val: added 'all_ok' (mean cosine distance to all labeled "
                     "ok embeddings) and 'centroid' (cosine distance to the labeled ok centroid). Reason, as observed "
                     "on the pre_logo run: with the 'all' budget (346 ok refs), k in {1, 5, 10, 20} only measures "
                     "local distance; every val image, "
                     "ok or defect, has a near ok neighbor (median nearest-ok distance ratio defect/ok = 1.01), so "
                     "SSL val AUROC fell to 0.57 while a global distance gave about 0.79. At 10 per class the chosen "
                     "k = 10 already averaged over every ok ref, i.e. it was already a global score. Because the "
                     "grid change was made after looking at val, val AUROC for the anomaly score is extra optimistic; "
                     "the test run is the check.")
CHOICES_PATH = Path("results/frozen_choices.json")
LR_GRID = (1e-4, 3e-4, 1e-3)   # chosen per budget on draw 0 by val accuracy
SUP = {"img_size": 96, "batch_size": 32, "lr_grid": list(LR_GRID), "patience": 8,
       "max_epochs": 60, "max_epochs_all": 20, "seed": 42, "aug_seed": 123,
       "max_seconds": 450,        # per process for budgets 10/25/50, under the 8-minute run limit
       "max_seconds_all": 600}    # "all": 20 epochs at ~24.5 s must fit for every lr, so no lr is cut by time
SUP_LOG = Path("results/supervised_log.json")
VAL_BUDGETS = ("10", "25", "50", "all")
# hue = feature source (SSL blue, random orange, supervised aqua); solid = probe/trained, dotted = k-NN
VAL_STYLES = {"SSL probe": ("#2a78d6", "-", "o"), "SSL k-NN": ("#2a78d6", ":", "s"),
              "random probe": ("#eb6834", "-", "o"), "random k-NN": ("#eb6834", ":", "s"),
              "supervised": ("#1baf7a", "-", "D")}
VAL_LABEL_NUDGE = {"SSL probe": 7, "random k-NN": -7}   # points; keeps close end labels apart


def make_budgets(path=BUDGET_PATH):
    """Draw per-class label budgets (3 seeds each) plus all labels from train.csv; save the exact image IDs."""
    train = data.load_split("train")
    draws = []
    for seed in DRAW_SEEDS:
        # One shuffle per class per seed, so budgets are nested: the 10 are inside the 25, inside the 50.
        rng = np.random.default_rng(seed)
        order = {label: rng.permutation(train.loc[train["label"] == label, "path"].to_numpy())
                 for label in sorted(data.CLASSES.values())}
        for n in BUDGETS:
            paths = sorted(p for label in order for p in order[label][:n])
            draws.append({"budget": n, "draw": seed, "paths": paths})
    draws.append({"budget": "all", "draw": 0, "paths": sorted(train["path"])})
    record = {"source": "splits/train.csv", "id": "path", "per_class": True,
              "nested_within_seed": True, "draws": draws}
    path.write_text(json.dumps(record, indent=2))
    return record


def check_budgets(record):
    """Print budget, draw, ok and defect counts; confirm every ID is in train.csv and none repeats."""
    train = data.load_split("train")
    label_of = dict(zip(train["path"], train["label"]))
    rows, missing = [], 0
    for d in record["draws"]:
        labels = [label_of.get(p) for p in d["paths"]]
        missing += sum(l is None for l in labels)
        rows.append({"budget": d["budget"], "draw": d["draw"],
                     "ok": labels.count(0), "defect": labels.count(1),
                     "unique ids": len(set(d["paths"]))})
    print(pd.DataFrame(rows).to_string(index=False))
    status = "PASS" if missing == 0 else "FAIL"
    print(f"{status}: {missing} IDs not in train.csv across {len(record['draws'])} draws")
    if missing:
        raise SystemExit(1)


def random_init_encoder(seed=42):
    """Return an untrained ResNet-18 encoder (fc = Identity) as the floor; same init as SSL epoch 0."""
    torch.manual_seed(seed)
    return simclr.build_encoder().eval()


def ssl_encoder(path=SSL_WEIGHTS):
    """Load the frozen clean SimCLR encoder (encoder only, no projection head)."""
    enc = simclr.build_encoder()
    enc.load_state_dict(torch.load(path, weights_only=True))
    return enc.eval()


@torch.no_grad()
def extract_features(encoder, images_u8, batch=128):
    """Frozen 512-d encoder features: 96 px, fixed normalization, no augmentation."""
    encoder.eval()
    x = simclr.to_float(images_u8)
    return torch.cat([encoder(simclr.to_model_input(x[i:i + batch])) for i in range(0, len(x), batch)]).numpy()


def save_features(name, encoder, splits=FEATURE_SPLITS):
    """Extract train and val features (never test) and write results/feats_{name}.npz."""
    out, t0 = {}, time.perf_counter()
    for split in splits:
        images, labels = data.load_images(split)
        out[f"{split}_feats"] = extract_features(encoder, images)
        out[f"{split}_labels"] = labels
        out[f"{split}_paths"] = data.load_split(split)["path"].to_numpy().astype(str)
    path = Path("results") / f"feats_{name}.npz"
    np.savez(path, **out)
    shapes = ", ".join(f"{s} {out[f'{s}_feats'].shape}" for s in splits)
    print(f"  {name:<7} {shapes}  {time.perf_counter() - t0:.1f} s  -> {path}")


def build_supervised(seed=SUP["seed"]):
    """resnet18(weights=None) with a 2-class head; seed 42 gives the same trunk init as SSL and random."""
    torch.manual_seed(seed)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    return model


@torch.no_grad()
def predict(model, x, batch=128):
    """Class predictions for float images (n, 1, H, W), eval mode, no augmentation."""
    model.eval()
    return torch.cat([model(simclr.to_model_input(x[i:i + batch])).argmax(1) for i in range(0, len(x), batch)])


def train_supervised(x, y, val_x, val_y, max_epochs, deadline, lr):
    """Train from scratch with one SimCLRAugment view per image; early stop on val accuracy."""
    model = build_supervised()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    augment = simclr.SimCLRAugment(aug_seed=SUP["aug_seed"])
    shuffle_gen = torch.Generator().manual_seed(SUP["seed"])
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    stop, epoch = "max_epochs", 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=shuffle_gen)
        for i in range(0, len(x), SUP["batch_size"]):
            idx = order[i:i + SUP["batch_size"]]
            if len(idx) < 2:                      # BatchNorm cannot train on a batch of 1
                continue
            view = torch.stack([augment(img) for img in x[idx]])
            loss = F.cross_entropy(model(simclr.to_model_input(view)), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = (predict(model, val_x) == val_y).float().mean().item()
        if val_acc > best["val_acc"]:             # strict: a tie keeps the earlier epoch
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if epoch - best["epoch"] >= SUP["patience"]:
            stop = "early_stop"
            break
        if time.perf_counter() > deadline:       # 8-minute process cap from CLAUDE.md
            stop = "time_cap"
            break
    return best, epoch, stop


def lr_key(lr):
    """Stable string key for a learning rate, e.g. 0.0003."""
    return f"{lr:g}"


def load_sup_log():
    """Read results/supervised_log.json, or start an empty one."""
    if SUP_LOG.exists():
        return json.loads(SUP_LOG.read_text())
    return {"config": SUP, "lr_rule": "max draw-0 val accuracy, tie -> lower lr",
            "lr_search": {}, "chosen_lr": {}, "runs": {}}


def run_supervised(budget, draw, lr, weights_path):
    """Train one supervised model (one budget draw, one lr) in this process; save best-val-epoch weights."""
    t0 = time.perf_counter()
    deadline = t0 + SUP["max_seconds_all" if str(budget) == "all" else "max_seconds"]
    d = next(d for d in json.loads(BUDGET_PATH.read_text())["draws"]
             if str(d["budget"]) == str(budget) and d["draw"] == draw)
    train_u8, train_labels = data.load_images("train")
    row_of = {p: i for i, p in enumerate(data.load_split("train")["path"])}
    val_u8, val_labels = data.load_images("val")
    val_x, val_y = simclr.to_float(val_u8), torch.tensor(val_labels)
    idx = np.array([row_of[p] for p in d["paths"]])
    x, y = simclr.to_float(train_u8[idx]), torch.tensor(train_labels[idx])
    max_epochs = SUP["max_epochs_all"] if str(budget) == "all" else SUP["max_epochs"]

    best, epochs_run, stop = train_supervised(x, y, val_x, val_y, max_epochs, deadline, lr)
    torch.save(best["state"], weights_path)
    seconds = time.perf_counter() - t0
    run = {"budget": d["budget"], "draw": draw, "lr": lr, "n_labeled": len(idx),
           "val_acc": round(best["val_acc"], 4), "best_epoch": best["epoch"], "epochs_run": epochs_run,
           "stop": stop, "seconds": round(seconds, 1), "sec_per_epoch": round(seconds / epochs_run, 2),
           "time_cap_flag": stop == "time_cap",   # True = cut off by time, not by early stopping or max epochs
           "weights": str(weights_path)}
    print(f"  budget {d['budget']:>3} draw {draw}  lr {lr_key(lr):<6}  n={len(idx):>3}  val_acc {best['val_acc']:.4f}  "
          f"best_epoch {best['epoch']:>2}/{epochs_run:>2}  stop {stop:<10}  {seconds:6.1f} s  -> {weights_path}")
    if run["time_cap_flag"]:
        print(f"  WARNING: TIME_CAP - budget {d['budget']} draw {draw} lr {lr_key(lr)} stopped by the time guard "
              f"after {epochs_run} of {max_epochs} epochs")
    return run


def lr_search_run(budget, lr):
    """Draw 0 at one candidate lr (one process each); log it under lr_search."""
    run = run_supervised(budget, 0, lr, Path("weights") / f"sup_{budget}_0_lr{lr_key(lr)}.pt")
    log = load_sup_log()
    log["lr_search"].setdefault(str(budget), {})[lr_key(lr)] = run
    SUP_LOG.write_text(json.dumps(log, indent=2))


def pick_lr(budget):
    """Choose the lr with the best draw-0 val accuracy; its draw-0 model becomes sup_<budget>_0.pt."""
    log = load_sup_log()
    cands = log["lr_search"].get(str(budget), {})
    missing = [lr_key(lr) for lr in LR_GRID if lr_key(lr) not in cands]
    if missing:
        raise SystemExit(f"lr search for budget {budget} incomplete, missing lr {missing}")
    best = max(LR_GRID, key=lambda lr: (cands[lr_key(lr)]["val_acc"], -lr))   # tie -> lower lr
    run = cands[lr_key(best)]
    final = Path("weights") / f"sup_{budget}_0.pt"
    shutil.copyfile(run["weights"], final)
    log["chosen_lr"][str(budget)] = best
    log["runs"][f"{budget}_0"] = {**run, "weights": str(final)}
    SUP_LOG.write_text(json.dumps(log, indent=2))
    print(f"budget {budget}: " + ", ".join(f"lr {k} val {v['val_acc']:.4f}" for k, v in cands.items())
          + f"  -> chosen lr {lr_key(best)}")
    capped = [k for k, v in cands.items() if v.get("time_cap_flag")]
    if capped:
        print(f"  WARNING: TIME_CAP in the lr search for budget {budget} at lr {capped}; the comparison is not even")


def train_remaining_draws(budget, draws):
    """Train the given draws at the lr chosen on draw 0; log them under runs."""
    log = load_sup_log()
    if str(budget) not in log["chosen_lr"]:
        raise SystemExit(f"no chosen lr for budget {budget}: run the lr search and --pick-lr first")
    lr = log["chosen_lr"][str(budget)]
    for draw in draws:
        run = run_supervised(budget, draw, lr, Path("weights") / f"sup_{budget}_{draw}.pt")
        log = load_sup_log()
        log["runs"][f"{budget}_{draw}"] = run
        SUP_LOG.write_text(json.dumps(log, indent=2))


# ---------- frozen-feature classifiers and anomaly score (shared with eval.py) ----------

def cosine_sim(query, ref):
    """Cosine similarity matrix (n_query, n_ref)."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    r = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    return q @ r.T


def knn_predict(ref_feats, ref_labels, query_feats, k):
    """Cosine k-NN majority vote; an exact tie goes to the single nearest neighbor's label."""
    nn_idx = np.argsort(-cosine_sim(query_feats, ref_feats), axis=1)[:, :k]
    votes = ref_labels[nn_idx].mean(axis=1)
    pred = (votes > 0.5).astype(int)
    tie = votes == 0.5
    pred[tie] = ref_labels[nn_idx[tie, 0]]
    return pred


def fit_probe(ref_feats, ref_labels, C):
    """StandardScaler fit on the labeled subset only, then LogisticRegression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(ref_feats, ref_labels)


def anomaly_scores(ok_feats, query_feats, k):
    """Mean cosine distance to the k nearest labeled ok embeddings (higher = more anomalous)."""
    dist = 1.0 - cosine_sim(query_feats, ok_feats)
    return np.sort(dist, axis=1)[:, :k].mean(axis=1)


def anomaly_option_scores(ok_feats, query_feats, option):
    """Anomaly score for one grid option: an int k (local), "all_ok" or "centroid" (global)."""
    if option == "all_ok":      # mean cosine distance to every labeled ok embedding
        return anomaly_scores(ok_feats, query_feats, len(ok_feats))
    if option == "centroid":    # cosine distance to the mean of the L2-normalized ok embeddings
        unit = ok_feats / np.linalg.norm(ok_feats, axis=1, keepdims=True)
        return anomaly_scores(unit.mean(axis=0, keepdims=True), query_feats, 1)
    return anomaly_scores(ok_feats, query_feats, option)


def threshold_for_recall(scores, labels, recall=DEFECT_RECALL):
    """Highest threshold that still flags at least `recall` of the defects (score >= threshold)."""
    defect = np.sort(scores[labels == 1])
    return float(defect[int(np.floor((1 - recall) * len(defect)))])


def pick_best(candidates, metric):
    """Best value by metric; a tie goes to the earlier (simpler) candidate."""
    return max(candidates, key=lambda c: (metric(c), -candidates.index(c)))


# ---------- validation tuning ----------

def tune_draw(feats, budget_paths):
    """Choose k, C, anomaly k and threshold on val for one encoder and one budget draw."""
    row_of = {p: i for i, p in enumerate(feats["train_paths"])}
    idx = np.array([row_of[p] for p in budget_paths])
    X, y = feats["train_feats"][idx], feats["train_labels"][idx]
    Xv, yv = feats["val_feats"], feats["val_labels"]
    ok = X[y == 0]

    ks = [k for k in K_GRID if k <= len(y)]
    knn_acc = {k: float((knn_predict(X, y, Xv, k) == yv).mean()) for k in ks}
    k = pick_best(ks, knn_acc.get)

    probe_acc = {C: float(fit_probe(X, y, C).score(Xv, yv)) for C in C_GRID}
    C = pick_best(list(C_GRID), probe_acc.get)

    options = [ka for ka in K_GRID if ka <= len(ok)] + list(ANOMALY_GLOBAL)   # local k first, then global
    auroc = {o: float(roc_auc_score(yv, anomaly_option_scores(ok, Xv, o))) for o in options}
    anom = pick_best(options, auroc.get)
    scores = anomaly_option_scores(ok, Xv, anom)
    thr = threshold_for_recall(scores, yv)
    flagged = scores >= thr
    return {"n_labeled": int(len(y)),
            "knn_k": k, "val_knn_acc": knn_acc[k],
            "probe_C": C, "val_probe_acc": probe_acc[C],
            "anomaly_option": anom, "val_auroc": auroc[anom],
            "val_auroc_by_option": {str(o): round(a, 4) for o, a in auroc.items()},
            "threshold": thr,
            "val_defect_recall": float(flagged[yv == 1].mean()),
            "val_false_reject_rate": float(flagged[yv == 0].mean())}


def tune_all(budget_path=BUDGET_PATH, out_path=CHOICES_PATH):
    """Tune every encoder x budget draw on val; save all choices to results/frozen_choices.json."""
    budgets = json.loads(budget_path.read_text())["draws"]
    choices = {"grids": {"k": list(K_GRID), "C": list(C_GRID),
                         "anomaly": list(K_GRID) + list(ANOMALY_GLOBAL)},
               "grid_note": ANOMALY_GRID_NOTE,
               "rules": {"knn_k": "max val accuracy, tie -> smaller k",
                         "probe_C": "max val accuracy, tie -> smaller C",
                         "anomaly_option": "max val AUROC, tie -> earlier option (k ascending, then all_ok, centroid)",
                         "threshold": f"highest score threshold with val defect recall >= {DEFECT_RECALL}"},
               "encoders": {}}
    for name in ENCODERS:
        feats = dict(np.load(Path("results") / f"feats_{name}.npz"))
        choices["encoders"][name] = [{"budget": d["budget"], "draw": d["draw"], **tune_draw(feats, d["paths"])}
                                     for d in budgets]
    if out_path.exists():   # keep pre-registered test strata (written by check_splits.py --strata)
        old = json.loads(out_path.read_text())
        if "test_strata" in old:
            choices["test_strata"] = old["test_strata"]
    out_path.write_text(json.dumps(choices, indent=2))
    return choices


def train_val_d8(train_paths, val_paths):
    """8-orientation pHash distance matrix (n_val, n_train) and orientation of val that best matches train."""
    cache = np.load(data.HASH_CACHE, allow_pickle=False)
    row_of = {p: i for i, p in enumerate(cache["paths"])}
    d8 = cache["h256_d8"]
    dv, dt = d8[[row_of[p] for p in val_paths]], d8[[row_of[p] for p in train_paths]]
    per_o = np.stack([data.cross_hamming(dv[:, o], dt[:, 0]) for o in range(dv.shape[1])])   # (8, n_val, n_train)
    return data.d8_distance(dv, dt), per_o.argmin(axis=0)


def residual_twins(threshold=RESIDUAL_D8):
    """VAL only: nearest train image by random-init cosine, its 8-orientation pHash distance, 1-NN accuracy."""
    f = dict(np.load(Path("results") / "feats_random.npz"))
    nn = cosine_sim(f["val_feats"], f["train_feats"]).argmax(axis=1)
    dist, orient_idx = train_val_d8(list(f["train_paths"]), list(f["val_paths"]))
    rows = np.arange(len(nn))
    df = pd.DataFrame({"val_path": f["val_paths"], "train_path": f["train_paths"][nn],
                       "d8_to_nn": dist[rows, nn], "orientation": orient_idx[rows, nn],
                       "d8_min_any_train": dist.min(axis=1),
                       "correct": f["train_labels"][nn] == f["val_labels"]})
    summary = []
    for col, desc in (("d8_to_nn", "partner = nearest train image by random-init cosine"),
                      ("d8_min_any_train", "nearest train image by 8-orient pHash (test strata rule)")):
        for name, mask in ((f"near (<= {threshold})", df[col] <= threshold), (f"clean (> {threshold})", df[col] > threshold)):
            summary.append({"distance": desc, "stratum": name, "val images": int(mask.sum()),
                            "1-NN acc": round(float(df.loc[mask, "correct"].mean()), 4) if mask.any() else None})
    print(f"\nVAL residual twins (random-init 1-NN, 8-orient 256-bit pHash, threshold {threshold})")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"  overall val 1-NN acc {df['correct'].mean():.4f}; d8_to_nn min {df['d8_to_nn'].min()}, "
          f"median {df['d8_to_nn'].median():.0f}")
    Path("results/residual_twins.json").write_text(json.dumps(   # for report.ipynb (VAL only)
        {"split": "val", "threshold": threshold, "overall_1nn_acc": round(float(df["correct"].mean()), 4),
         "summary": summary}, indent=2))
    save_residual_figure(df)
    return df, summary


def save_residual_figure(df, path=Path("results/residual_twins.png"), n_pairs=3):
    """The 3 closest val / nearest-train pairs: val image, train partner, partner un-oriented to match."""
    pairs = df.sort_values(["d8_to_nn", "val_path"]).head(n_pairs)
    fig, axes = plt.subplots(len(pairs), 3, figsize=(10, 3.4 * len(pairs)), squeeze=False)
    for r, row in enumerate(pairs.itertuples()):
        val_img, train_img = data.resize_paths([row.val_path, row.train_path], px=256)
        k, m = data.ORIENTATIONS[row.orientation]
        tiles = [(val_img, f"VAL {Path(row.val_path).stem}"),
                 (train_img, f"TRAIN 1-NN {Path(row.train_path).stem}\n{'same' if row.correct else 'DIFFERENT'} label"),
                 (data.unorient(train_img, k, m), f"train, undo {data.ORIENTATION_NAMES[row.orientation]}\n"
                                                  f"8-orient Hamming {row.d8_to_nn}")]
        for c, (img, title) in enumerate(tiles):
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(title, fontsize=12)
            axes[r, c].axis("off")
    fig.suptitle("Residual twins: closest val / nearest-train pairs (random-init cosine 1-NN)", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def split_counts(split):
    """Images, ok and defect counts for a train or val split (reads the label column only)."""
    y = data.load_split(split, columns=["label"])["label"]
    return {"n": len(y), "ok": int((y == 0).sum()), "defect": int((y == 1).sum())}


def majority_acc(counts):
    """Accuracy of always predicting the larger class."""
    return max(counts["ok"], counts["defect"]) / counts["n"]


def val_curve_table():
    """Mean and std of val accuracy per method and budget, from frozen_choices.json and supervised_log.json."""
    choices = json.loads(CHOICES_PATH.read_text())["encoders"]
    sup = pd.DataFrame(json.loads(SUP_LOG.read_text())["runs"].values())
    rows = []
    for enc, name in (("ssl", "SSL"), ("random", "random")):
        df = pd.DataFrame(choices[enc])
        for col, clf in (("val_knn_acc", "k-NN"), ("val_probe_acc", "probe")):
            for b, g in df.groupby("budget", sort=False):
                rows.append({"method": f"{name} {clf}", "budget": str(b), "mean": g[col].mean(),
                             "std": g[col].std(ddof=1), "draws": len(g)})
    for b, g in sup.groupby("budget", sort=False):
        rows.append({"method": "supervised", "budget": str(b), "mean": g["val_acc"].mean(),
                     "std": g["val_acc"].std(ddof=1), "draws": len(g)})
    return pd.DataFrame(rows)


def probe_knn_flags(table):
    """Flag budgets where the probe beats k-NN by > 10 points, or k-NN beats the probe (per encoder)."""
    flags = []
    for name in ("SSL", "random"):
        for b in VAL_BUDGETS:
            knn = table.query("method == @name + ' k-NN' and budget == @b")["mean"].item()
            probe = table.query("method == @name + ' probe' and budget == @b")["mean"].item()
            gap = 100 * (probe - knn)
            if gap > 10:
                flags.append(f"{name}, budget {b}: probe beats k-NN by {gap:.1f} points (> 10)")
            elif gap < 0:
                flags.append(f"{name}, budget {b}: k-NN beats probe by {-gap:.1f} points")
    return flags


def plot_val_curve(table, path=Path("results/val_budget_curve.png")):
    """Val accuracy vs labels per class, mean with std bars, majority-class dashed line. VAL only."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(VAL_BUDGETS))
    for i, (method, (color, style, marker)) in enumerate(VAL_STYLES.items()):
        t = table[table["method"] == method].set_index("budget").loc[list(VAL_BUDGETS)]
        xs = x + (i - 2) * 0.05                              # small offset so std bars do not overlap
        ax.errorbar(xs, t["mean"], yerr=t["std"], color=color, ls=style, marker=marker, ms=8, lw=2,
                    capsize=4, elinewidth=1.5, label=method)
        ax.annotate(f"{method} {t['mean'].iloc[-1]:.3f}", (xs[-1], t["mean"].iloc[-1]),
                    xytext=(12, VAL_LABEL_NUDGE.get(method, 0)), textcoords="offset points",
                    va="center", fontsize=12, color="#333333")
    val, train = split_counts("val"), split_counts("train")
    majority, major_name = majority_acc(val), "defect" if val["defect"] >= val["ok"] else "ok"
    ax.axhline(majority, color="#888888", ls="--", lw=1.5)
    ax.text(x[0] - 0.25, majority + 0.006, f"majority class ({major_name}) {majority:.1%}", fontsize=12, color="#555555")
    ax.set_xticks(x, ["10", "25", "50", f"all\n({train['ok']} ok / {train['defect']} defect)"], fontsize=13)
    ax.set_xlim(-0.35, len(x) - 1 + 1.25)
    ax.set_ylim(0.55, 0.95)
    ax.set_xlabel("labeled images per class", fontsize=14)
    ax.set_ylabel(f"VAL accuracy ({val['n']} images)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    ax.set_title("VAL: accuracy vs label budget (selection only, not the reported result)\n"
                 "mean +/- std over 3 draws; 'all' is 1 draw", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_val_selection(path=Path("results/val_selection.md")):
    """Write results/val_selection.md and results/val_budget_curve.png. VAL only."""
    table = val_curve_table()
    plot_val_curve(table)
    flags = probe_knn_flags(table)
    wide = table.assign(cell=[f"{m:.3f} +/- {s:.3f}" if d > 1 else f"{m:.3f}"
                              for m, s, d in zip(table["mean"], table["std"], table["draws"])])
    wide = wide.pivot(index="method", columns="budget", values="cell").loc[list(VAL_STYLES), list(VAL_BUDGETS)]
    val = split_counts("val")
    major_name = "defect" if val["defect"] >= val["ok"] else "ok"
    lines = ["# VAL selection results (selection only, not the reported result)", "",
             f"These are VALIDATION numbers. Val ({val['n']} images: {val['ok']} ok, {val['defect']} defect) "
             "chose k, C, the anomaly k,",
             "the threshold, and the supervised best epoch, so every number here is optimistic.",
             "The reported result is the single test run in eval.py.", "",
             f"Majority class (always '{major_name}') on val: {majority_acc(val):.1%}. "
             f"One val image = {100 / val['n']:.2f} points.", "",
             "## VAL accuracy by labels per class (mean +/- std over 3 draws; 'all' = 1 draw)", "",
             "| method | " + " | ".join(VAL_BUDGETS) + " |",
             "|---|" + "---|" * len(VAL_BUDGETS)]
    lines += [f"| {m} | " + " | ".join(wide.loc[m]) + " |" for m in wide.index]
    lines += ["", "## Flags (VAL)", ""]
    lines += [f"- {f}" for f in flags] or ["- none"]
    n_wide = sum(1 for f in flags if "(> 10)" in f)
    lines += ["", f"Budgets where the probe beats k-NN by > 10 points on VAL: {n_wide}. "
              f"Budgets where k-NN beats the probe on VAL: {len(flags) - n_wide}.", "",
              "![VAL budget curve](val_budget_curve.png)", ""]
    anom = anomaly_table()
    lines += ["## Anomaly score (VAL)", "",
              "Score = cosine distance of a val image to the labeled ok embeddings of the budget; the option "
              "(local k, all_ok, centroid) and the threshold (catch 95% of val defects) are chosen on val.", "",
              "| encoder | budget | AUROC | false rejects at 95% defect recall | chosen option per draw |",
              "|---|---|---|---|---|"]
    lines += [f"| {r['encoder']} | {r['budget']} | {r['AUROC']} | {r['false rej']} | {r['chosen']} |"
              for r in anom]
    lines += ["", "### Grid change after seeing val", "", ANOMALY_GRID_NOTE, ""]
    path.write_text("\n".join(lines), encoding="ascii")
    print(wide.to_string())
    print("\nanomaly (VAL):")
    print(pd.DataFrame(anom).to_string(index=False))
    print("\nflags (VAL):")
    for f in flags:
        print(f"  {f}")
    print(f"wrote {path} and results/val_budget_curve.png")


def anomaly_table():
    """Val AUROC and false-reject rate (mean +/- std over draws) and chosen option per encoder and budget."""
    rows = []
    for enc, draws in json.loads(CHOICES_PATH.read_text())["encoders"].items():
        df = pd.DataFrame(draws)
        for b, g in df.groupby("budget", sort=False):
            ms = (lambda c: f"{g[c].mean():.3f} +/- {g[c].std(ddof=1):.3f}") if len(g) > 1 else (lambda c: f"{g[c].mean():.3f}")
            rows.append({"encoder": enc, "budget": str(b), "AUROC": ms("val_auroc"),
                         "false rej": ms("val_false_reject_rate"),
                         "chosen": "/".join(map(str, g["anomaly_option"]))})
    return rows


def sha256_of(path):
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def supervised_weight_files():
    """The supervised weights eval.py will score: draws 0-2 for 10/25/50 and draw 0 for all, at the chosen lr."""
    log = load_sup_log()
    files = {}
    for budget in SUP_BUDGETS:
        draws = [0] if budget == "all" else [0, 1, 2]
        for draw in draws:
            run = log["runs"][f"{budget}_{draw}"]
            if run["lr"] != log["chosen_lr"][budget]:
                raise SystemExit(f"sup {budget}_{draw} trained at lr {run['lr']}, chosen lr is {log['chosen_lr'][budget]}")
            files[f"sup_{budget}_{draw}"] = {"path": run["weights"], "lr": run["lr"]}
        # draw 0 is a copy of the winning lr-search candidate; they must be byte-identical
        cand = log["lr_search"][budget][lr_key(log["chosen_lr"][budget])]["weights"]
        if sha256_of(cand) != sha256_of(files[f"sup_{budget}_0"]["path"]):
            raise SystemExit(f"sup_{budget}_0.pt differs from its chosen lr candidate {cand}")
    return files


def freeze(path=MANIFEST_PATH):
    """Hash every frozen artifact into results/frozen_manifest.json; eval.py refuses to run if any hash changes."""
    choices = json.loads(CHOICES_PATH.read_text())
    needed = {"test_strata", "grid_note"}
    per_draw = {"knn_k", "probe_C", "anomaly_option", "threshold"}
    if not needed <= set(choices) or any(not per_draw <= set(d) for v in choices["encoders"].values() for d in v):
        raise SystemExit("frozen_choices.json is missing strata, the grid note, or a per-draw choice; re-run --tune/--strata")
    files = {"ssl_encoder": {"path": str(SSL_WEIGHTS)},
             **supervised_weight_files(),
             "budgets": {"path": str(BUDGET_PATH)},
             "frozen_choices": {"path": str(CHOICES_PATH)},
             # the splits themselves, so a silent re-split is also caught
             **{f"split_{s}": {"path": f"splits/{s}.csv"} for s in ("train", "val", "test", "pretrain")}}
    for entry in files.values():
        entry["sha256"] = sha256_of(entry["path"])
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "random_init_encoder": "resnet18(weights=None), torch.manual_seed(42); no file, rebuilt by seed",
                "rule": "eval.py recomputes every sha256 and refuses to run if any differs",
                "files": files}
    path.write_text(json.dumps(manifest, indent=2))
    print(f"frozen {len(files)} files at {manifest['frozen_at_utc']} -> {path}")
    for name, e in files.items():
        print(f"  {name:<16} {e['sha256'][:16]}...  {e['path']}" + (f"  (lr {lr_key(e['lr'])})" if "lr" in e else ""))


def summarize(choices):
    """Print mean +/- std over draws per encoder and budget, with probe vs k-NN flags."""
    rows = []
    for name, draws in choices["encoders"].items():
        df = pd.DataFrame(draws)
        for budget, g in df.groupby("budget", sort=False):
            ms = lambda c: f"{g[c].mean():.3f} +/- {g[c].std(ddof=1):.3f}"
            gap = g["val_probe_acc"].mean() - g["val_knn_acc"].mean()
            flag = "probe >> kNN" if gap > 0.10 else ("kNN > probe" if gap < 0 else "")
            rows.append({"encoder": name, "budget": budget, "draws": len(g),
                         "kNN acc": ms("val_knn_acc"), "probe acc": ms("val_probe_acc"),
                         "AUROC": ms("val_auroc"), "false rej @95% rec": ms("val_false_reject_rate"),
                         "k": "/".join(map(str, g["knn_k"])), "C": "/".join(map(str, g["probe_C"])),
                         "flag": flag})
    print("\nVAL ONLY (tuning; these numbers chose the settings, so they are optimistic)")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    """Make budgets and train the supervised baselines."""
    raise NotImplementedError


if __name__ == "__main__":
    import sys
    if "--budgets" in sys.argv:
        check_budgets(make_budgets())
    elif "--features" in sys.argv:
        print("frozen features, 96 px, no augmentation (train and val only):")
        save_features("ssl", ssl_encoder())
        save_features("random", random_init_encoder(seed=42))
    elif "--tune" in sys.argv:
        summarize(tune_all())
    elif "--freeze" in sys.argv:
        freeze()
    elif "--residual" in sys.argv:
        residual_twins()
    elif "--val-report" in sys.argv:
        write_val_selection()
    elif "--pick-lr" in sys.argv:
        # python baselines.py --pick-lr <budget>: choose lr from the draw-0 search
        pick_lr(sys.argv[sys.argv.index("--pick-lr") + 1])
    elif "--sup" in sys.argv:
        # python baselines.py --sup <budget> --lr <lr>   : draw 0 at one candidate lr (one process each)
        # python baselines.py --sup <budget> [--draw <d>]: draws 1 and 2 (or one draw) at the chosen lr
        budget = sys.argv[sys.argv.index("--sup") + 1]
        if "--lr" in sys.argv:
            lr_search_run(budget, float(sys.argv[sys.argv.index("--lr") + 1]))
        else:
            draws = [int(sys.argv[sys.argv.index("--draw") + 1])] if "--draw" in sys.argv else [1, 2]
            train_remaining_draws(budget, draws)
    else:
        main()
