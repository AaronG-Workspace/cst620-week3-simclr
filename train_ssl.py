"""SimCLR pretraining on the train pool only; tau sweep on val, then one full run."""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
import torch.nn.functional as F

import data
import simclr

CONFIG = {
    "img_size": 96,
    "batch_size": 64,
    "tau": 0.2,
    "lr": 1e-3,
    "epochs": 9,          # full run overrides this with choose_epochs() from measured sweep times
    "max_seconds": 450,   # training-loop budget; with load and save the process stays under 8 minutes
    "sweep_epochs": 3,
    "seed": 42,           # model init and batch shuffling
    "aug_seed": 123,      # augmentation generator only
    "knn_k": 5,
    "pretrain_csv": "splits/pretrain.csv",
}
FULL_SPREAD_STD = 1 / 128 ** 0.5   # 0.088: per-dim std of unit vectors spread evenly over 128 dims
TAUS = (0.1, 0.2, 0.5)
SWEEP_PATH = Path("results/tau_sweep.json")
TARGET_SECONDS = 460      # aim for about 8 minutes of wall time, under the hard cap


def load_pretrain_images(csv_path):
    """Resize the images on the pretraining list once (96 px uint8) and return float (n, 1, H, W)."""
    paths = pd.read_csv(csv_path)["path"]
    return simclr.to_float(data.resize_paths(paths, px=CONFIG["img_size"]))


@torch.no_grad()
def embed(encoder, head, x, batch=128):
    """Encoder features and L2-normalized projections for x, in eval mode, no augmentation."""
    encoder.eval(), head.eval()
    feats = torch.cat([encoder(simclr.to_model_input(x[i:i + batch])) for i in range(0, len(x), batch)])
    return feats, F.normalize(head(feats), dim=1)


def loo_knn_acc(feats, labels, k, groups=None):
    """Cosine k-NN accuracy within one set (val labels only).

    groups=None: plain leave-one-out (only the image itself is excluded).
    groups given: leave-one-group-out, every image with the same group_id is excluded, so a
    rotated twin of the query (same casting, same split) cannot vote.
    """
    f = F.normalize(feats, dim=1)
    sim = f @ f.T
    if groups is None:
        sim.fill_diagonal_(float("-inf"))                          # an image cannot vote for itself
    else:
        sim[groups[:, None] == groups[None, :]] = float("-inf")    # nor can its same-group twins
    nn_idx = sim.topk(k, dim=1).indices
    votes = labels[nn_idx].float().mean(dim=1)     # binary labels, k odd, so no ties
    return ((votes > 0.5).long() == labels).float().mean().item()


def epoch_metrics(encoder, head, val_x, val_y, val_g, k, with_loo=False):
    """Val k-NN on encoder features (leave-one-group-out) and mean per-dim std of normalized projections."""
    feats, z = embed(encoder, head, val_x)
    row = {"val_knn": round(loo_knn_acc(feats, val_y, k, groups=val_g), 4)}
    if with_loo:   # plain leave-one-out, kept at epoch 0 only, for the record
        row["val_knn_loo"] = round(loo_knn_acc(feats, val_y, k), 4)
    row["proj_std"] = round(z.std(dim=0).mean().item(), 4)
    return row


def train_epoch(encoder, head, x, opt, tau, augment, batch_size, shuffle_gen):
    """One pass over x in shuffled full batches (drop last); return mean NT-Xent loss."""
    encoder.train(), head.train()
    order = torch.randperm(len(x), generator=shuffle_gen)
    losses = []
    for i in range(0, len(x) - batch_size + 1, batch_size):
        v1, v2 = simclr.two_views(x[order[i:i + batch_size]], augment)
        z1 = head(encoder(simclr.to_model_input(v1)))
        z2 = head(encoder(simclr.to_model_input(v2)))
        loss = simclr.nt_xent(z1, z2, tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def log_line(row):
    """Print one ASCII history row."""
    loss = "   n/a" if row["loss"] is None else f"{row['loss']:.4f}"
    loo = f"  (val_knn_loo {row['val_knn_loo']:.4f})" if "val_knn_loo" in row else ""
    print(f"  epoch {row['epoch']:>3}  loss {loss}  val_knn {row['val_knn']:.4f}  "
          f"proj_std {row['proj_std']:.4f}  sec {row['seconds']:.1f}{loo}")


def pretrain(cfg, tag="clean"):
    """Pretrain encoder + head on the pretraining list; log epoch 0 (random init) and every epoch after."""
    torch.manual_seed(cfg["seed"])
    x = load_pretrain_images(cfg["pretrain_csv"])
    val_u8, val_labels = data.load_images("val")
    val_x, val_y = simclr.to_float(val_u8), torch.tensor(val_labels)   # copy: pandas array is read-only
    val_g = torch.tensor(data.load_split("val")["group_id"].to_numpy())  # same row order as load_images

    encoder, head = simclr.build_encoder(), simclr.ProjectionHead()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=cfg["lr"])
    augment = simclr.SimCLRAugment(px=cfg["img_size"], aug_seed=cfg["aug_seed"])
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])

    print(f"pretrain [{tag}]: {len(x)} images, batch {cfg['batch_size']}, "
          f"{simclr.negatives_per_anchor(cfg['batch_size'])} negatives per anchor, tau {cfg['tau']}")
    history = [{"epoch": 0, "loss": None, "seconds": 0.0,
                **epoch_metrics(encoder, head, val_x, val_y, val_g, cfg["knn_k"], with_loo=True)}]
    log_line(history[0])
    t_start, slowest = time.perf_counter(), 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        # Hard cap: skip an epoch that would push the run past the time budget.
        if time.perf_counter() - t_start + slowest > cfg["max_seconds"]:
            print(f"  stopping before epoch {epoch}: would exceed {cfg['max_seconds']} s budget")
            break
        t0 = time.perf_counter()
        loss = train_epoch(encoder, head, x, opt, cfg["tau"], augment, cfg["batch_size"], shuffle_gen)
        seconds = time.perf_counter() - t0          # training time only, excludes the val metrics
        history.append({"epoch": epoch, "loss": round(loss, 4), "seconds": round(seconds, 1),
                        **epoch_metrics(encoder, head, val_x, val_y, val_g, cfg["knn_k"])})
        slowest = max(slowest, time.perf_counter() - t0)   # epoch plus its val metrics
        log_line(history[-1])
    return encoder, head, history


def time_one_epoch(cfg):
    """Run 1 epoch and print seconds per epoch, so every run is known to fit under 8 minutes."""
    _, _, history = pretrain({**cfg, "epochs": 1}, tag="timing")
    sec = history[-1]["seconds"]
    print(f"seconds per epoch: {sec:.1f}  (fits {int(8 * 60 // sec)} epochs in 8 minutes, before eval overhead)")
    return sec


def save_weights(encoder, cfg, tag="clean", out_dir="weights"):
    """Save the encoder state dict only (the projection head is never evaluated) to weights/ssl_encoder.pt."""
    suffix = "" if tag == "clean" else f"_{tag}"
    path = Path(out_dir) / f"ssl_encoder{suffix}.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(encoder.state_dict(), path)
    print(f"wrote {path}")
    return path


def choose_epochs(target_seconds, overhead_seconds=15.0, eval_seconds=1.5, path=SWEEP_PATH):
    """Epochs that fit target_seconds, from the mean measured epoch time across all sweep epochs."""
    runs = json.loads(path.read_text())["runs"].values()
    times = [h["seconds"] for r in runs for h in r["history"][1:]]
    mean_sec = sum(times) / len(times)
    epochs = int((target_seconds - overhead_seconds) // (mean_sec + eval_seconds))
    print(f"measured {mean_sec:.1f} s per epoch over {len(times)} sweep epochs (max {max(times):.1f}); "
          f"{epochs} epochs ~ {overhead_seconds + epochs * (mean_sec + eval_seconds):.0f} s")
    return epochs


def save_history(history, cfg, tag="clean", out_dir="results"):
    """Write results/pretrain_history.json (with batch size and negatives) and results/pretrain_curve.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    suffix = "" if tag == "clean" else f"_{tag}"
    trained = history[1:]
    record = {"tag": tag, "config": cfg,
              "batch_size": cfg["batch_size"],
              "negatives_per_anchor": simclr.negatives_per_anchor(cfg["batch_size"]),
              "seconds_per_epoch": round(sum(h["seconds"] for h in trained) / max(len(trained), 1), 1),
              "full_spread_proj_std": round(FULL_SPREAD_STD, 4),
              "history": history}
    json_path = out_dir / f"pretrain_history{suffix}.json"
    json_path.write_text(json.dumps(record, indent=2))

    # Three panels of the same run: loss, val k-NN, collapse check. Epoch 0 = random init.
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot([h["epoch"] for h in trained], [h["loss"] for h in trained], "o-")
    axes[0].set_title("NT-Xent loss", fontsize=15)
    axes[1].plot(epochs, [h["val_knn"] for h in history], "o-")
    axes[1].axhline(history[0]["val_knn"], ls="--", c="gray", label="random init")
    axes[1].set_title(f"val k-NN (leave-one-group-out, k={cfg['knn_k']})", fontsize=15)
    axes[1].legend(fontsize=12)
    axes[2].plot(epochs, [h["proj_std"] for h in history], "o-")
    axes[2].axhline(FULL_SPREAD_STD, ls="--", c="gray", label="fully spread 0.088")
    axes[2].set_ylim(0, FULL_SPREAD_STD * 1.15)
    axes[2].set_title("embedding std (collapse check)", fontsize=15)
    axes[2].set_ylabel("mean per-dim std, L2-norm projection", fontsize=11)
    axes[2].legend(fontsize=12)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole epochs only
        ax.set_xlabel("epoch", fontsize=13)
        ax.tick_params(labelsize=12)
    label = "" if tag == "clean" else f" [{tag.upper()}]"
    fig.suptitle(f"SimCLR pretraining{label}: tau {cfg['tau']}, batch {cfg['batch_size']}, "
                 f"{simclr.negatives_per_anchor(cfg['batch_size'])} negatives per anchor", fontsize=16)
    fig.tight_layout()
    png_path = out_dir / f"pretrain_curve{suffix}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {json_path} and {png_path}")


def sweep_one_tau(cfg, tau, path=SWEEP_PATH):
    """Short run at one tau with the same seeds; add it to results/tau_sweep.json (one process per tau)."""
    _, _, history = pretrain({**cfg, "tau": tau, "epochs": cfg["sweep_epochs"]}, tag=f"sweep tau {tau}")
    sweep = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    sweep["config"] = {k: v for k, v in cfg.items() if k != "tau"}
    sweep["batch_size"] = cfg["batch_size"]
    sweep["negatives_per_anchor"] = simclr.negatives_per_anchor(cfg["batch_size"])
    sweep["runs"][str(tau)] = {"final_val_knn": history[-1]["val_knn"],
                               "final_proj_std": history[-1]["proj_std"],
                               "final_loss": history[-1]["loss"],
                               "seconds_per_epoch": round(sum(h["seconds"] for h in history[1:]) / cfg["sweep_epochs"], 1),
                               "history": history}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(sweep, indent=2))


def pick_tau(path=SWEEP_PATH):
    """Best final val k-NN wins; a tie goes to the higher projection std (further from collapse)."""
    sweep = json.loads(path.read_text())
    runs = sweep["runs"]
    missing = [t for t in map(str, TAUS) if t not in runs]
    if missing:
        raise SystemExit(f"tau sweep incomplete, missing tau {missing}")
    best = max(runs, key=lambda t: (runs[t]["final_val_knn"], runs[t]["final_proj_std"]))
    sweep["best_tau"] = float(best)
    sweep["rule"] = "max final val leave-one-group-out k-NN (k=5, cosine); tie -> higher proj_std"
    path.write_text(json.dumps(sweep, indent=2))

    print(f"\ntau sweep ({sweep['config']['sweep_epochs']} epochs each, batch {sweep['batch_size']}, "
          f"{sweep['negatives_per_anchor']} negatives per anchor)")
    rows = {f"tau {t}": {"val_knn ep0": r["history"][0]["val_knn"],
                         "val_knn final": r["final_val_knn"],
                         "proj_std final": r["final_proj_std"],
                         "loss final": r["final_loss"],
                         "sec/epoch": r["seconds_per_epoch"],
                         "picked": "<--" if t == best else ""}
            for t, r in runs.items()}
    print(pd.DataFrame(rows).T.to_string())
    return float(best)


def sweep_tau(cfg):
    """Short runs at tau 0.1, 0.2, 0.5; return the tau with the best val k-NN."""
    for tau in TAUS:
        sweep_one_tau(cfg, tau)
    return pick_tau()


def main(cfg=CONFIG):
    """Full clean run at the tau picked by the sweep; save weights first, then history and curve."""
    if not SWEEP_PATH.exists() or "best_tau" not in json.loads(SWEEP_PATH.read_text()):
        raise SystemExit("run the tau sweep first (--sweep-tau for each tau, then --pick-tau)")
    cfg = {**cfg, "tau": json.loads(SWEEP_PATH.read_text())["best_tau"],
           "epochs": choose_epochs(TARGET_SECONDS)}
    encoder, _, history = pretrain(cfg, tag="clean")
    save_weights(encoder, cfg, tag="clean")
    save_history(history, cfg, tag="clean")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", action="store_true", help="time one epoch")
    parser.add_argument("--sweep-tau", type=float, help="run the short sweep at one tau (one process per tau)")
    parser.add_argument("--pick-tau", action="store_true", help="pick the best tau from results/tau_sweep.json")
    args = parser.parse_args()
    if args.time:
        time_one_epoch(CONFIG)
    elif args.sweep_tau is not None:
        sweep_one_tau(CONFIG, args.sweep_tau)
    elif args.pick_tau:
        pick_tau()
    else:
        main()
