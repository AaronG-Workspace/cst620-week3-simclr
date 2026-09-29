"""CONTAMINATED control: pretrain on train pool + test, score it with eval.py --contaminated, report the gap."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import data

CONTAM_CSV = Path("splits/pretrain_contaminated.csv")
CONTAM_WEIGHTS = Path("weights/ssl_encoder_contaminated.pt")
CLEAN_METRICS = Path("results/test_metrics.json")
CONTAM_METRICS = Path("results/contaminated/test_metrics.json")
LOCK = Path("results/eval.lock")
GAP_PATH = Path("results/contaminated/leakage_gap.json")
# Contaminated list is 866 + 260 images (~30% more per epoch); the clean run's epoch count must fit.
CONTAM_MAX_SECONDS = 780


def require_clean_results():
    """Refuse to run until the clean test scores exist and the eval lock says complete."""
    if not CLEAN_METRICS.exists() or not LOCK.exists() or json.loads(LOCK.read_text())["status"] != "complete":
        raise SystemExit("REFUSING: clean test results must be written (results/eval.lock complete) first")


def build_contaminated_list():
    """CONTAMINATED: splits/pretrain.csv plus every test path (reads test.csv's path column only)."""
    require_clean_results()
    clean = pd.read_csv("splits/pretrain.csv")
    test = data.load_split("test", allow_test=True, columns=["path"])
    contam = pd.concat([clean, test], ignore_index=True)
    contam.to_csv(CONTAM_CSV, index=False)
    print(f"CONTAMINATED pretrain list: {len(clean)} train pool + {len(test)} test = {len(contam)} -> {CONTAM_CSV}")


def check_list():
    """Run check_splits.py on the contaminated list; it MUST fail (that proves the contamination is real)."""
    code = subprocess.run([sys.executable, "check_splits.py", "--pretrain", str(CONTAM_CSV)]).returncode
    print(f"\ncheck_splits exit code {code}: " + ("FAILED as expected, the list is CONTAMINATED"
                                                  if code else "PASSED, which means the list is NOT contaminated"))
    return code


def pretrain_contaminated():
    """Pretrain with the clean full run's exact config, only the pretraining list changed. Run by the user."""
    import train_ssl
    clean = json.loads(Path("results/pretrain_history.json").read_text())
    cfg = {**clean["config"], "pretrain_csv": str(CONTAM_CSV), "max_seconds": CONTAM_MAX_SECONDS}
    encoder, _, history = train_ssl.pretrain(cfg, tag="contaminated")
    done = len(history) - 1
    if done < clean["config"]["epochs"]:
        print(f"WARNING: CONTAMINATED run stopped at {done} of {clean['config']['epochs']} epochs; gap is not like for like")
    train_ssl.save_weights(encoder, cfg, tag="contaminated")      # -> weights/ssl_encoder_contaminated.pt
    train_ssl.save_history(history, cfg, tag="contaminated")      # -> results/pretrain_history_contaminated.json


def leakage_gap():
    """CONTAMINATED minus clean for every SSL k-NN, probe and anomaly AUROC row (same frozen choices)."""
    clean, contam = (json.loads(p.read_text()) for p in (CLEAN_METRICS, CONTAM_METRICS))
    key = lambda r: (r["encoder"], r["budget"], r["draw"])
    c_rows = {key(r): r for r in clean["frozen_draws"] if r["encoder"] == "ssl"}
    rows = []
    for r in contam["frozen_draws"]:
        if r["encoder"] != "ssl":
            continue
        c = c_rows[key(r)]
        rows.append({"budget": r["budget"], "draw": r["draw"],
                     "knn_acc_gap": r["knn"]["acc"] - c["knn"]["acc"],
                     "probe_acc_gap": r["probe"]["acc"] - c["probe"]["acc"],
                     "auroc_gap": r["anomaly"]["auroc"] - c["anomaly"]["auroc"]})
    df = pd.DataFrame(rows)
    summary = df.groupby("budget", sort=False)[["knn_acc_gap", "probe_acc_gap", "auroc_gap"]].agg(["mean", "std"])
    GAP_PATH.write_text(json.dumps({"label": "CONTAMINATED minus clean (leakage estimate)",
                                    "per_draw": rows}, indent=2))
    print("LEAKAGE ESTIMATE (CONTAMINATED minus clean, test, SSL encoder only)")
    print(summary.round(4).to_string())


def main():
    """--build: list + check (must FAIL); --pretrain: user-run (> 8 min); --gap: after eval.py --contaminated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="write the CONTAMINATED list and run check_splits on it")
    parser.add_argument("--pretrain", action="store_true", help="CONTAMINATED pretraining (user starts it, > 8 min)")
    parser.add_argument("--gap", action="store_true", help="leakage gap after python eval.py --contaminated")
    args = parser.parse_args()
    if args.build:
        build_contaminated_list()
        check_list()
    elif args.pretrain:
        require_clean_results()
        pretrain_contaminated()
    elif args.gap:
        leakage_gap()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
