"""Rerun everything from the locked split to the VAL report, one process per stage; stops before freeze/eval."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TAUS = ("0.1", "0.2", "0.5")
SUP_BUDGETS = ("10", "25", "50", "all")
LR_GRID = ("0.0001", "0.0003", "0.001")
STAGE_LIMIT_S = 600          # CLAUDE.md: a stage inside run.py may take up to 10 minutes
FROM_ALIASES = {"tau": "sweep tau 0.1"}   # --from tau restarts at the tau sweep
# Stages that must not rerun when the supervised weights are reused: the models were trained on these budgets.
SUP_DEPENDENT = ("budgets.json",)
TIMING_PATH = Path("results/run_timing.json")
# Logs that later stages append to; cleared when their first stage runs so stale runs never mix in.
STALE = {"sweep tau 0.1": [Path("results/tau_sweep.json")],
         "sup 10 lr 0.0001": [Path("results/supervised_log.json")]}


def build_stages():
    """Ordered (name, argv) list; every entry runs in its own python process."""
    stages = [("check_splits", ["check_splits.py"])]
    stages += [(f"sweep tau {t}", ["train_ssl.py", "--sweep-tau", t]) for t in TAUS]
    stages += [("pick tau", ["train_ssl.py", "--pick-tau"]),
               ("pretrain (full, last epoch kept)", ["train_ssl.py"]),   # epochs sized to ~8 min from sweep times
               ("budgets.json", ["baselines.py", "--budgets"]),
               ("features", ["baselines.py", "--features"]),
               ("tune (VAL)", ["baselines.py", "--tune"])]
    for b in SUP_BUDGETS:
        stages += [(f"sup {b} lr {lr}", ["baselines.py", "--sup", b, "--lr", lr]) for lr in LR_GRID]
        stages += [(f"sup {b} pick lr", ["baselines.py", "--pick-lr", b])]
        if b != "all":   # "all" has a single draw (0), already trained in its lr search
            stages += [(f"sup {b} draw {d}", ["baselines.py", "--sup", b, "--draw", str(d)]) for d in (1, 2)]
    stages += [("val report", ["baselines.py", "--val-report"])]
    return stages


def run_stage(name, argv):
    """Run one stage in a fresh process; return (seconds, exit code)."""
    for path in STALE.get(name, []):
        if path.exists():
            path.unlink()
            print(f"[run.py] removed stale {path} (pre-fix copy is in results/pre_fix/)")
    print(f"\n[run.py] === {name}: python {' '.join(argv)}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.run([sys.executable, *argv]).returncode
    return time.perf_counter() - t0, code


def print_timing(rows):
    """Stage timing table; flags any stage over the 10-minute stage limit."""
    print("\n[run.py] stage timing")
    print(f"  {'#':>2}  {'stage':<34} {'seconds':>8} {'min':>6}  status")
    for i, r in enumerate(rows, 1):
        flag = "  OVER 10 MIN" if r["seconds"] > STAGE_LIMIT_S else ""
        print(f"  {i:>2}  {r['stage']:<34} {r['seconds']:>8.1f} {r['seconds'] / 60:>6.1f}  {r['status']}{flag}")
    total = sum(r["seconds"] for r in rows)
    print(f"      {'TOTAL':<34} {total:>8.1f} {total / 60:>6.1f}")


def main():
    """Run the stages in order, stop on the first failure, print timing, stop before freeze/eval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start",
                        help="resume at this stage name, or 'tau' for the tau sweep (earlier stages are skipped)")
    parser.add_argument("--skip-sup", action="store_true",
                        help="reuse existing supervised weights and log: skip sup stages and budgets.json")
    parser.add_argument("--list", action="store_true", help="list the stages that would run and exit")
    args = parser.parse_args()

    stages = build_stages()
    names = [n for n, _ in stages]
    start = FROM_ALIASES.get(args.start, args.start)
    if start and start not in names:
        raise SystemExit(f"unknown stage {args.start!r}; see python run.py --list")
    if start:
        stages = stages[names.index(start):]
    if args.skip_sup:
        needed = [Path("results/supervised_log.json"), Path("splits/budgets.json")]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise SystemExit(f"--skip-sup needs the existing supervised run, missing: {missing}")
        stages = [(n, a) for n, a in stages if not n.startswith("sup ") and n not in SUP_DEPENDENT]
    if args.list:
        for i, (name, argv) in enumerate(stages, 1):
            print(f"  {i:>2}  {name:<34} python {' '.join(argv)}")
        return

    rows = []
    for name, argv in stages:
        seconds, code = run_stage(name, argv)
        rows.append({"stage": name, "seconds": round(seconds, 1), "status": "ok" if code == 0 else f"FAIL ({code})"})
        TIMING_PATH.write_text(json.dumps(rows, indent=2))
        if code != 0:
            print_timing(rows)
            raise SystemExit(f"[run.py] stopped: stage {name!r} failed. Fix it, then: python run.py --from \"{name}\"")
    print_timing(rows)
    print("\n[run.py] done through the VAL report. Stopped before freeze and eval: "
          "review results/val_selection.md first; eval.py scores test once, later.")


if __name__ == "__main__":
    main()
