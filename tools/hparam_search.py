"""
Hyperparameter Search for UA-KF OVTracker
==========================================
Runs a grid or random search over the UAKFOVTracker / AdaptiveKalmanFilter
parameters, evaluating each candidate with TETA on the OVT-B benchmark.

Usage (from project root in the ovtrack conda env):

  # Random search (default 20 trials):
  conda run -n ovtrack python tools/hparam_search.py

  # Grid search:
  conda run -n ovtrack python tools/hparam_search.py --mode grid

  # Custom trial count:
  conda run -n ovtrack python tools/hparam_search.py --mode random --trials 30

  # Resume from a previous log (skips already-completed trials):
  conda run -n ovtrack python tools/hparam_search.py --resume

Results are saved to:
  hparam_results/search_log.csv        - per-trial metrics
  hparam_results/best_config.txt       - best config so far
  hparam_results/trial_<N>/            - per-trial result files

Optimisation target: COMBINED TETA50 (higher is better).
"""

import argparse
import csv
import itertools
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime

# ---------------------------------------------------------------------------
# Search Space Definition
# ---------------------------------------------------------------------------
# Each entry: param_name -> list of candidate values.
# Params map 1-to-1 to cfg-options keys used in the test.py call.
# Prefixes:
#   tracker.*       -> model.tracker.<key>
#   motion.*        -> model.motion.<key>

SEARCH_SPACE = {
    # --- Matching ---
    "tracker.match_score_thr":      [0.4, 0.45, 0.5, 0.55, 0.6],

    # --- Memory / retain ---
    "tracker.num_frames_retain":    [30, 45, 60, 80],

    # --- Appearance EMA ---
    "tracker.momentum_embed":       [0.2, 0.30, 0.35, 0.40, 0.50],

    # --- Motion fusion weights ---
    "tracker.motion_weight":        [0.01, 0.03, 0.05, 0.08],
    "tracker.motion_weight_max":    [0.20, 0.25, 0.30, 0.40],

    # --- Uncertainty estimator weights (must sum to ~1; renormalized internally) ---
    "tracker.w_entropy":            [0.4, 0.5, 0.6, 0.7],
    "tracker.w_margin":             [0.15, 0.20, 0.25, 0.30],
    "tracker.w_score":              [0.10, 0.15, 0.20],

    # --- EMA smoothing for track uncertainty ---
    "tracker.ema_gamma":            [0.3, 0.4, 0.5, 0.6],

    # --- Appearance gate floor ---
    "tracker.appearance_gate_min":  [0.03, 0.05, 0.08, 0.10],

    # --- Kalman R scaling ---
    "motion.beta":                  [2.0, 3.0, 5.0, 8.0],
    "motion.max_scale":             [5.0, 8.0, 10.0, 15.0],
    "motion.mapping":               ["linear", "exponential"],
}

# The baseline config and checkpoint that will be used for every trial.
# (These are the paths used in your successful eval run.)
BASE_CONFIG     = "configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py"
CHECKPOINT      = "saved_models/pretrained_models/ovtrack_clip_distillation.pth"
CONDA_ENV       = "ovtrack"
OUTPUT_DIR      = "hparam_results"
LOG_CSV         = os.path.join(OUTPUT_DIR, "search_log.csv")
BEST_TXT        = os.path.join(OUTPUT_DIR, "best_config.txt")

# TETA metric columns in the order printed by evaluate()
METRIC_COLS = ["TETA", "LocA", "AssocA", "ClsA",
               "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cfg_key_to_override(param: str, value) -> str:
    """Convert search-space key to mmcv cfg-option string.

    tracker.motion_weight  ->  model.tracker.motion_weight=0.03
    motion.beta            ->  model.motion.beta=3.0
    """
    parts = param.split(".", 1)
    prefix = parts[0]
    rest   = parts[1]
    if prefix in ("tracker", "motion"):
        path = f"model.{prefix}.{rest}"
    else:
        path = param
    return f"{path}={value}"


def build_cfg_options(params: dict) -> str:
    parts = [cfg_key_to_override(k, v) for k, v in params.items()]
    return " ".join(parts)


def parse_teta_from_log(log_path: str):
    """Parse COMBINED TETA row + Base/Novel rows from eval_log.txt."""
    results = {}
    if not os.path.isfile(log_path):
        return results

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Find COMBINED line
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("COMBINED"):
            nums = re.findall(r"[\d.]+", line_s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"combined_{col}"] = float(nums[j])
        elif line_s.startswith("Base"):
            nums = re.findall(r"[\d.]+", line_s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"base_{col}"] = float(nums[j])
        elif line_s.startswith("Novel"):
            nums = re.findall(r"[\d.]+", line_s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"novel_{col}"] = float(nums[j])

    return results


def load_existing_trials(csv_path: str):
    """Load already-completed trials from the log CSV."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_random_params(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def generate_grid_params(space: dict):
    keys = list(space.keys())
    for combo in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, combo))


def run_trial(trial_id: int, params: dict, trial_dir: str) -> dict:
    """Run a single evaluation trial. Returns dict of metrics or {} on failure."""
    os.makedirs(trial_dir, exist_ok=True)
    log_file  = os.path.join(trial_dir, "eval_log.txt")
    cfg_opts  = build_cfg_options(params)

    cmd = [
        "conda", "run", "-n", CONDA_ENV, "python",
        "tools/test.py", BASE_CONFIG, CHECKPOINT,
        "--eval", "track", "--eval-options", f"resfile_path={trial_dir}",
        "--cfg-options", *cfg_opts.split(),
    ]

    print(f"\n{'='*70}")
    print(f"  Trial {trial_id:03d}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"  Params: {json.dumps(params, indent=4)}")
    print(f"  Log   : {log_file}")
    print(f"{'='*70}")

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        ret = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"  [FAILED] return code {ret.returncode}")
        return {"status": "failed", "elapsed_s": elapsed}

    metrics = parse_teta_from_log(log_file)
    if not metrics:
        print(f"  [WARNING] Could not parse metrics from {log_file}")
        return {"status": "no_metrics", "elapsed_s": elapsed}

    teta = metrics.get("combined_TETA", 0.0)
    print(f"  TETA={teta:.3f}  AssocA={metrics.get('combined_AssocA', 0):.3f}  "
          f"LocA={metrics.get('combined_LocA', 0):.3f}  "
          f"elapsed={elapsed/60:.1f}min")
    metrics["status"]    = "ok"
    metrics["elapsed_s"] = elapsed
    return metrics


def append_to_csv(csv_path: str, trial_id: int, params: dict, metrics: dict):
    row = {"trial_id": trial_id}
    row.update({f"p_{k}": v for k, v in params.items()})
    row.update(metrics)

    fieldnames = list(row.keys())
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_best(best_trial: dict):
    with open(BEST_TXT, "w") as f:
        f.write(f"Best trial: {best_trial['trial_id']}\n")
        f.write(f"TETA: {best_trial['metrics'].get('combined_TETA', 'N/A')}\n")
        f.write(f"AssocA: {best_trial['metrics'].get('combined_AssocA', 'N/A')}\n")
        f.write(f"LocA: {best_trial['metrics'].get('combined_LocA', 'N/A')}\n")
        f.write(f"Base TETA: {best_trial['metrics'].get('base_TETA', 'N/A')}\n")
        f.write(f"Novel TETA: {best_trial['metrics'].get('novel_TETA', 'N/A')}\n\n")
        f.write("Params:\n")
        for k, v in best_trial["params"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n-- cfg-options string --\n")
        f.write(build_cfg_options(best_trial["params"]) + "\n")
    print(f"\n  [BEST] Saved to {BEST_TXT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UA-KF Hyperparameter Search")
    parser.add_argument("--mode",    choices=["random", "grid"], default="random")
    parser.add_argument("--trials",  type=int, default=20,
                        help="Number of random trials (ignored for grid search)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--resume",  action="store_true",
                        help="Skip trials already logged in search_log.csv")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    # Build candidate list
    if args.mode == "random":
        candidates = [generate_random_params(SEARCH_SPACE, rng)
                      for _ in range(args.trials)]
    else:
        candidates = list(generate_grid_params(SEARCH_SPACE))
        print(f"Grid search: {len(candidates)} total combinations")

    # Optionally skip already-done trials
    completed_ids = set()
    if args.resume:
        for row in load_existing_trials(LOG_CSV):
            completed_ids.add(int(row["trial_id"]))
        print(f"Resuming: {len(completed_ids)} trials already done.")

    best = {"trial_id": -1, "teta": -1.0, "params": {}, "metrics": {}}

    for trial_id, params in enumerate(candidates):
        if trial_id in completed_ids:
            print(f"  Skipping trial {trial_id} (already done)")
            continue

        trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial_id:03d}")
        metrics   = run_trial(trial_id, params, trial_dir)
        append_to_csv(LOG_CSV, trial_id, params, metrics)

        teta = metrics.get("combined_TETA", -1.0)
        if isinstance(teta, str):
            teta = -1.0
        if teta > best["teta"]:
            best = {"trial_id": trial_id, "teta": teta,
                    "params": params, "metrics": metrics}
            save_best(best)

    print("\n" + "="*70)
    print("SEARCH COMPLETE")
    print(f"Best trial : {best['trial_id']}")
    print(f"Best TETA  : {best['teta']:.3f}")
    print(f"Best params: {best['params']}")
    print("="*70)


if __name__ == "__main__":
    main()
