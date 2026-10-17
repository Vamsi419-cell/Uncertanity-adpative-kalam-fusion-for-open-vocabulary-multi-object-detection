"""
Focused Hyperparameter Search — Round 2
========================================
Narrow grid around Trial 2's best params from Round 1.

Only Tier-1 params (highest impact on AssocA/TETA) are varied.
Tier-2/3 params are fixed at Trial 2's proven values.

Expected: 10 trials × ~2-3 hrs each (with 20-core TETA parallelism) ≈ 20-30 hrs.

Usage:
  conda run -n ovtrack python tools/hparam_search_focused.py > hparam_results_r2/log.txt 2>&1
"""

import argparse
import csv
import itertools
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime

BASE_CONFIG = "configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py"
CHECKPOINT  = "saved_models/pretrained_models/ovtrack_clip_distillation.pth"
CONDA_ENV   = "ovtrack"
OUTPUT_DIR  = "hparam_results_r2"
LOG_CSV     = os.path.join(OUTPUT_DIR, "search_log.csv")
BEST_TXT    = os.path.join(OUTPUT_DIR, "best_config.txt")

METRIC_COLS = ["TETA", "LocA", "AssocA", "ClsA",
               "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr"]

# -----------------------------------------------------------------------
# Trial 2 best (our anchor point)
# -----------------------------------------------------------------------
#   match_score_thr=0.5, num_frames_retain=45, momentum_embed=0.3,
#   motion_weight=0.05, motion_weight_max=0.2,
#   w_entropy=0.4, w_margin=0.3, w_score=0.1,
#   ema_gamma=0.5, appearance_gate_min=0.08,
#   beta=5.0, max_scale=5.0, mapping=exponential
#   -> TETA=39.46, AssocA=48.39

# Fixed params (Tier 2/3 — locked at Trial 2 values)
FIXED_PARAMS = {
    "tracker.num_frames_retain":    45,
    "tracker.ema_gamma":            0.5,
    "tracker.appearance_gate_min":  0.08,
    "tracker.w_score":              0.1,
    "motion.mapping":               "exponential",
}

# Tier-1 search: tight grid ±1 step around Trial 2's best
SEARCH_GRID = {
    "tracker.match_score_thr":  [0.45, 0.50, 0.55],
    "tracker.momentum_embed":   [0.25, 0.30, 0.35],
    "tracker.motion_weight":    [0.03, 0.05, 0.07],
    "tracker.motion_weight_max":[0.15, 0.20, 0.25],
    "tracker.w_entropy":        [0.35, 0.40, 0.50],
    "tracker.w_margin":         [0.25, 0.30, 0.35],
    "motion.beta":              [3.0, 5.0, 7.0],
    "motion.max_scale":         [4.0, 5.0, 7.0],
}

# Full grid = 3^8 = 6561 combos — way too many.
# We use Latin Hypercube-style sampling: pick ~10 well-spread combos.
# Strategy: fix all but 2 params at anchor, sweep pairs.

def generate_focused_trials():
    """Generate a small set of targeted trials."""
    anchor = {
        "tracker.match_score_thr":  0.50,
        "tracker.momentum_embed":   0.30,
        "tracker.motion_weight":    0.05,
        "tracker.motion_weight_max":0.20,
        "tracker.w_entropy":        0.40,
        "tracker.w_margin":         0.30,
        "motion.beta":              5.0,
        "motion.max_scale":         5.0,
    }

    trials = []

    # Trial 0: Anchor (reproduce Trial 2 to confirm)
    trials.append(dict(anchor))

    # Trials 1-3: Sweep match_score_thr (most impactful single param)
    for v in [0.45, 0.55]:
        t = dict(anchor); t["tracker.match_score_thr"] = v; trials.append(t)

    # Trials 3-4: Sweep motion fusion weights together
    trials.append({**anchor, "tracker.motion_weight": 0.03, "tracker.motion_weight_max": 0.15})
    trials.append({**anchor, "tracker.motion_weight": 0.07, "tracker.motion_weight_max": 0.25})

    # Trials 5-6: Sweep momentum_embed (appearance aggressiveness)
    for v in [0.25, 0.35]:
        t = dict(anchor); t["tracker.momentum_embed"] = v; trials.append(t)

    # Trials 7-8: Sweep uncertainty composition (entropy vs margin)
    trials.append({**anchor, "tracker.w_entropy": 0.50, "tracker.w_margin": 0.25})
    trials.append({**anchor, "tracker.w_entropy": 0.35, "tracker.w_margin": 0.35})

    # Trial 9: Sweep Kalman R shape
    trials.append({**anchor, "motion.beta": 3.0, "motion.max_scale": 4.0})

    # Trial 10: Aggressive Kalman R
    trials.append({**anchor, "motion.beta": 7.0, "motion.max_scale": 7.0})

    # Trial 11: "Best guess" combo — push AssocA further
    trials.append({
        "tracker.match_score_thr":  0.50,
        "tracker.momentum_embed":   0.25,
        "tracker.motion_weight":    0.05,
        "tracker.motion_weight_max":0.25,
        "tracker.w_entropy":        0.50,
        "tracker.w_margin":         0.25,
        "motion.beta":              5.0,
        "motion.max_scale":         5.0,
    })

    return trials


def cfg_key_to_override(param, value):
    parts = param.split(".", 1)
    prefix, rest = parts[0], parts[1]
    if prefix in ("tracker", "motion"):
        path = f"model.{prefix}.{rest}"
    else:
        path = param
    return f"{path}={value}"


def build_cfg_options(params):
    return " ".join(cfg_key_to_override(k, v) for k, v in params.items())


def parse_teta_from_log(log_path):
    results = {}
    if not os.path.isfile(log_path):
        return results
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for line in lines:
        s = line.strip()
        if s.startswith("COMBINED"):
            nums = re.findall(r"[\d.]+", s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"combined_{col}"] = float(nums[j])
        elif s.startswith("Base"):
            nums = re.findall(r"[\d.]+", s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"base_{col}"] = float(nums[j])
        elif s.startswith("Novel"):
            nums = re.findall(r"[\d.]+", s)
            if len(nums) >= len(METRIC_COLS):
                for j, col in enumerate(METRIC_COLS):
                    results[f"novel_{col}"] = float(nums[j])
    return results


def run_trial(trial_id, params, trial_dir):
    os.makedirs(trial_dir, exist_ok=True)
    log_file = os.path.join(trial_dir, "eval_log.txt")
    cfg_opts = build_cfg_options(params)

    cmd = [
        "conda", "run", "-n", CONDA_ENV, "python",
        "tools/test.py", BASE_CONFIG, CHECKPOINT,
        "--eval", "track", "--eval-options", f"resfile_path={trial_dir}",
        "--cfg-options", *cfg_opts.split(),
    ]

    print(f"\n{'='*70}")
    print(f"  Trial {trial_id:03d}  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  Varied params:")
    for k, v in params.items():
        if k not in FIXED_PARAMS:
            print(f"    {k}: {v}")
    print(f"  Log: {log_file}")
    print(f"{'='*70}")

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as log:
        ret = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"  [FAILED] return code {ret.returncode} ({elapsed/60:.1f} min)")
        return {"status": "failed", "elapsed_s": elapsed}

    metrics = parse_teta_from_log(log_file)
    if not metrics:
        print(f"  [WARNING] No metrics parsed from {log_file}")
        return {"status": "no_metrics", "elapsed_s": elapsed}

    teta = metrics.get("combined_TETA", 0.0)
    assoc = metrics.get("combined_AssocA", 0.0)
    print(f"  ✓ TETA={teta:.3f}  AssocA={assoc:.3f}  ({elapsed/60:.1f} min)")
    metrics["status"] = "ok"
    metrics["elapsed_s"] = elapsed
    return metrics


def append_to_csv(csv_path, trial_id, params, metrics):
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


def save_best(best_trial):
    with open(BEST_TXT, "w") as f:
        f.write(f"Best trial: {best_trial['trial_id']}\n")
        f.write(f"TETA:       {best_trial['metrics'].get('combined_TETA', 'N/A')}\n")
        f.write(f"AssocA:     {best_trial['metrics'].get('combined_AssocA', 'N/A')}\n")
        f.write(f"LocA:       {best_trial['metrics'].get('combined_LocA', 'N/A')}\n")
        f.write(f"Base TETA:  {best_trial['metrics'].get('base_TETA', 'N/A')}\n")
        f.write(f"Novel TETA: {best_trial['metrics'].get('novel_TETA', 'N/A')}\n\n")
        f.write("All params (including fixed):\n")
        for k, v in best_trial["params"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n-- cfg-options string (copy-paste ready) --\n")
        f.write(build_cfg_options(best_trial["params"]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trials = generate_focused_trials()

    # Merge fixed params into each trial
    for t in trials:
        t.update(FIXED_PARAMS)

    print(f"Focused Round 2: {len(trials)} trials")
    print(f"Fixed params: {json.dumps(FIXED_PARAMS, indent=2)}")

    completed_ids = set()
    if args.resume and os.path.isfile(LOG_CSV):
        with open(LOG_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                completed_ids.add(int(row["trial_id"]))
        print(f"Resuming: {len(completed_ids)} already done")

    best = {"trial_id": -1, "teta": -1.0, "params": {}, "metrics": {}}

    for trial_id, params in enumerate(trials):
        if trial_id in completed_ids:
            print(f"  Skipping trial {trial_id} (done)")
            continue

        trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial_id:03d}")
        metrics = run_trial(trial_id, params, trial_dir)
        append_to_csv(LOG_CSV, trial_id, params, metrics)

        teta = metrics.get("combined_TETA", -1.0)
        if isinstance(teta, str):
            teta = -1.0
        if teta > best["teta"]:
            best = {"trial_id": trial_id, "teta": teta,
                    "params": params, "metrics": metrics}
            save_best(best)

    print(f"\n{'='*70}")
    print("ROUND 2 COMPLETE")
    print(f"Best trial : {best['trial_id']}")
    print(f"Best TETA  : {best['teta']:.3f}")
    print(f"Best AssocA: {best['metrics'].get('combined_AssocA', 'N/A')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
