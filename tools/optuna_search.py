"""
Optuna-based Hyperparameter Optimization for UA-KF OVTracker
=============================================================

Bayesian (TPE) search with crash resilience, memory safety, and
automatic seeding from Round 1 results.

Features:
  - Optuna TPE sampler: learns from past trials, converges fast
  - SQLite storage: survives crashes/reboots, auto-resumes
  - Memory guard: skips trial if available RAM < 20 GB
  - All 13 tracker/motion params searched
  - Seeds best-known config from Round 1 as trial 0

Usage (Anaconda Prompt):
  conda activate ovtrack
  pip install optuna
  python tools/optuna_search.py --trials 25

  # Resume after crash/reboot (same command, it just continues):
  python tools/optuna_search.py --trials 25
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Install optuna if missing
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Installing optuna...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna
    from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_CONFIG = "configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py"
CHECKPOINT  = "saved_models/pretrained_models/ovtrack_clip_distillation.pth"
OUTPUT_DIR  = "optuna_results"
DB_PATH     = os.path.join(OUTPUT_DIR, "optuna_study.db")
STUDY_NAME  = "uakf_ovtrack_teta"

METRIC_COLS = ["TETA", "LocA", "AssocA", "ClsA",
               "LocRe", "LocPr", "AssocRe", "AssocPr", "ClsRe", "ClsPr"]

# Minimum available RAM in GB before starting a trial
MIN_RAM_GB = 20.0


# ---------------------------------------------------------------------------
# Memory safety
# ---------------------------------------------------------------------------
def get_available_ram_gb():
    """Get available RAM in GB (Windows only)."""
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 999.0  # Can't check, assume OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cfg_key_to_override(param, value):
    parts = param.split(".", 1)
    prefix, rest = parts[0], parts[1]
    path = f"model.{prefix}.{rest}" if prefix in ("tracker", "motion") else param
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


def run_eval(trial_id, params, trial_dir):
    """Run a single evaluation. Returns combined TETA or None on failure."""
    # Resolve project root (parent of tools/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(trial_dir, exist_ok=True)
    abs_trial_dir = os.path.abspath(trial_dir)
    log_file = os.path.join(abs_trial_dir, "eval_log.txt")
    cfg_opts = build_cfg_options(params)

    # Match the exact command pattern that worked in Round 1:
    # conda run -n ovtrack python tools/test.py ... > logfile 2>&1
    cmd = [
        "conda", "run", "-n", "ovtrack", "python",
        "tools/test.py", BASE_CONFIG, CHECKPOINT,
        "--eval", "track", "--eval-options",
        f"resfile_path={abs_trial_dir}", "--cfg-options", *cfg_opts.split(),
    ]

    print(f"\n{'='*70}")
    print(f"  Trial {trial_id}  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  RAM: {get_available_ram_gb():.1f} GB available")
    print(f"  Params: {json.dumps(params, indent=2)}")
    print(f"  Log: {log_file}")
    print(f"{'='*70}")

    t0 = time.time()
    # Run from project root with PYTHONPATH set as safety belt
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_file, "w", encoding="utf-8") as log:
        ret = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=project_root,
            env=env,
            check=False,
        )
    elapsed = time.time() - t0

    # Force garbage collection between trials
    gc.collect()

    if ret.returncode != 0:
        print(f"  [FAILED] exit code {ret.returncode} ({elapsed/60:.1f} min)")
        return None

    metrics = parse_teta_from_log(log_file)
    teta = metrics.get("combined_TETA")
    assoc = metrics.get("combined_AssocA")

    if teta is None:
        print(f"  [WARNING] No metrics parsed ({elapsed/60:.1f} min)")
        return None

    print(f"  OK  TETA={teta:.3f}  AssocA={assoc:.3f}  "
          f"Base={metrics.get('base_TETA', 0):.3f}  Novel={metrics.get('novel_TETA', 0):.3f}  "
          f"({elapsed/60:.1f} min)")

    return teta


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial):
    """Optuna objective function: maximize TETA."""

    # --- Memory guard ---
    avail = get_available_ram_gb()
    if avail < MIN_RAM_GB:
        print(f"\n  MEMORY GUARD: only {avail:.1f} GB free (need {MIN_RAM_GB}). "
              f"Waiting 60s...")
        gc.collect()
        time.sleep(60)
        avail = get_available_ram_gb()
        if avail < MIN_RAM_GB:
            raise optuna.TrialPruned(f"Insufficient RAM: {avail:.1f} GB")

    # --- Sample all 13 params ---
    params = {
        # Tier 1: Association / matching
        "tracker.match_score_thr":
            trial.suggest_float("match_score_thr", 0.40, 0.60, step=0.05),
        "tracker.momentum_embed":
            trial.suggest_float("momentum_embed", 0.20, 0.50, step=0.05),
        "tracker.motion_weight":
            trial.suggest_float("motion_weight", 0.01, 0.10, step=0.01),
        "tracker.motion_weight_max":
            trial.suggest_float("motion_weight_max", 0.15, 0.40, step=0.05),
        "tracker.w_entropy":
            trial.suggest_float("w_entropy", 0.30, 0.70, step=0.05),

        # Tier 2: Kalman filter
        "tracker.w_margin":
            trial.suggest_float("w_margin", 0.15, 0.35, step=0.05),
        "tracker.w_score":
            trial.suggest_float("w_score", 0.05, 0.25, step=0.05),
        "tracker.ema_gamma":
            trial.suggest_float("ema_gamma", 0.3, 0.6, step=0.1),
        "motion.beta":
            trial.suggest_float("beta", 2.0, 8.0, step=1.0),
        "motion.max_scale":
            trial.suggest_float("max_scale", 4.0, 15.0, step=1.0),
        "motion.mapping":
            trial.suggest_categorical("mapping", ["linear", "exponential"]),

        # Tier 3: Memory / gating
        "tracker.num_frames_retain":
            trial.suggest_int("num_frames_retain", 30, 80, step=15),
        "tracker.appearance_gate_min":
            trial.suggest_float("appearance_gate_min", 0.03, 0.10, step=0.01),
    }

    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial.number:03d}")
    teta = run_eval(trial.number, params, trial_dir)

    if teta is None:
        raise optuna.TrialPruned("Eval failed")

    # Store extra metrics as user attrs for analysis
    metrics = parse_teta_from_log(os.path.join(trial_dir, "eval_log.txt"))
    for k, v in metrics.items():
        trial.set_user_attr(k, v)

    return teta


# ---------------------------------------------------------------------------
# Seed best known configs from Round 1
# ---------------------------------------------------------------------------
def seed_prior_knowledge(study):
    """Enqueue the best known configs so Optuna evaluates them first."""
    # Round 1 Trial 2 (TETA=39.46, AssocA=48.39)
    study.enqueue_trial({
        "match_score_thr": 0.50,
        "momentum_embed": 0.30,
        "motion_weight": 0.05,
        "motion_weight_max": 0.20,
        "w_entropy": 0.40,
        "w_margin": 0.30,
        "w_score": 0.10,
        "ema_gamma": 0.5,
        "beta": 5.0,
        "max_scale": 5.0,
        "mapping": "exponential",
        "num_frames_retain": 45,
        "appearance_gate_min": 0.08,
    })

    # Original UA config (TETA=39.10, AssocA=47.37)
    study.enqueue_trial({
        "match_score_thr": 0.50,
        "momentum_embed": 0.35,
        "motion_weight": 0.03,
        "motion_weight_max": 0.30,
        "w_entropy": 0.60,
        "w_margin": 0.25,
        "w_score": 0.15,
        "ema_gamma": 0.4,
        "beta": 3.0,
        "max_scale": 10.0,
        "mapping": "linear",
        "num_frames_retain": 60,
        "appearance_gate_min": 0.05,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna UA-KF Hyperparameter Search")
    parser.add_argument("--trials", type=int, default=25,
                        help="Total number of trials (including already completed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    storage = f"sqlite:///{DB_PATH}"

    # Create or load study (crash-resilient via SQLite)
    sampler = TPESampler(seed=args.seed, n_startup_trials=3)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",  # maximize TETA
        sampler=sampler,
        load_if_exists=True,   # resume after crash
    )

    # Only seed if study is brand new (no completed trials)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        seed_prior_knowledge(study)
        print("Seeded 2 best-known configs as starting points")

    remaining = args.trials - len(completed)
    print(f"\nStudy '{STUDY_NAME}': {len(completed)} done, {remaining} remaining")
    print(f"Storage: {DB_PATH}")
    print(f"RAM available: {get_available_ram_gb():.1f} GB (guard: {MIN_RAM_GB} GB)\n")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    # Print summary
    print(f"\n{'='*70}")
    print("SEARCH COMPLETE")
    print(f"{'='*70}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        print(f"\nBest trial : #{best.number}")
        print(f"Best TETA  : {best.value:.3f}")
        print(f"Best AssocA: {best.user_attrs.get('combined_AssocA', 'N/A')}")
        print(f"Best LocA  : {best.user_attrs.get('combined_LocA', 'N/A')}")
        print(f"Base TETA  : {best.user_attrs.get('base_TETA', 'N/A')}")
        print(f"Novel TETA : {best.user_attrs.get('novel_TETA', 'N/A')}")
        print(f"\nBest params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        # Save best config
        with open(os.path.join(OUTPUT_DIR, "best_config.txt"), "w") as f:
            f.write(f"Best trial: #{best.number}\n")
            f.write(f"TETA:       {best.value:.3f}\n")
            f.write(f"AssocA:     {best.user_attrs.get('combined_AssocA', 'N/A')}\n")
            f.write(f"LocA:       {best.user_attrs.get('combined_LocA', 'N/A')}\n")
            f.write(f"Base TETA:  {best.user_attrs.get('base_TETA', 'N/A')}\n")
            f.write(f"Novel TETA: {best.user_attrs.get('novel_TETA', 'N/A')}\n\n")
            f.write("Params:\n")
            for k, v in best.params.items():
                f.write(f"  {k}: {v}\n")
            # Build cfg-options for direct use
            param_map = {}
            for k, v in best.params.items():
                # Reverse-map Optuna param names to cfg keys
                if k in ("beta", "max_scale", "mapping"):
                    param_map[f"motion.{k}"] = v
                else:
                    param_map[f"tracker.{k}"] = v
            f.write(f"\n-- cfg-options string --\n")
            f.write(build_cfg_options(param_map) + "\n")

        # Print all trials ranked
        print(f"\nAll {len(completed)} trials ranked:")
        ranked = sorted(completed, key=lambda t: t.value or 0, reverse=True)
        print(f"  {'#':>4}  {'TETA':>7}  {'AssocA':>7}  {'LocA':>7}")
        for t in ranked:
            assoc = t.user_attrs.get('combined_AssocA', 0)
            loc = t.user_attrs.get('combined_LocA', 0)
            print(f"  {t.number:>4}  {t.value:>7.3f}  {assoc:>7.3f}  {loc:>7.3f}")
    else:
        print("No completed trials.")


if __name__ == "__main__":
    main()
