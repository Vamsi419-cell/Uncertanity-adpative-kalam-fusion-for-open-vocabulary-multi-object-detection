# UA-KF-OVT Project Handoff Document

> **Purpose:** This file exists so the next AI agent or developer does NOT need to re-read the entire codebase from scratch. Read this file first. Everything done, everything working, everything broken, everything left to do is here.
>
> **Also see:** `ISSUES_AND_RECOMMENDATIONS.md` at the repo root — detailed read-only audit report from the first session (covers Windows compat issues, dead code analysis, exhaustive config diff, motion module consistency). That document is the audit; this document is the action summary.

---

## 1. Project Identity

**Full Name:** UA-KF-OVT — Uncertainty-Adaptive Kalman Fusion for Open-Vocabulary Multi-Object Tracking  
**Based On:** OVTrack (CVPR 2023) by SysCV — https://github.com/SysCV/ovtrack  
**Repo Root:** `C:\Users\HPCLAB\Documents\OVTRACK\Uncertanity-adpative-kalam-fusion-for-open-vocabulary-multi-object-detection`  
**Python Env:** `C:\Users\HPCLAB\.conda\envs\ovtrack` (Python 3.7.16)  
**Origin:** Developed in Google Colab (Linux), pushed to GitHub, cloned to this Windows machine. Dataset and weights downloaded separately.

---

## 2. What This Project Does

OVTrack is a baseline open-vocabulary multi-object tracker that uses CLIP embeddings for appearance matching across 1053 classes on the OVT-B benchmark. It uses greedy matching and no motion model.

UA-KF-OVT **extends** OVTrack with:
1. **Uncertainty estimation** from CLIP class probability distribution (entropy + margin + score)
2. **Adaptive fusion weight** — when a detection is uncertain, trust IoU/motion more than appearance
3. **Adaptive Kalman filter** — scale measurement noise R based on per-detection uncertainty
4. **Appearance gating** — reduce appearance momentum update for uncertain detections
5. **Per-track EMA uncertainty** — smooth uncertainty over time per track ID

The core claim: uncertain detections should be handled differently from confident ones during association.

---

## 3. Key Files Reference

### Configs (all under `configs/ovtrack-teta/ovtb/`)
| File | Tracker | Motion | Status |
|------|---------|--------|--------|
| `ovtrack_r50.py` | OVTracker (greedy, no Kalman) | None | ✅ Runs, baseline |
| `ovtrack_plus.py` | OVSortTracker (Hungarian + Kalman) | KalmanFilter | ✅ Should run |
| `ovtrack_r50_ua.py` | UAKFOVTracker (Hungarian + adaptive Kalman) | AdaptiveKalmanFilter | ✅ Ready, first run pending |

### Core Model Files
| File | Role |
|------|------|
| `ovtrack/models/mot/ovtrack.py` | Main model. `simple_test()` runs the full inference pipeline. |
| `ovtrack/models/roi_heads/ovtrack_roi_head.py` | Detection + CLIP classification. `simple_test_bboxes()` returns class probs. `simple_test()` computes uncertainty and returns 5-tuple. |
| `ovtrack/models/builder.py` | Local registries: `MODELS`, `TRACKERS`, `MOTIONS`/`MOTION`. NOT mmdet global registries. |

### Tracker Files (all under `ovtrack/models/trackers/`)
| File | Class | Notes |
|------|-------|-------|
| `ovtracker.py` | `OVTracker` | Baseline. Greedy matching. No Kalman. `momentum_embed=0.8`. Ignores `model` kwarg silently. |
| `ovsort_tracker.py` | `OVSortTracker` | Hungarian matching. Has **double Kalman update bug** (see §7). |
| `sort_tracker.py` | `SortTracker` | Base class. Checks `model.with_motion` which OVTrack doesn't implement — only safe because UAKFOVTracker overrides `track()`. |
| `uakf_ovtracker.py` | `UAKFOVTracker` | UA-KF-OVT tracker. Inherits OVSortTracker but overrides both `update_track()` and `track()` — avoids double update bug. |
| `uncertainty_estimator.py` | `UncertaintyEstimator` | Entropy + margin + score uncertainty. EMA per track. |
| `adaptive_kalman.py` | `AdaptiveKalmanFilter` | Extends base KalmanFilter. `compute_r_factor()` scales R by uncertainty. |

### Motion Files (under `ovtrack/models/motions/`)
| File | Notes |
|------|-------|
| `kalman_filter.py` | Standard DeepSORT Kalman filter. |
| `__init__.py` | KalmanFilter import **commented out** intentionally. KalmanFilter still gets registered via side-effect import chain through `trackers/__init__.py` → `adaptive_kalman.py` → `from ..motions.kalman_filter import KalmanFilter`. Do NOT "fix" this without understanding the side-effect. |

### Dataset
| File | Role |
|------|------|
| `ovtrack/datasets/ovtb_dataset.py` | OVT-B dataset loader + TETA evaluation. |
| `data/ovtb/ovtb_ann.json` | 1973 videos / 88,098 images. Confirmed correct. |
| `data/ovtb/ovtb_classes.txt` | 1053 OVT-B class names. |

### Weights
| File | Role |
|------|------|
| `saved_models/pretrained_models/ovtrack_clip_distillation.pth` | Main checkpoint. Loads clean — no missing/unexpected key warnings. |
| `saved_models/pretrained_models/ovtb_class_prompt.pt` | CLIP class prompt tensor. Shape `[1, 1053, 512]`. Confirmed correct. |

---

## 4. Confirmed Working State (All Verified by Import Check)

```
UAKFOVTracker:         REGISTERED in TRACKERS registry ✅
AdaptiveKalmanFilter:  REGISTERED in MOTIONS registry  ✅
UncertaintyEstimator:  Importable                       ✅
cls_probs wired:       simple_test_bboxes + simple_test ✅
ovtrack.py cleaned:    triple self.motion = None → 1    ✅
Checkpoint:            ovtrack_clip_distillation.pth    ✅
Prompt tensor:         ovtb_class_prompt.pt [1,1053,512]✅
Dataset:               1973 videos / 88098 images       ✅
```

---

## 5. All Bugs Fixed (Complete List)

### `ovtrack/models/mot/ovtrack.py`
- **Triple `self.motion = None`** (was L33-36, now L34): Three redundant assignments collapsed to one clean assignment. Order also fixed: `motion_cfg` is now set before `self.motion = None`.
- **`print(self.method)` removed** (was L40): Was a debug statement from Colab development. Removed.
- **Dead `pickle_name` variable removed** (was L152-153): Variable was computed (`img_name.replace(...)`) but never used anywhere. Removed.

### `ovtrack/datasets/ovtb_dataset.py`
- **`os.path.join` instead of f-string slash** (L152, L158): Was `f"{resfile_path}/ovtb_bbox.json"` — forward slash doesn't work on Windows. Now `os.path.join(resfile_path, "ovtb_bbox.json")`.
- *(Three earlier bugs were fixed before this session: resfile_path propagation bug, tab/space syntax error, USE_PARALLEL=False for Windows multiprocessing.)*

### `configs/ovtrack-teta/ovtb/ovtrack_plus.py`
- **`resfile_path='/scratch/tmp/'` → `None`** (L274): Hardcoded Linux path. `None` uses Python's `tempfile` module instead, which is cross-platform.

### `configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py`
- **`resfile_path='/scratch/tmp/'` → `None`** (L277): Same fix as above.

### `ovtrack/models/trackers/uakf_ovtracker.py`
- **Broad `except TypeError` narrowed** (L160): Was catching all TypeErrors including ones from inside the adaptive update. Now checks `if 'uncertainty' not in str(e): raise` — only catches the specific "unexpected keyword argument 'uncertainty'" TypeError from standard KalmanFilter fallback.

### `setup.py`
- **Windows extension patterns added** (L102): Was `'*/*.so'` (Linux only). Now `['*/*.so', '*/*.pyd', '*/*.dll']`.

### `ovtrack/models/roi_heads/ovtrack_roi_head.py` ← **Architecture Completion**
- **`simple_test_bboxes` now returns `cls_probs`**: After NMS, each surviving detection is matched back to its pre-NMS CLIP class probability row (shape `[N_dets, 1053]`) by finding the proposal whose score for the detected class is closest to `det_bbox[:, -1]`. Returns `(det_bboxes, det_labels, det_cls_probs)`.
- **`simple_test` now uses full uncertainty**: `estimator.estimate(cls_probs=cls_probs_tensor, scores=det_bboxes[:, -1])` — entropy + margin + score all computed. Previously was score-only.

---

## 6. Architecture Data Flow (Complete)

```
simple_test_bboxes()
  → CLIP classification: cls_score_text (N_proposals, 1054) [text + bg]
  → Ensemble with cls_score_image → cls_score (N_proposals, 1054)
  → bbox_head.get_bboxes() → NMS → det_bboxes (N_dets, 5), det_labels (N_dets,)
  → Post-NMS cls_probs matching: for each det, find proposal row by score match
  → Returns: (det_bboxes, det_labels, det_cls_probs[N_dets, 1054])

simple_test() in OVTrackRoIHead
  → Calls simple_test_bboxes → unpacks 3 values
  → If return_uncertainty=True:
      cls_probs_tensor = det_cls_probs[0]          # (N_dets, 1054)
      uncertainties = estimator.estimate(
          cls_probs=cls_probs_tensor,               # → entropy + margin
          scores=det_bboxes[:, -1]                  # → score component
      )                                             # full u = 0.6H + 0.25M + 0.15S
  → Returns 5-tuple: (det_bboxes, det_labels, cem_feats, track_feats, uncertainties)

OVTrack.simple_test() in ovtrack.py
  → Unpacks 5-tuple (handles 3, 4, or 5 element returns)
  → Passes uncertainties in track_kwargs to tracker

UAKFOVTracker.track()
  → Receives uncertainties tensor (N_dets,)
  → Adaptive fusion: dyn_w = motion_weight_base + (motion_weight_max - base) * u
  → match_dists = (1 - dyn_w) * reid_dists + dyn_w * iou_dists
  → Hungarian matching (linear_sum_assignment)
  → Per-matched-track: update_track_ema(track_id, u_val) → smoothed_u
  → update_track(): appearance gating + adaptive Kalman update
      AdaptiveKalmanFilter.update(mean, cov, bbox, uncertainty=smoothed_u)
        → r_factor = compute_r_factor(u) → R_adaptive = R_base * r_factor
        → Update with scaled measurement noise
```

---

## 7. Known Issues (Deliberately Not Fixed)

### 🔴 Double Kalman Update in `OVSortTracker` (Medium Risk)
**File:** `ovtrack/models/trackers/ovsort_tracker.py` L56-70  
**What:** `OVSortTracker.update_track()` calls `super().update_track()` (which runs `kf.update()` in SortTracker), then calls `self.kf.update()` again — two Kalman updates per frame.  
**Impact:** Only affects `ovtrack_plus.py` (OVSortTracker). `UAKFOVTracker` is **not affected** — it overrides `update_track()` completely and does not call super.  
**Why not fixed:** Changing it modifies tracking behavior and needs controlled before/after evaluation.

### 🟡 `with_motion` Property Missing from `OVTrack`
**File:** `ovtrack/models/trackers/sort_tracker.py` L184  
**What:** `SortTracker.track()` checks `model.with_motion` which OVTrack doesn't implement.  
**Impact:** Zero — `UAKFOVTracker` overrides `track()` entirely and never calls `SortTracker.track()`.

### 🟡 `dist_params = dict(backend='nccl')` in configs
**What:** nccl is Linux-only for distributed training.  
**Impact:** Zero for single-GPU eval. Only matters if distributed training is attempted.

### 🟡 Hardcoded `/scratch/tmp/` in other configs
**Fixed:** `ovtrack_plus.py` and `ovtrack_r50_ua.py`  
**Not fixed:** 12+ other configs in `configs/ovtrack-tao/`, `configs/ovtrack-custom/` etc.  
**Impact:** Zero for current OVT-B eval — those configs are not used.

### 🟢 `motions/__init__.py` fragile side-effect chain
**What:** `KalmanFilter` import is commented out in `__init__.py`. It still gets registered because `trackers/__init__.py` imports `adaptive_kalman.py` which imports `KalmanFilter`. If import order changes, registration could break.  
**Why not fixed:** Intentional state. Fixing it requires restructuring imports.

---

## 8. Paper Baseline Numbers (OVT-B)

From the OVTrack CVPR 2023 paper (OVT-B benchmark):

| Method | TETA | AssocA | LocA | DetA |
|--------|------|--------|------|------|
| OVTrack (baseline) | 46.1 | 66.1 | — | — |
| OVTrack+ (with Kalman/Sort) | higher | higher | — | — |
| UA-KF-OVT (ours) | **Not yet run** | **Not yet run** | — | — |

> ⚠️ The baseline OVTrack eval has been attempted locally and shows an AssocA gap vs paper numbers. Root cause not fully isolated. `max_per_img` was previously changed from 50→150 by an earlier session. Checkpoint provenance vs official paper checkpoint not 100% confirmed, though it loads cleanly.

---

## 8.5 AssocA Gap Investigation (Partially Resolved)

Running the baseline `ovtrack_r50.py` locally does NOT exactly reproduce the paper's reported numbers (TETA=46.1, AssocA=66.1). There is an AssocA gap.

**Already ruled out:**
- ❌ Wrong dataset — confirmed correct (1973 videos / 88098 images matches paper)
- ❌ "TAO dataset" string in console — cosmetic class-name string in teta library, not an actual dataset switch
- ❌ Checkpoint shape/key mismatch — loads clean, no missing/unexpected keys
- ❌ Prompt tensor mismatch — correct shape [1, 1053, 512]
- ❌ `max_per_img` mismatch — all three configs now have `max_per_img=150`

**Possible remaining causes (not yet verified):**
- Checkpoint provenance: the downloaded checkpoint may not be the exact same model weights as the paper's final evaluation
- mmcv/mmdet version differences between our setup and the paper's
- Windows vs Linux numerical differences in batch norm / floating point

**Status:** The gap has NOT been fully explained. When comparing UA-KF-OVT vs baseline, compare against YOUR local baseline numbers, not the paper's numbers. The relative improvement is what matters for the research contribution.

---

## 9. How to Run

### Python environment
```
C:\Users\HPCLAB\.conda\envs\ovtrack\python.exe
```

### Run baseline OVTrack (reference, already works)
```powershell
C:\Users\HPCLAB\.conda\envs\ovtrack\python.exe tools/test.py `
  configs/ovtrack-teta/ovtb/ovtrack_r50.py `
  saved_models/pretrained_models/ovtrack_clip_distillation.pth `
  --eval track
```

### Run UA-KF-OVT (first run — ready to go)
```powershell
C:\Users\HPCLAB\.conda\envs\ovtrack\python.exe tools/test.py `
  configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py `
  saved_models/pretrained_models/ovtrack_clip_distillation.pth `
  --eval track
```

### Run OVTrack+ (Kalman baseline)
```powershell
C:\Users\HPCLAB\.conda\envs\ovtrack\python.exe tools/test.py `
  configs/ovtrack-teta/ovtb/ovtrack_plus.py `
  saved_models/pretrained_models/ovtrack_clip_distillation.pth `
  --eval track
```

---

## 10. Current OVT-B Hyperparameters for UA-KF-OVT

These are **OVT-B tuned** (set deliberately, not defaults):

| Parameter | Value | Reason |
|-----------|-------|--------|
| `num_frames_retain` | 60 | Sparse TAO frames — objects disappear for longer |
| `momentum_embed` | 0.35 | Faster appearance adaptation than default 0.2 |
| `motion_weight` (base) | 0.03 | Low — sparse frames make motion prediction weak |
| `motion_weight_max` | 0.30 | Conservative max IoU weight for sparse video |
| `w_entropy` | 0.6 | 1053 classes → entropy is the richest signal |
| `w_margin` | 0.25 | Secondary signal |
| `w_score` | 0.15 | Least informative for OVT-B |
| `ema_gamma` | 0.4 | Slightly less reactive uncertainty smoothing |
| `appearance_gate_min` | 0.05 | Min appearance momentum even at max uncertainty |
| `kalman_beta` | 3.0 | R scaling factor |
| `kalman_max_scale` | 10.0 | Max R scale at max uncertainty |

---

## 11. Future Work / Next Steps

### Immediate (next run)
1. **Run `ovtrack_r50_ua.py`** — first-ever run of the complete architecture. Check for crashes, then compare TETA/AssocA vs baseline `ovtrack_r50.py`.
2. **Ablation study** (all on OVT-B):
   - OVTracker baseline (ovtrack_r50.py)
   - OVSortTracker + Kalman (ovtrack_plus.py)
   - UAKFOVTracker, adaptive weight only (set `enable_adaptive_r=False`)
   - UAKFOVTracker, adaptive Kalman only (set `enable_adaptive_weight=False`)
   - UAKFOVTracker, full (ovtrack_r50_ua.py)
3. **Hyperparameter tuning** — `w_entropy`, `motion_weight_max`, `momentum_embed` all set by intuition. Grid search on OVT-B val could move numbers 2-3 points.

### BDD100K Extension (Not Done Yet)
BDD100K is 30fps dense video — a much better fit for Kalman motion prediction than OVT-B's sparse frames. Expected to show clearer improvement from UA-KF-OVT.

**What's needed to run on BDD100K:**
- BDD100K dataset download and setup (see official BDD100K repo)
- New config file: `configs/ovtrack-teta/bdd100k/ovtrack_r50_ua_bdd.py`
- BDD100K-specific dataset class or adapter
- Different `num_classes` (8 classes: car, truck, bus, person, rider, bicycle, motorcycle, traffic light)
- Different prompt tensor for 8 BDD classes
- Tuned hyperparameters for dense video: `num_frames_retain=30`, `motion_weight_max=0.5`, `w_entropy=0.35` (only 8 classes, entropy less meaningful)

**Reference numbers from literature (BDD100K MOT test set):**
| Method | mMOTA | mIDF1 |
|--------|-------|-------|
| SORT | ~20 | ~35 |
| DeepSORT | ~25 | ~45 |
| QDTrack | 35.5 | 64.3 |
| OC-SORT | ~37 | ~67 |
| StrongSORT | ~38 | ~68 |
| ByteTrack | 40.1 | 69.6 |
| OVTrack (estimated) | ~36 | ~63 |
| UA-KF-OVT target | TBD | TBD |

**Why BDD100K favors UA-KF-OVT:**
- Dense 30fps → Kalman prediction is actually meaningful
- High occlusion in traffic → adaptive R-scaling helps reacquisition
- Multi-class with confusion (car/truck/bus) → CLIP uncertainty signal useful

### High-Impact Code Improvement (Not Yet Implemented)
**ByteTrack-style two-stage matching** — high probability of improving mIDF1 significantly on BDD100K. For OVT-B benefit is marginal.

The idea: 
1. Round 1: Match high-confidence dets (score > 0.5) to confirmed tracks via Hungarian
2. Round 2: Match low-confidence dets (0.1 < score < 0.5) to unmatched tracks from Round 1 using IoU only

This recovers occluded tracks when the detector returns low-confidence boxes. Currently `UAKFOVTracker` discards low-confidence boxes early via distractor removal (L219-225 of `uakf_ovtracker.py`). Would need to save those before removal and use in a second pass.

---

## 12. Architecture Inheritance Chain

```
UAKFOVTracker
  └── OVSortTracker
        └── SortTracker
              └── BaseTracker (uses addict.Dict for tracks — supports both v['key'] and v.key)
                    └── BaseModule (mmcv)
```

`UAKFOVTracker` overrides:
- `__init__()` — adds uncertainty estimator, adaptive params
- `reset()` — also resets uncertainty EMA memory
- `update_track()` — completely custom, avoids OVSortTracker double-update bug
- `track()` — completely custom, full UA-KF pipeline
- `pop_invalid_tracks()` — also prunes uncertainty EMA memory

---

## 13. Registry System

The project uses **local registries** in `ovtrack/models/builder.py` — NOT mmdet global registries.

```python
MODELS   = Registry('model')
TRACKERS = Registry('tracker')
MOTIONS  = Registry('motion')
MOTION   = MOTIONS  # alias
```

`AdaptiveKalmanFilter` is decorated with `@MOTIONS.register_module()` and registered when `adaptive_kalman.py` is imported (which happens via `trackers/__init__.py`).

`UAKFOVTracker` is decorated with `@TRACKERS.register_module()` and registered when `uakf_ovtracker.py` is imported (which happens via `trackers/__init__.py`).

---

## 14. Important: What NOT to Change

- **`motions/__init__.py`** — the commented-out import is intentional. Do not uncomment.
- **`ovtrack_r50.py`** — do not touch this. It is the baseline reference.
- **The `double Kalman update` in `ovsort_tracker.py`** — changing this alters behavior. Run controlled experiments first.
- **`return_uncertainty=True`** in `ovtrack_r50_ua.py` roi_head config — this is what triggers the 5-tuple return from the roi_head. If removed, the tracker gets no uncertainties.
- **The post-NMS cls_probs matching in `simple_test_bboxes`** — this is the key architectural completion. The for-loop matching by score is O(N_dets × N_proposals) which is fine for typical eval sizes. Do not remove.

---

*Last updated by AI agent session. All changes above are confirmed applied and verified via import check.*
