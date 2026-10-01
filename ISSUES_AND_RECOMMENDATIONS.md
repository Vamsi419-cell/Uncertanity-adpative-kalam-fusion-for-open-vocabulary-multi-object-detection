# UA-KF-OVT Codebase Audit: Issues & Recommendations

**Audit Date:** 2026-09-29  
**Scope:** Read-only review of the files listed in the project context.  
**Convention:** Each item is marked **CONFIRMED** (traced in code) or **SUSPECTED** (plausible but requires runtime verification).

---

## Table of Contents

1. [Windows-Compatibility Issues](#1-windows-compatibility-issues)
2. [UA-KF-OVT Tracker Dead Code & Bug Analysis](#2-ua-kf-ovt-tracker-dead-code--bug-analysis)
3. [Exhaustive Config Diff: ovtrack_r50 vs ovtrack_plus vs ovtrack_r50_ua](#3-exhaustive-config-diff)
4. [Motion Module Consistency Audit](#4-motion-module-consistency-audit)
5. [Other Correctness Concerns](#5-other-correctness-concerns)

---

## 1. Windows-Compatibility Issues

Beyond the three already-fixed bugs in `ovtb_dataset.py` (resfile_path propagation, tab/space syntax error, `USE_PARALLEL` flag), the following issues remain.

### 1.1 Hardcoded Linux path `/scratch/tmp/` in 14 config files — **CONFIRMED**

Every config file **except** `ovtrack_r50.py` (which uses `resfile_path=None`) hardcodes a Linux-only absolute path for the evaluation result output directory:

| File | Line |
|------|------|
| `configs/ovtrack-teta/ovtb/ovtrack_plus.py` | 274 |
| `configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py` | 277 |
| `configs/ovtrack-teta/ovtb/ovtrack_bytetrack.py` | 275 |
| `configs/ovtrack-teta/ovtb/ovtrack_ocsort.py` | 277 |
| `configs/ovtrack-teta/ovtb/ovtrack_r50_10k.py` | 278 |
| `configs/ovtrack-teta/ovtb/ovtrack_strongsort.py` | 278 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_plus.py` | 274 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_r50.py` | 280 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_bytetrack.py` | 276 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_ocsort.py` | 278 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_strongsort.py` | 277 |
| `configs/ovtrack-teta/ov_tao_val/ovtrack_r50_no_dynamic_threshold.py` | 281 |
| `configs/ovtrack-custom/ovtrack_r50.py` | 236 |
| `configs/ovtrack-tao/ovtrack_r50.py` | 281 |

**Impact:** On Windows, `os.makedirs('/scratch/tmp/', exist_ok=True)` will attempt to create `C:\scratch\tmp\` (or fail, depending on drive context). If running eval with any config other than `ovtrack_r50.py`, this will either write to an unexpected location or crash.

### 1.2 Hardcoded forward-slash path construction in `ovtb_dataset.py` — **CONFIRMED**

- **Line 152:** `result_files["bbox"] = f"{resfile_path}/ovtb_bbox.json"`
- **Line 158:** `result_files["track"] = f"{resfile_path}/ovtb_track.json"`

These use f-string concatenation with `/` instead of `os.path.join()`. On Windows, when `resfile_path` is a Windows-style path (e.g., `C:\Users\...\tmp`), the result is a mixed-separator path like `C:\Users\...\tmp/ovtb_bbox.json`. Python's `open()` typically handles this, but `mmcv.dump()` or downstream tools may not.

### 1.3 `setup.py` hardcodes Linux `.so` extension — **CONFIRMED**

- **Line 102:** `package_data={'ovtrack.ops': ['*/*.so']}`

On Windows, compiled C extensions use `.pyd` or `.dll`, not `.so`. This means `pip install -e .` will not package any compiled ops on Windows. **Impact:** Only affects training or ops that require compiled CUDA kernels. For inference-only evaluation this is unlikely to be triggered, but it will silently omit any compiled operators.

### 1.4 `dist_params = dict(backend='nccl')` in all configs — **CONFIRMED**

All 15 config files hardcode `backend='nccl'` for distributed training. NCCL is Linux-only. On Windows, distributed training requires `backend='gloo'`. **Impact:** Only affects multi-GPU distributed training, not single-GPU inference/eval. But running `tools/dist_train.sh` or equivalent on Windows would fail immediately.

### 1.5 `pickle_name` path separator assumption in `ovtrack.py` — **CONFIRMED**

- **Line 153:** `pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")`

This assumes image filenames contain forward slashes. On Windows, `img_metas[0]["filename"]` may contain backslashes. The replacement would leave backslashes intact, producing filenames containing `\` characters. **Impact:** This variable is currently dead code (never used after assignment — see §5.1), so the bug has no runtime effect. But if the `save_dets` code path were ever re-enabled, it would produce incorrect pickle filenames on Windows.

### 1.6 `checkpoint_config = dict(interval=1, create_symlink=False)` — **CONFIRMED (benign)**

All configs set `create_symlink=False`, which is correct for Windows. This is noted as a **non-issue** — the original codebase correctly disabled symlink creation.

### 1.7 No `os.fork()`, `os.symlink()`, `subprocess` calls found — **CONFIRMED**

A repository-wide search for `os.fork`, `os.symlink`, `os.link`, `os.chmod`, `os.chown`, `subprocess`, `Popen`, and `os.system` across all `.py` files returned no matches. No `multiprocessing` `start_method` or `fork`/`forkserver` references were found outside the already-fixed `USE_PARALLEL` flag.

---

## 2. UA-KF-OVT Tracker Dead Code & Bug Analysis

### 2.1 Are `compute_uncertainty()`, `adaptive_r_scale()`, and `adaptive_fusion_weight()` called? — **CONFIRMED: These exact function names do not exist**

The user's question references three function names that do not appear in the codebase. The equivalent logic is implemented under different names:

| Conceptual Function | Actual Implementation | File | Line(s) |
|---|---|---|---|
| `compute_uncertainty()` | `UncertaintyEstimator.estimate()` | `uncertainty_estimator.py` | 100–143 |
| `adaptive_r_scale()` | `AdaptiveKalmanFilter.compute_r_factor()` | `adaptive_kalman.py` | 50–78 |
| `adaptive_fusion_weight()` | Inline computation in `UAKFOVTracker.track()` | `uakf_ovtracker.py` | 258–263 |

### 2.2 Is the UA-KF-OVT logic dead code? — **CONFIRMED: NOT dead code, but never actually executed**

The code is fully wired internally — `UAKFOVTracker.track()` does call all three pieces of logic. However, **no config file has ever been run with this tracker**, so these code paths have never been exercised at runtime. The full call chain is:

1. **Uncertainty estimation** — `UAKFOVTracker.track()` lines 199–208 call `self.uncertainty_estimator.estimate(...)`.
2. **Adaptive fusion weight** — lines 258–263 compute `dyn_w` using the per-detection uncertainty and blend IoU + ReID costs.
3. **Adaptive Kalman R-scaling** — after matching, `self._matched_uncertainties` is populated (lines 283–285). The overridden `update_track()` (line 134) retrieves `u` from `self._matched_uncertainties` and passes it to `self.kf.update(..., uncertainty=u)` (line 159), which flows into `AdaptiveKalmanFilter.project()` → `compute_r_factor()`.

**Verdict:** The code is internally connected and not dead code in the traditional sense. But it has never been tested at runtime. It requires running `ovtrack_r50_ua.py`, which has never been done.

### 2.3 Bug: Dangerous `except TypeError` silently masks errors — **CONFIRMED**

**File:** `uakf_ovtracker.py`, lines 157–163

```python
try:
    self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
        self.tracks[id].mean, self.tracks[id].covariance, bbox, uncertainty=u)
except TypeError:
    # Standard KalmanFilter fallback
    self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
        self.tracks[id].mean, self.tracks[id].covariance, bbox)
```

The intent is to gracefully fall back if `self.kf` is a base `KalmanFilter` (whose `update()` doesn't accept `uncertainty=`). However, `except TypeError` will also silently catch:
- Shape mismatches inside NumPy/SciPy operations
- `None` values passed where arrays are expected
- Any other `TypeError` inside `AdaptiveKalmanFilter.update()` or `project()`

This is a **correctness hazard**: if the adaptive Kalman update has a runtime bug, it will silently fall back to the non-adaptive path with no warning, making debugging extremely difficult.

### 2.4 Bug: `roi_head` uncertainty computation is score-only, ignores class probabilities — **CONFIRMED**

**File:** `ovtrack_roi_head.py`, lines 405–414

When `return_uncertainty=True` (as configured in `ovtrack_r50_ua.py` line 42), the ROI head computes uncertainty using **only detection scores**:

```python
uncertainties = estimator.estimate(scores=det_bboxes[:, -1], num_dets=det_bboxes.size(0))
```

It does **not** pass `cls_probs` (the full class probability distribution). This means:
- `compute_entropy()` is never called from the ROI head path
- `compute_margin()` is never called from the ROI head path
- The uncertainty degrades to `u = 1 - score`, a simple scalar inversion

The `UncertaintyEstimator` is designed for three-component uncertainty (entropy + margin + score), but only the score component is ever supplied. The `UAKFOVTracker.track()` has a fallback path (lines 205–208) that also only uses score-based uncertainty when `cls_probs` is not supplied. Since `cls_probs` is never passed from `ovtrack.py` `simple_test()` (line 183–194, `track_kwargs` does not include `cls_probs`), **entropy and margin uncertainty are always zero in practice**.

### 2.5 Bug: Duplicate Kalman `update()` call in `UAKFOVTracker` — **CONFIRMED**

**File:** `uakf_ovtracker.py`, line 134 (`update_track`) and parent class `sort_tracker.py`, line 101–111

`UAKFOVTracker.update_track()` overrides `OVSortTracker.update_track()`, which itself calls `super().update_track(id, obj)` — i.e., `SortTracker.update_track()`. Looking at the chain:

- `SortTracker.update_track()` (line 101–111): calls `super().update_track(id, obj)` (BaseTracker), then does `self.kf.update(...)` on lines 110–111.
- `OVSortTracker.update_track()` (line 56–70): calls `super().update_track(id, obj)` (→ SortTracker, which already does a Kalman update), then does **another** `self.kf.update(...)` on lines 69–70.

This means `OVSortTracker` already performs the Kalman update **twice** per frame per track. `UAKFOVTracker.update_track()` avoids calling `super().update_track()` and instead manually iterates over `self.memo_items` (lines 139–148) and then does its own Kalman update (lines 155–166). The comment on line 138 ("Custom buffer update to avoid OVSortTracker double-update bug") shows the author was aware of this issue in `OVSortTracker`.

**However**, this means that the **baseline `OVSortTracker` (used by `ovtrack_plus.py`) has a double Kalman update bug** — each track's state is Kalman-updated twice per frame, once in `SortTracker.update_track()` and once in `OVSortTracker.update_track()`. This would corrupt the Kalman filter state (double-counting the measurement) and degrade motion prediction quality.

### 2.6 Observation: `UAKFOVTracker` self-instantiates `AdaptiveKalmanFilter` even when config provides one — **CONFIRMED**

**File:** `uakf_ovtracker.py`, lines 181–192

```python
if not hasattr(self, 'kf'):
    if isinstance(model.motion, AdaptiveKalmanFilter):
        self.kf = model.motion
    elif self.enable_adaptive_r:
        self.kf = AdaptiveKalmanFilter(
            mapping=self.kalman_mapping,
            beta=self.kalman_beta,
            max_scale=self.kalman_max_scale,
        )
    else:
        self.kf = model.motion
```

When `model.motion` is a `KalmanFilter` (not `AdaptiveKalmanFilter`) and `enable_adaptive_r=True`, the tracker creates its own `AdaptiveKalmanFilter` with default `center_only=False`. This is intentional fallback behavior, but it means the `center_only` parameter from any configured motion model is silently discarded.

When `model.motion` is `None` (as in the baseline `ovtrack_r50.py` config where `motion` is absent and `self.motion = None`), and the tracker is `UAKFOVTracker`, `isinstance(None, AdaptiveKalmanFilter)` is `False`, so it falls through to self-instantiation. This path works correctly.

---

## 3. Exhaustive Config Diff

Every parameter that differs across the three config files is listed below. Parameters not listed are **identical** across all three files (backbone, neck, rpn_head, bbox_head, track_head, train_cfg, test_cfg rpn/rcnn, test_pipeline, img_norm_cfg, optimizer, lr_config, total_epochs, load_from).

### 3.1 Model-Level Differences

| Parameter | `ovtrack_r50.py` | `ovtrack_plus.py` | `ovtrack_r50_ua.py` |
|---|---|---|---|
| `model.motion` | *absent* (→ `None`) | `dict(type='KalmanFilter')` (L138) | `dict(type='AdaptiveKalmanFilter', mapping='linear', beta=3.0, min_scale=1.0, max_scale=10.0)` (L139–145) |
| `model.roi_head.return_uncertainty` | *absent* (→ `False`) | *absent* (→ `False`) | `True` (L42) |
| `model.tracker.type` | `'OVTracker'` (L139) | `'OVSortTracker'` (L140) | `'UAKFOVTracker'` (L147) |
| `model.tracker.init_score_thr` | `0.0001` (L140) | *absent* | *absent* |
| `model.tracker.obj_score_thr` | `0.0001` (L141) | *absent* (default `0.0001` in `OVSortTracker.__init__`) | *absent* (default `0.0001` in `UAKFOVTracker.__init__`) |
| `model.tracker.match_score_thr` | `0.5` (L142) | `0.5` (L141) | `0.5` (L148) |
| `model.tracker.memo_frames` | `10` (L143) | *absent* (not applicable — OVSortTracker uses `num_frames_retain`) | *absent* |
| `model.tracker.num_frames_retain` | *absent* (not applicable — OVTracker uses `memo_frames`) | `30` (L142) | `30` (L149) |
| `model.tracker.momentum_embed` | `0.8` (L144) | `0.2` (L143) | `0.2` (L150) |
| `model.tracker.momentum_obj_score` | `0.5` (L145) | *absent* | *absent* |
| `model.tracker.match_metric` | `'bisoftmax'` (L146) | *absent* | *absent* |
| `model.tracker.match_with_cosine` | `True` (L147) | *absent* | *absent* |
| `model.tracker.contrastive_thr` | `0.5` (L148) | *absent* | *absent* |
| `model.tracker.motion_weight` | *absent* | `0.03` (L144) | `0.03` (L151) |
| `model.tracker.motion_weight_max` | *absent* | *absent* | `0.35` (L152) |
| `model.tracker.enable_adaptive_r` | *absent* | *absent* | `True` (L153) |
| `model.tracker.enable_adaptive_weight` | *absent* | *absent* | `True` (L154) |
| `model.tracker.enable_appearance_gating` | *absent* | *absent* | `True` (L155) |
| `model.tracker.kalman_mapping` | *absent* | *absent* | `'linear'` (L156) |
| `model.tracker.kalman_beta` | *absent* | *absent* | `3.0` (L157) |
| `model.tracker.kalman_max_scale` | *absent* | *absent* | `10.0` (L158) |
| `model.tracker.w_entropy` | *absent* | *absent* | `0.4` (L159) |
| `model.tracker.w_margin` | *absent* | *absent* | `0.3` (L160) |
| `model.tracker.w_score` | *absent* | *absent* | `0.3` (L161) |
| `model.tracker.ema_gamma` | *absent* | *absent* | `0.5` (L162) |
| `model.tracker.appearance_gate_min` | *absent* | *absent* | `0.05` (L163) |

### 3.2 `test_cfg` Structural Difference

| Aspect | `ovtrack_r50.py` | `ovtrack_plus.py` | `ovtrack_r50_ua.py` |
|---|---|---|---|
| Placement | `test_cfg` is a **top-level key inside the `model` dict** but at the same indentation level as `tracker` (L150–162). It is syntactically valid but placed oddly — it appears as a sibling to `tracker` inside the `model` dict. | Same structure (L146–158) | `test_cfg` is a **proper nested key** inside `model` using `test_cfg=dict(...)` syntax (L165–175) |

> [!NOTE]
> All three files produce identical final `test_cfg` values (`rcnn.max_per_img=150`, `rcnn.score_thr=0.0001`, etc.). The syntactic difference in placement does not change the parsed config because mmcv's Config parser treats them equivalently.

### 3.3 Non-Model Differences

| Parameter | `ovtrack_r50.py` | `ovtrack_plus.py` | `ovtrack_r50_ua.py` |
|---|---|---|---|
| `evaluation.resfile_path` | `None` (L278) | `'/scratch/tmp/'` (L274) | `'/scratch/tmp/'` (L277) |
| `gpu_ids` | `range(0, 8)` (L286) | `range(0, 2)` (L282) | `range(0, 2)` (L285) |
| `data.train.dataset.dataset.ann_file` | `'data/tao/annotations/ovtrack/...'` (L236) | `'data/tao/annotations/diff_track/...'` (L232) | `'data/tao/annotations/diff_track/...'` (L236) |
| `data.train.dataset.dataset.pipeline[0].file_client_args.img_db_path` | `'data/tao/ovtrack/...'` (L243) | `'data/tao/diff_track/...'` (L239) | `'data/tao/diff_track/...'` (L242) |
| `data.val.classes` | `'data/lvis/annotations/lvis_classes_v1.txt'` (L252) | `'data/tao/annotations/lvis_classes_v1.txt'` (L248) | `'data/tao/annotations/lvis_classes_v1.txt'` (L252) |

> [!IMPORTANT]
> The `momentum_embed` difference (`0.8` in r50 vs `0.2` in plus/ua) is a significant behavioral difference beyond just tracker type. A momentum of `0.8` means 80% of the new embedding is mixed in per frame (aggressive update), while `0.2` means only 20% (conservative update). This could meaningfully affect association accuracy independently of motion modeling.

### 3.4 Previously Suspected `max_per_img` Difference — **CONFIRMED: Not a difference**

The user suspected that `rcnn.max_per_img` differs between configs (50 vs 150). **All three files set `rcnn.max_per_img=150`** in `test_cfg`. There is no 50-vs-150 discrepancy across these three configs. If a value of 50 was observed during a run, it may have come from a different config file, a command-line override, or an mmdet default.

---

## 4. Motion Module Consistency Audit

### 4.1 Triple `self.motion = None` assignment — **CONFIRMED: Messy but functionally harmless**

**File:** `ovtrack.py`, lines 33–38

```python
self.tracker_cfg = tracker
self.motion = None          # line 33
self.motion = None          # line 34
self.motion_cfg = motion    # line 35
self.motion = None          # line 36
if self.motion_cfg is not None:   # line 37
    self.init_motion()             # line 38
```

This is clearly a leftover from hasty iterative patching. The triple assignment is redundant — only the final `self.motion = None` on line 36 matters, and it is immediately overwritten by `self.init_motion()` if `motion_cfg` is not None.

**Functional impact:** None for the baseline config (`ovtrack_r50.py`) where `motion` is absent (defaults to `None` in the constructor signature on line 21). For `ovtrack_plus.py` and `ovtrack_r50_ua.py` where `motion` is configured, `init_motion()` is called on line 38 and overwrites `self.motion` with a built `KalmanFilter` or `AdaptiveKalmanFilter`.

### 4.2 `motions/__init__.py` commented-out import — **CONFIRMED: Consistent with baseline but creates a side-effect dependency**

**File:** `ovtrack/models/motions/__init__.py`, lines 1–4

```python
# Motion models are disabled for the current OVTracker baseline evaluation.
# from .kalman_filter import KalmanFilter
__all__ = []
```

Because `KalmanFilter` is not imported here, importing `ovtrack.models.motions` does **not** register `KalmanFilter` in the `MOTIONS` registry.

**However**, `KalmanFilter` is still registered in practice because:
1. `ovtrack/models/trackers/adaptive_kalman.py` line 3 does `from ..motions.kalman_filter import KalmanFilter` (a direct module import bypassing `__init__.py`)
2. `ovtrack/models/trackers/__init__.py` line 37 imports `adaptive_kalman.py` unconditionally (no try/except)
3. `ovtrack/models/__init__.py` line 6 does `from .trackers import *`

So `KalmanFilter` gets registered as a **side effect** of importing the trackers module. This works but is fragile — if `adaptive_kalman.py` were ever wrapped in a `try/except ImportError` like the other trackers in `trackers/__init__.py`, `KalmanFilter` would silently stop being registered, and `ovtrack_plus.py` would crash with a registry key error.

### 4.3 `init_motion()` re-called at frame 0 — **CONFIRMED: Redundant but harmless**

**File:** `ovtrack.py`, lines 181–182

```python
if getattr(self, "motion", None) is not None and frame_id == 0:
    self.init_motion()
```

If `motion` was already built in `__init__` (line 37–38), this rebuilds it again at frame 0. Since `KalmanFilter` is stateless (all per-track state lives in `track.mean`/`track.covariance`), re-instantiation is redundant but harmless. For `AdaptiveKalmanFilter`, the same applies — it only holds configuration parameters, not per-track state.

### 4.4 Overall Consistency Verdict

The `self.motion = None` patch and the `motions/__init__.py` commented-out import are **internally consistent** for the baseline evaluation scenario:
- Baseline (`ovtrack_r50.py`): `motion` config absent → `self.motion` stays `None` → `OVTracker` never touches `model.motion` → ✅ works
- Plus (`ovtrack_plus.py`): `motion=dict(type='KalmanFilter')` → `init_motion()` builds it → `OVSortTracker` uses `model.motion` via `self.kf = model.motion` → ✅ works (but has the double-update bug from §2.5)
- UA (`ovtrack_r50_ua.py`): `motion=dict(type='AdaptiveKalmanFilter', ...)` → `init_motion()` builds it → `UAKFOVTracker` checks `isinstance(model.motion, AdaptiveKalmanFilter)` → uses it → ✅ would work (never tested)

There is **no half-applied state** that would cause a crash for any of the three configs. The messiness is cosmetic (triple assignment, commented import) not functional.

---

## 5. Other Correctness Concerns

### 5.1 Dead code: `pickle_name` variable computed but never used — **CONFIRMED**

**File:** `ovtrack.py`, lines 151–153

```python
img_name = img_metas[0]["filename"]
if img_name is not None:
    pickle_name = img_name.replace("/", "-").replace(".jpg", ".pth")
```

`pickle_name` is assigned on every call to `simple_test()` but is never read, stored, or returned. This appears to be a leftover from the `save_dets` / `use_fixed_dets` feature that was partially implemented (the constructor accepts these flags on lines 42–49) but no saving logic exists in `simple_test()`.

### 5.2 Greedy matching in `OVTracker` vs Hungarian matching in `OVSortTracker` — **CONFIRMED**

**File:** `ovtracker.py`, lines 234–242 vs `ovsort_tracker.py`, line 122

`OVTracker.track()` uses **greedy** matching — iterating over detections in order and greedily assigning the best-scoring memo track:

```python
for i in range(num_objs):
    if bboxes[i, -1] < self.obj_score_thr:
        continue
    conf, memo_ind = torch.max(scores[i, :], dim=0)
    if conf > self.match_score_thr:
        ids[i] = memo_ids[memo_ind]
        scores[:i, memo_ind] = 0
        scores[i + 1:, memo_ind] = 0
```

`OVSortTracker.track()` and `UAKFOVTracker.track()` use **Hungarian** (optimal) matching via `linear_sum_assignment()`.

This is a fundamental algorithmic difference that affects association quality. The greedy approach is sensitive to detection ordering (higher-scored detections get priority), which could partially explain AssocA differences between configs.

> [!WARNING]
> **Regarding the AssocA gap:** This audit cannot determine whether the greedy-vs-Hungarian difference, the `momentum_embed` difference (0.8 vs 0.2), or some other factor is the primary cause of the AssocA gap relative to the paper's numbers. This requires **controlled single-variable experiments**: run the baseline `ovtrack_r50.py` config, then change only one parameter at a time and re-evaluate. The config audit in §3 identifies all variables that differ and need to be isolated.

### 5.3 Double Kalman update in `OVSortTracker` — **CONFIRMED** (repeated from §2.5 for emphasis)

**File:** `ovsort_tracker.py`, lines 56–70

`OVSortTracker.update_track()` calls `super().update_track(id, obj)` (line 57), which goes to `SortTracker.update_track()`, which itself calls `self.kf.update()` (line 110–111). Then `OVSortTracker.update_track()` calls `self.kf.update()` **again** (lines 69–70).

This means every matched track gets its Kalman state updated with the same measurement twice per frame. This effectively double-counts the observation, causing:
- Overconfident covariance (shrinks too fast)
- State that snaps too aggressively to measurements
- Degraded prediction quality for occluded/missed frames

**This is a real bug in the OVTrack+ baseline (`ovtrack_plus.py`).** It does not affect `ovtrack_r50.py` (which uses `OVTracker`, no Kalman filter) or `ovtrack_r50_ua.py` (where `UAKFOVTracker.update_track()` deliberately avoids calling `super().update_track()`).

### 5.4 `SortTracker.track()` references `model.with_motion` which `OVTrack` doesn't implement — **CONFIRMED**

**File:** `sort_tracker.py`, line 184; `ovtrack.py` — no `with_motion` property

```python
if model.with_motion:
    self.tracks, costs = model.motion.track(self.tracks, bbox_xyxy_to_cxcyah(bboxes))
```

`OVTrack` inherits from `TwoStageDetector` and does not define a `with_motion` property. If `SortTracker.track()` is ever called directly (not via the `OVSortTracker` or `UAKFOVTracker` overrides, which have their own `track()` methods), this would crash with `AttributeError: 'OVTrack' object has no attribute 'with_motion'`.

**Impact:** Currently no config uses bare `SortTracker` with `OVTrack`, so this is unreachable. But it means the `SortTracker` base class cannot be used directly without adding `with_motion` to `OVTrack`.

### 5.5 `OVTracker.track()` receives but ignores `model` kwarg — **CONFIRMED**

**File:** `ovtracker.py`, line 154–163; `ovtrack.py`, line 183–194

`OVTrack.simple_test()` passes `model=self` in `track_kwargs` (line 184). `OVTracker.track()` receives this via `**kwargs` (line 163) and silently discards it. This is correct behavior (OVTracker doesn't need the model reference), but it means `OVTracker` cannot access `model.motion` even if motion were configured — reinforcing that motion is truly inert for the baseline config.

### 5.6 `print(self.method)` debug statement left in constructor — **CONFIRMED**

**File:** `ovtrack.py`, line 40

```python
self.method = method
print(self.method)
```

This unconditionally prints the method string (e.g., `ovtrack-teta`) to stdout on every model initialization. Minor cosmetic issue.

### 5.7 `AdaptiveKalmanFilter` file placement inconsistency — **CONFIRMED**

`AdaptiveKalmanFilter` is located in `ovtrack/models/trackers/adaptive_kalman.py` but registers with the `MOTION` registry (line 7: `@MOTION.register_module()`). By convention, motion models should live in `ovtrack/models/motions/`. This doesn't cause functional issues but is architecturally inconsistent and could confuse future maintainers.

### 5.8 Suggestion: Verify `return_uncertainty` actually passes class probabilities — **SUSPECTED (requires runtime verification)**

The `OVTrackRoIHead.simple_test()` (line 411) computes uncertainty using only detection scores. However, the RoI head has access to the full CLIP-based class logits/probabilities that were used to produce `det_labels`. If entropy and margin uncertainty are desired (as the `UncertaintyEstimator` is designed for), the class probability tensor should be extracted from the classification step and passed to `estimator.estimate(cls_probs=..., scores=...)`. This would require examining the full `simple_test` flow of the RoI head, which is beyond the scope of a read-only diff audit — it requires tracing the actual tensor flow at runtime.

---

## Summary of Findings by Severity

### Must-Fix Before Running UA-KF-OVT

| # | Issue | Section |
|---|---|---|
| 1 | `resfile_path='/scratch/tmp/'` in `ovtrack_r50_ua.py` — invalid on Windows | §1.1 |
| 2 | `except TypeError` silently masks Kalman update bugs | §2.3 |
| 3 | Uncertainty is score-only; entropy/margin components are dead weight | §2.4 |

### Should Investigate for Baseline AssocA Gap

| # | Issue | Section |
|---|---|---|
| 4 | `momentum_embed` 0.8 (r50) vs 0.2 (plus/ua) — significant behavioral difference | §3.1 |
| 5 | Greedy (OVTracker) vs Hungarian (OVSortTracker) matching | §5.2 |
| 6 | Double Kalman update bug in OVSortTracker (affects ovtrack_plus.py) | §5.3 |

### Cleanup / Low Priority

| # | Issue | Section |
|---|---|---|
| 7 | Triple `self.motion = None` — cosmetic mess | §4.1 |
| 8 | `motions/__init__.py` commented import — fragile side-effect dependency | §4.2 |
| 9 | Dead `pickle_name` variable | §5.1 |
| 10 | `print(self.method)` debug output | §5.6 |
| 11 | `AdaptiveKalmanFilter` file placement inconsistency | §5.7 |
| 12 | `setup.py` `.so` extension hardcoded | §1.3 |
| 13 | `dist_params` hardcodes `nccl` | §1.4 |
