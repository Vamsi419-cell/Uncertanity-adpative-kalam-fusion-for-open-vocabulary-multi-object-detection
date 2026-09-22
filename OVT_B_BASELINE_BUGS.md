# 🛠️ Official OVT-B/OVTrack Repository Bug Fixes

This document details the bugs discovered in the official `OVTrack` and `OVT-B-Dataset` codebase during our evaluation. These fixes restore the baseline performance reported in the paper and fix critical crashes. 

This list is intended to serve as a reference for a Pull Request to the official repository.

---

### 1. The `max_per_img = 50` Config Bug (Massive `AssocA` Drop)
- **File:** `configs/ovtrack-teta/ovtb/ovtrack_r50.py`
- **Issue:** The original config artificially capped the `test_cfg.rcnn.max_per_img` at `50`. Since OVT-B dense frames have up to 86 objects per frame, the detector was dropping up to 36 valid objects every frame. This breaks active tracklets and decimates Association Accuracy (`AssocA`), dropping it by nearly 28 points compared to the paper's reported metrics.
- **Fix:** Changed `max_per_img` to `150` (matching the `ovtrack_plus.py` config) to ensure no valid targets are dropped during evaluation. We also adjusted the `optimizer` learning rate from `0.02` to `0.005` to match `ovtrack_plus.py`.

### 2. The `MOTION` vs `MOTIONS` Registry Crash
- **File:** `ovtrack/models/builder.py`
- **Issue:** The registry was created as `MOTION = Registry('motion')`, but `kalman_filter.py` and other modules attempted to use `@MOTIONS.register_module()`. This typo caused instant crashes (`KeyError` / `NameError`) when trying to pass motion modules through configs.
- **Fix:** Aliased the registries in `builder.py` (`MOTIONS = MOTION`).

### 3. The Frame-by-Frame Kalman Memory Wipe
- **File:** `ovtrack/models/mot/ovtrack.py` (inside `simple_test`)
- **Issue:** The tracker was re-instantiating the motion model (`self.motion = build_motion(tracker['motion'])`) on *every single frame*. This continuously wiped the Kalman filter's state matrix, covariance, and velocity history 30 times a second, completely destroying its predictive ability.
- **Fix:** Wrapped the instantiation in a basic initialization check:
  ```python
  if tracker.get('motion', None) and getattr(self, "motion", None) is None:
      self.motion = build_motion(tracker['motion'])
  ```

### 4. Floating Point Softmax Overflow (`NaN` / `Inf` Crash)
- **Files:** `ovtrack/models/trackers/ovtracker.py` and `ovtrack/models/trackers/ovsort_tracker.py`
- **Issue:** When computing appearance similarity for Hungarian matching, the trackers blindly exponentiate dot-products using `torch.exp(sims)`. In standard 32-bit floating point, `exp(x)` overflows to `inf` if $x > 88.7$, resulting in `NaN` values. Passing a `NaN` matrix to `scipy.optimize.linear_sum_assignment` instantly crashes the pipeline with `ValueError: matrix contains invalid numeric entries`.
- **Fix:** Added strict numerical clamping before exponentiation in all trackers:
  ```python
  sims_clamped = torch.clamp(sims, min=-30.0, max=30.0)
  sims_clamped = torch.nan_to_num(sims_clamped, nan=0.0)
  exps = torch.exp(sims_clamped)
  ```

### 5. Tracker Ignoring its Own Kalman Filter
- **File:** `ovtrack/models/trackers/ovsort_tracker.py`
- **Issue:** During tracking prediction, the tracker hardcoded a call to `model.motion.predict()`. If a custom Kalman filter was assigned specifically to the tracker, it was ignored.
- **Fix:** Updated the tracker to prefer its own internal `self.kf` if it exists.

### 6. C++ Extension Import Failures (`mmdet`, `mmcv`) on CPU Nodes
- **Files:** `ovtrack/models/trackers/ovsort_tracker.py` and `ovtrack/models/trackers/sort_tracker.py`
- **Issue:** When running lightweight evaluations, the tracker strictly imported `bbox_overlaps` from `mmdet.core` and `linear_sum_assignment` from `motmetrics.lap`. If the `mmdet` C++ binaries were missing, it crashed completely.
- **Fix:** Added graceful Python-native `try/except` fallbacks. For example, if `mmdet` is missing, it falls back to a vectorized PyTorch implementation of `bbox_overlaps`, and if `motmetrics` is missing, it falls back to `scipy.optimize`.
