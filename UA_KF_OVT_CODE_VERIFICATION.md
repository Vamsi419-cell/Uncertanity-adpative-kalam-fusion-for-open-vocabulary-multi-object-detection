# UA-KF-OVT Code Verification Report

**Audit scope:** repository source and configuration as present in the workspace on 2026-10-12. This report does not modify the manuscript or source code. Claims about completed benchmark experiments require an evaluation log or result artifact; none is present in this workspace.

## Status convention

- **VERIFIED:** directly established from executable source or configuration.
- **NOT VERIFIED:** the repository does not contain enough executable evidence to establish the claim.
- **CONFLICT:** repository prose or supplied manuscript claims disagree with executable source/configuration.

## 1. Baseline and Data Flow

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Detector architecture | **VERIFIED** | [`configs/ovtrack-teta/ovtb/ovtrack_r50.py`](configs/ovtrack-teta/ovtb/ovtrack_r50.py#L1-L65), `model`; [`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L20-L43), `OVTrack.__init__` | ResNet-50, FPN, RPN, and `OVTrackRoIHead`; `freeze_detector=True`. | State the detector as frozen ResNet-50/FPN/RPN OVTrack. |
| Checkpoint and prompt | **VERIFIED** | [`configs/ovtrack-teta/ovtb/ovtrack_r50.py`](configs/ovtrack-teta/ovtb/ovtrack_r50.py#L42-L54), `model.roi_head`; [`configs/ovtrack-teta/ovtb/ovtrack_r50.py`](configs/ovtrack-teta/ovtb/ovtrack_r50.py#L277-L280), `load_from` | Prompt: `saved_models/pretrained_models/ovtb_class_prompt.pt`; checkpoint: `saved_models/pretrained_models/ovtrack_clip_distillation.pth`. The files themselves are not present in this workspace, so their binary contents are not independently verified here. | Use these paths only as configured paths, not as proof that a particular binary was used in a completed run. |
| OVTrack baseline | **VERIFIED** | [`configs/ovtrack-teta/ovtb/ovtrack_r50.py`](configs/ovtrack-teta/ovtb/ovtrack_r50.py#L135-L149), `model.tracker` | `OVTracker`, greedy appearance tracker, no configured motion model. | This is the repository baseline configuration, not OVTrack+. |
| OVTrack+ baseline | **VERIFIED** | [`configs/ovtrack-teta/ovtb/ovtrack_plus.py`](configs/ovtrack-teta/ovtb/ovtrack_plus.py#L135-L155), `model.motion` and `model.tracker` | `OVSortTracker` with `KalmanFilter`, `motion_weight=0.03`, `match_score_thr=0.5`, `num_frames_retain=30`, and `momentum_embed=0.2`. | Report this as the local OVTrack+ implementation. Published values cannot be claimed as reproduced without results. |
| UA-KF-OVT experiment configuration | **VERIFIED** | [`configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L135-L172), `model.motion` and `model.tracker` | `UAKFOVTracker` with `AdaptiveKalmanFilter`, linear mapping, `beta=3`, `max_scale=10`, base motion weight `0.03`, maximum `0.30`, and uncertainty weights `0.60/0.25/0.15`. | These are the active experiment values. They override the estimator defaults. |
| OVT-B test data | **VERIFIED for configuration; NOT VERIFIED for data contents** | [`configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L263-L269), `data.test` | Test config points to `data/ovtb/ovtb_classes.txt` and `data/ovtb/ovtb_ann.json`. The annotation file is absent from the workspace. | Do not state the annotation category count as independently verified. |
| Fine-tuning during the documented test command | **VERIFIED** | [`tools/test.py`](tools/test.py#L93-L121), `main`; [`configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L1-L4), `freeze_detector` | `tools/test.py` builds the test dataset/model and loads the supplied checkpoint; it does not train. The config's training data/workflow does not execute under this command. | OVT-B is evaluation-only for the stated `tools/test.py ... --eval track` procedure. Do not generalize this to every possible training invocation. |

### One-frame data flow

1. `OVTrack.simple_test()` extracts backbone/neck features and generates RPN proposals ([`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L145-L158)).
2. `OVTrackRoIHead.simple_test()` calls `simple_test_bboxes()` and obtains `det_bboxes`, `det_labels`, and `det_cls_probs` ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L384-L386)).
3. `simple_test_bboxes()` creates normalized region embeddings, computes text/image CLIP score distributions, ensembles them, and applies `bbox_head.get_bboxes()` ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L470-L553)).
4. After NMS, it heuristically associates each surviving detection with the pre-NMS row whose score for that detected label is closest to the retained score, producing `det_cls_probs` ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L556-L603)).
5. The ROI head computes uncertainty when `return_uncertainty=True` and returns a five-tuple ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L405-L420)).
6. `OVTrack.simple_test()` forwards boxes, labels, track embeddings, class embeddings, frame ID, and uncertainty to `self.tracker.track()` ([`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L161-L192)).
7. `UAKFOVTracker.track()` filters distractors, predicts existing boxes, computes appearance/motion association, assigns IDs with Hungarian matching, and updates track state ([`ovtrack/models/trackers/uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L175-L330)).

### Per-detection fields

| Field | Status | Evidence and qualification |
|---|---|---|
| Bounding box | **VERIFIED** | `det_bboxes` is the detector result from `get_bboxes()` and has coordinates plus a final score ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L528-L553)). |
| Detection confidence | **VERIFIED** | `det_bboxes[:, -1]` is used as the detection score for uncertainty and tracker state ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L413-L415); [`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L180-L185)). |
| Class label | **VERIFIED** | `det_labels` is returned by `get_bboxes()` and forwarded to the tracker ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L386-L390)). |
| Class-probability vector | **VERIFIED with qualification** | `det_cls_probs` contains the matched pre-NMS `cls_score` row ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L556-L603)). The row is probability-like, but the geometric ensemble is not explicitly renormalized after multiplication; entropy renormalizes it before entropy computation. |
| Appearance/ReID embedding | **VERIFIED** | `_track_forward()` returns the track head output ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L350-L360)); `OVTrack.simple_test()` passes it as `embeds` ([`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L178-L185)). The configured `QuasiDenseEmbedHead` has `embed_channels=256` ([`configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L64-L75)). |
| Separate class-exemplar embedding | **NOT VERIFIED / NOT PRESENT in active config** | `cem_head` is absent in the three OVT-B configs; the main model copies `track_feats` into `cem_feats` when `cem_feats is None` ([`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L163-L169)). Thus `cls_embeds` is not a distinct active representation. |

### Vocabulary dimensions and mapping

| Item | Status | Evidence |
|---|---|---|
| Detector class count | **VERIFIED** | `bbox_head.num_classes=1053` in [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L49-L58). The prompt is sliced to this count in `OVTrackRoIHead.__init__` ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L80-L116)). |
| Background in uncertainty vector | **VERIFIED** | Unless `fix_bg` is enabled, `bg_class_embedding` is concatenated to the 1053 text features ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L485-L495)). Therefore the `cls_score`/`det_cls_probs` vector has 1054 columns in the active default path. |
| Entropy normalization dimension | **VERIFIED** | `compute_entropy()` uses `probs.size(1)` / `probs.shape[1]`, not a hard-coded 1053 ([`ovtrack/models/trackers/uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L48-L78)). In the active path that is 1054, including background. |
| OVT-B benchmark category count and 1053-to-benchmark mapping | **NOT VERIFIED** | The checked-in class list has 1053 lines, while `data/ovtb/ovtb_ann.json` is absent. `OVTBDataset` constructs `cat_ids` and `cat2label` from the annotation categories and configured `CLASSES` ([`ovtrack/datasets/ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L23-L32)); the actual annotation category count and mapping cannot be checked here. Do not assert 1048 from repository evidence. |

## 2. Uncertainty Estimation

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Estimator | **VERIFIED** | [`ovtrack/models/trackers/uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L5-L42), `UncertaintyEstimator` | Implements entropy, top-two margin, score uncertainty, composite uncertainty, and per-track EMA state. | The implementation name is `estimate()`, not `compute_uncertainty()`. |
| Entropy equation | **VERIFIED with qualification** | `UncertaintyEstimator.compute_entropy()` ([`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L48-L78)) | $H=-\sum_c p_c\ln(p_c+\epsilon)/\ln C$, with `C` equal to the input vector width, `eps=1e-7`, `p_norm = p/(sum(p)+eps)`, and output clipped to `[0,1]`. | State $C=1054$ for the active non-`fix_bg` path, not 1053. |
| Background inclusion | **VERIFIED for active default path** | `OVTrackRoIHead.__init__` and `simple_test_bboxes()` ([`ovtrack/models/roi_heads/ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L80-L116); [#L485-L495](ovtrack/models/roi_heads/ovtrack_roi_head.py#L485-L495)) | Background is appended to text features unless `fix_bg=True`; configs do not set `fix_bg=True`. | Entropy includes background in the active path. |
| Zero-probability protection | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L60-L65), `compute_entropy()` | Adds `eps` to the row sum and inside `log()`. | The exact numerical protection may be stated. |
| Top-two margin | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L80-L98), `compute_margin()` | $M=\operatorname{clip}(1-(p_{(1)}-p_{(2)}),0,1)$. It uses the input values directly; it does not call `p_norm`. | Do not claim margin is computed from renormalized probabilities. |
| Score uncertainty | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L100-L117), `estimate()` | $S=\operatorname{clip}(1-s,0,1)$. | State the implementation clip, rather than assuming detector scores are always in range. |
| Composite weights | **VERIFIED** | Active values: [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L159-L163); normalization: [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L25-L40) | Active weights are entropy `0.60`, margin `0.25`, score `0.15`; constructor renormalizes any supplied weights by their sum. Defaults in the estimator are `0.4/0.3/0.3`. | Use `0.60/0.25/0.15` for UA-KF-OVT config claims. Hyperparameter search makes them configurable ([`tools/hparam_search.py`](tools/hparam_search.py#L35-L55)); no completed tuning result is available. |
| Output range | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L118-L143) | Composite output is clipped to `[0,1]`; score-only fallback is already clipped. | The range claim is implementation-supported. |
| Score-only fallback | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L135-L143); [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L198-L211) | If no class probabilities are available but scores are available, returns score-only uncertainty. In the active ROI-head UA path, class probabilities are supplied. | Describe fallback as available, not as the active UA path. |

## 3. Temporal Smoothing

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| State | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L40-L46), `track_uncertainties`; [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L90-L98) | Dictionary keyed by track ID; reset clears it. |
| EMA equation and initialization | **VERIFIED** | [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L145-L163), `update_track_ema()` | First observation initializes `bar_u=u`; later updates are `bar_u=(1-\gamma)bar_u+\gamma u`, with configurable `ema_gamma`. |
| Active smoothing parameter | **VERIFIED** | [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L162-L163) | `ema_gamma=0.4`. The estimator default is `0.5`. |
| Reset/removal | **VERIFIED** | Reset: [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L100-L103); pruning: [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L165-L170) | Tracker reset clears EMA; after each update, entries for IDs no longer in `self.tracks` are removed. |
| Raw versus smoothed for fusion | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L198-L211;L258-L273) | Association `dyn_w` uses the per-detection raw `uncertainties` tensor. |
| Raw versus smoothed for Kalman and appearance | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L281-L290); [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L160) | Matched detections update EMA; `_matched_uncertainties` stores smoothed values; `update_track()` uses that smoothed value for adaptive Kalman and appearance gating. |
| Raw versus smoothed for tentative deletion | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L105-L124), `pop_invalid_tracks()` | Tentative deletion/grace logic reads the smoothed track value, defaulting to `1.0` if no EMA exists. |

## 4. Adaptive Kalman Filter

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Adaptive implementation | **VERIFIED** | [`ovtrack/models/trackers/adaptive_kalman.py`](ovtrack/models/trackers/adaptive_kalman.py#L5-L47), `AdaptiveKalmanFilter` | Subclasses the repository `KalmanFilter`. |
| Covariance changed | **VERIFIED** | [`adaptive_kalman.py`](ovtrack/models/trackers/adaptive_kalman.py#L80-L107), `project()` | Only measurement/innovation covariance `R` is scaled. The predicted covariance/process noise, transition matrix, and observation matrix are inherited unchanged. |
| Mapping | **VERIFIED** | [`adaptive_kalman.py`](ovtrack/models/trackers/adaptive_kalman.py#L50-L78), `compute_r_factor()` | Config supports linear `1+beta*u`, exponential `exp(beta*u)`, and sigmoid mapping; active config selects linear. |
| Active parameters and cap | **VERIFIED** | [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L139-L145); [`adaptive_kalman.py`](ovtrack/models/trackers/adaptive_kalman.py#L33-L47) | `mapping='linear'`, `beta=3.0`, `min_scale=1.0`, `max_scale=10.0`; linear active range is `1` to `4`, so the cap is not reached for finite `u in [0,1]`. |
| Safeguards | **VERIFIED** | [`adaptive_kalman.py`](ovtrack/models/trackers/adaptive_kalman.py#L59-L78;L121-L143) | NaN/Inf uncertainty returns `max_scale`; uncertainty is clipped to `[0,1]`; factor is clipped to `[min_scale,max_scale]`; Cholesky failures use diagonal jitter and pseudoinverse; output covariance is symmetrized. |
| State and measurement | **VERIFIED** | State/transition: [`ovtrack/models/motions/kalman_filter.py`](ovtrack/models/motions/kalman_filter.py#L26-L45); initialization/measurement: [#L47-L72](ovtrack/models/motions/kalman_filter.py#L47-L72) | State is `[x,y,a,h,vx,vy,va,vh]` (8D); measurement is `[x,y,a,h]` (4D), obtained from `bbox_xyxy_to_cxcyah()` ([`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L150-L153)). |
| Uncertainty input | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L154-L160) | Active matched-track Kalman update receives smoothed uncertainty. |
| Fallback/bypass | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L154-L171) | If adaptive R is disabled, standard `kf.update()` is called. If a configured filter rejects the `uncertainty` keyword, only that specific `TypeError` path falls back to standard update. The active config supplies `AdaptiveKalmanFilter`. |

## 5. Association

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Association algorithm | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L273-L287), `track()` | Uses `linear_sum_assignment` (Hungarian assignment) on a track-by-detection cost matrix. |
| Appearance similarity | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L231-L257) | Computes exponentiated dot-product scores, bidirectional softmax (`d2t_scores`, `t2d_scores`), averages them, computes normalized cosine similarity mapped to `[0,1]`, then averages bi-softmax and cosine: `reid_dists=(scores+cos)/2`. Despite the variable name, this is a similarity before conversion to cost. |
| Motion/IoU | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L213-L231) | Predicts each active track with Kalman `predict()`, converts predicted `[x,y,a,h]` to xyxy, and computes box IoU. |
| Fusion formula | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L258-L273) | `w(u)=w_base+(w_max-w_base)u`; similarity `=(1-w)reid+w*IoU`; cost is `1-similarity` after transpose. Active values are `w_base=0.03`, `w_max=0.30`. |
| Appearance gate in association | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L148) | No separate pre-/post-fusion association gate exists. `appearance_gate_min` gates the memory EMA only. |
| Match threshold | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L281-L287); config [#L147-L172](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L147-L172) | Hungarian pairs are accepted when cost `<= match_score_thr`; active threshold is `0.5`. |
| Distractor preprocessing | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L213-L224) | Labels below `0` are invalid; detections with score `<0.5` are removed when overlapping an earlier detection by IoU `>0.3`. This occurs before association. |
| Published OVTrack+ equivalence | **NOT VERIFIED / CONFLICT** | Local implementation: [`ovsort_tracker.py`](ovtrack/models/trackers/ovsort_tracker.py#L38-L70); local UA override: [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L171) | `OVSortTracker.update_track()` calls its parent update and then performs another Kalman update. `UAKFOVTracker` deliberately avoids that parent path. No external OVTrack+ source or completed comparison is present. | Do not call the local OVTrack+ implementation an exact published-baseline reproduction. |

## 6. Appearance-Memory Update

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Base EMA convention | **VERIFIED** | [`ovtrack/models/trackers/base_tracker.py`](ovtrack/models/trackers/base_tracker.py#L116-L126), `BaseTracker.update_track()` | `memory_new=(1-m)*memory_old+m*new`; `m` is the new-detection update rate, not old-memory momentum. |
| UA gating equation | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L148) | For `embeds`, `m_eff=m*max(appearance_gate_min,1-u)`. Active `m=0.35`, floor `0.05`, so the effective update rate is bounded below by `0.0175` when gating is enabled. |
| Uncertainty used | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L281-L290) | `u` in `update_track()` is the smoothed matched-track uncertainty, not raw detection uncertainty. |
| Normalization before/after EMA | **NOT VERIFIED as performed** | Association normalizes temporary matrices at [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L244-L250); memory update itself is [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L139-L148) | No explicit normalization of the stored embedding before or after the EMA is performed in the memory update. | Do not claim normalized-memory EMA unless established by the track head implementation and runtime tensor contract. |
| Applied to every matched track | **VERIFIED with scope** | [`base_tracker.py`](ovtrack/models/trackers/base_tracker.py#L86-L114); [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L148) | Every ID passed to `BaseTracker.update()` that already exists invokes `UAKFOVTracker.update_track()`; the gate applies when `embeds` and uncertainty are available. New tracks are initialized directly. |

## 7. Tentative-Track Management

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Creation and IDs | **VERIFIED** | [`base_tracker.py`](ovtrack/models/trackers/base_tracker.py#L86-L114); [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L287-L307) | Unmatched detections receive monotonically increasing IDs and are initialized by `BaseTracker.init_track()`. `num_tracks` is never decremented, so removed IDs are not reused. |
| Tentative flag | **VERIFIED with caveat** | `SortTracker.init_track()` and promotion: [`ovtrack/models/trackers/sort_tracker.py`](ovtrack/models/trackers/sort_tracker.py#L83-L103); UAKF override: [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L134-L171) | `SortTracker.init_track()` sets `tentative=True`, but UAKF's custom `update_track()` does not contain the parent's promotion-to-confirmed code. Therefore the source does not establish that UAKF tracks become non-tentative. | Do not claim a tested confirmation lifecycle without a runtime test. |
| Aging/deletion | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L105-L124), `pop_invalid_tracks()` | Deletes tracks at `disappeared >= num_frames_retain` (`60` active config). Tentative unmatched tracks are deleted immediately except for a one-frame grace when smoothed uncertainty `<0.4`. |
| Uncertainty controlling deletion | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L112-L121) | Smoothed `bar_u`; missing state defaults to `1.0`. Grace condition is `bar_u<0.4` and `disappeared<=1`. |
| Reset/removal of uncertainty state | **VERIFIED** | [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L100-L103); [`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L165-L170) | Tracker reset clears EMA; post-update pruning removes EMA entries for removed IDs. |
| Independent track-management ablation | **NOT VERIFIED** | The active tracker exposes no separate `enable_tentative_management` flag in [`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L45-L98). `scratch/test_ablation_matrix.py` varies adaptive R, adaptive weight, and gating, but not lifecycle management ([`scratch/test_ablation_matrix.py`](scratch/test_ablation_matrix.py#L1-L25)). | The Capstone-style claim of a separately measured track-management ablation cannot be supported. |

## 8. Evaluation and Reproducibility

| Item | Status | Exact source, symbol, and lines | Verified behavior | Manuscript consequence |
|---|---|---|---|---|
| Evaluation command | **VERIFIED as prescribed; NOT VERIFIED as executed** | [`HANDOFF.md`](HANDOFF.md#L244-L270); command implementation [`tools/test.py`](tools/test.py#L1-L27) | Prescribed commands use `tools/test.py <config> saved_models/pretrained_models/ovtrack_clip_distillation.pth --eval track`. The checkpoint and result logs are absent here. |
| Evaluator | **VERIFIED** | [`ovtrack/datasets/ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L178-L218), `OVTBDataset.evaluate()` | Track evaluation imports external `teta`, constructs `teta.datasets.TAO`, and evaluates `teta.metrics.TETA()`. |
| Metric names | **NOT VERIFIED for actual evaluator output** | External call: [`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L193-L218); local parser expectation: [`tools/hparam_search.py`](tools/hparam_search.py#L91-L116) | The local parser expects columns `TETA`, `LocA`, `AssocA`, `ClsA`, plus recall/precision fields, but the TETA package is not vendored and no evaluator log is present. The source itself does not print a literal `AssA` field. | Preserve the evaluator/package spelling found in an actual log. `AssocA` is locally expected; `AssA` is not verified from this workspace. |
| Returned evaluation fields | **VERIFIED** | [`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L220-L275), `compute_teta_on_ovsetup()` | Dataset `evaluate()` explicitly returns `Base_TETA50` and `Novel_TETA50`. The full combined TETA table is printed/read from external TETA output but is not returned in `eval_results`. |
| Majority-vote class assignment | **VERIFIED** | [`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L9-L22;L85-L119) | `format_results()` calls `_track2json(..., tcc=True)` by default; `_track2json()` applies `majority_vote()` before writing `ovtb_track.json`. |
| IoU thresholds | **PARTIALLY VERIFIED** | Detector NMS thresholds: [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L171-L181); local CLEAR MOT default: [`ovtrack/core/evaluation/mot.py`](ovtrack/core/evaluation/mot.py#L22-L36;L117-L127) | Detector NMS uses 0.5 (RCNN) and 0.7 (RPN); UA distractor removal uses 0.3; TETA evaluation thresholds are taken from external TETA defaults and are not set in this repository. | Do not state a TETA IoU threshold without the installed TETA version/config. |
| Base/novel handling | **VERIFIED for source logic; NOT VERIFIED for counts** | [`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L232-L275) | Base means annotation category `frequency != 'r'`; novel means `frequency == 'r'`; means are computed from TETA's per-class `TETA[50]`. Annotation is absent, so counts cannot be checked. |
| Identical baseline/UA detections and post-processing | **VERIFIED at config level; NOT VERIFIED at result level** | Compare [`ovtrack_r50.py`](configs/ovtrack-teta/ovtb/ovtrack_r50.py#L1-L65;L150-L165) and [`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L1-L65;L171-L181) | Backbone, ROI head, score threshold, NMS, and max-per-image are configured identically. Tracker preprocessing differs: UA has its own uncertainty-aligned distractor filtering; OVTrack+ uses `OVSortTracker.remove_distractor()`. No paired runtime outputs exist. |
| Runtime, seeds, completed results | **NOT VERIFIED** | Search of repository files found no `paper.tex`, `project_context.md`, Capstone draft, checkpoint, evaluation log, CSV, result JSON/PTH, runtime report, or completed OVT-B ablation output. Synthetic tests are source files only. | Do not place numerical UA-KF-OVT results, speed, seed, or ablation improvements in the paper as completed experiments. |

## 9. Conflict Resolution

1. **`AssocA` versus `AssA`: NOT VERIFIED for evaluator output.** The external evaluator is invoked as `teta.metrics.TETA()` ([`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L193-L218)). The local hyperparameter parser explicitly expects `AssocA` ([`tools/hparam_search.py`](tools/hparam_search.py#L91-L116)), while no TETA output log is present. Use the spelling from an actual installed evaluator log; repository evidence supports `AssocA` as the local consumer spelling, not `AssA` as an independently verified evaluator field.
2. **Raw `u_i` versus smoothed `bar_u_j`: VERIFIED.** Raw per-detection uncertainty is used in adaptive association fusion ([`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L258-L273)). Smoothed track uncertainty is used for adaptive Kalman update, appearance gating, and tentative deletion ([`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L105-L124;L281-L290)).
3. **Independent uncertainty-aware tentative-track ablation: NOT VERIFIED.** No lifecycle switch exists and `scratch/test_ablation_matrix.py` does not measure it ([`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L45-L98); [`scratch/test_ablation_matrix.py`](scratch/test_ablation_matrix.py#L1-L25)).
4. **Appearance EMA equation: CONFLICT with any formulation that treats the coefficient as old-memory momentum.** Executable code uses `new=(1-m)old+m*det` ([`base_tracker.py`](ovtrack/models/trackers/base_tracker.py#L116-L126)); UA changes the new-detection rate to `m*max(gate_min,1-u)` ([`uakf_ovtracker.py`](ovtrack/models/trackers/uakf_ovtracker.py#L139-L148)). Active `m=0.35`, `gate_min=0.05` ([`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L150-L163)).
5. **1,048 benchmark categories versus 1,053 detector classes: NOT VERIFIED.** The checked-in detector/class-list dimension is 1053; `ovtb_ann.json` is absent. The class mapping is constructed from annotation `cat_ids` and configured `CLASSES` ([`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L23-L32)). Active entropy dimension is the input vector width, 1054 including background ([`uncertainty_estimator.py`](ovtrack/models/trackers/uncertainty_estimator.py#L48-L78); [`ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L485-L495)).
6. **Class probabilities available: VERIFIED.** `simple_test_bboxes()` returns post-NMS matched class-score rows ([`ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L556-L603)); `simple_test()` passes them to `UncertaintyEstimator.estimate()` ([`ovtrack_roi_head.py`](ovtrack/models/roi_heads/ovtrack_roi_head.py#L405-L420)). The earlier score-only statement in `ISSUES_AND_RECOMMENDATIONS.md` is stale and conflicts with current code ([`ISSUES_AND_RECOMMENDATIONS.md`](ISSUES_AND_RECOMMENDATIONS.md#L100-L125)).
7. **Published OVTrack+ row versus local baseline: NOT VERIFIED.** The corrected local commands/configs are listed in [`HANDOFF.md`](HANDOFF.md#L244-L270); the local OVTrack+ config uses `OVSortTracker` + `KalmanFilter` ([`ovtrack_plus.py`](configs/ovtrack-teta/ovtb/ovtrack_plus.py#L135-L155)). No checkpoint or completed baseline log is present, so the discrepancy cannot be attributed to weights versus versions versus platform from repository evidence.

## 10. Implementation-Verified Equations

The following equations can be stated as implementation facts, with the qualifications above:

1. Normalized entropy, using the actual input width $C$:
   $$H(p) = \operatorname{clip}\left(-\frac{\sum_c \hat p_c\ln(\hat p_c+\epsilon)}{\ln C},0,1\right), \qquad \hat p=\frac{p}{\sum_c p_c+\epsilon},\quad \epsilon=10^{-7}.$$
2. Top-two margin uncertainty:
   $$M(p)=\operatorname{clip}\left(1-(p_{(1)}-p_{(2)}),0,1\right).$$
3. Detection-score uncertainty:
   $$S(s)=\operatorname{clip}(1-s,0,1).$$
4. Active composite uncertainty:
   $$u_i=0.60H_i+0.25M_i+0.15S_i,$$ with final clipping to $[0,1]$.
5. Track EMA:
   $$\bar u_j^{(t)}=(1-\gamma)\bar u_j^{(t-1)}+\gamma u_i^{(t)},\qquad \gamma=0.4\text{ in the active UA config},$$
   with first observation used directly.
6. Active adaptive measurement-noise scaling:
   $$R(u)=R_0(1+3u),$$
   with uncertainty clipped to $[0,1]$ and scale clipped to $[1,10]$.
7. Active association fusion:
   $$w(u)=0.03+(0.30-0.03)u,$$ $$S_{match}=(1-w(u))S_{ReID}+w(u)IoU,\qquad D=1-S_{match}.$$
8. Appearance memory update:
   $$m_{eff}=0.35\max(0.05,1-\bar u_j),\qquad e_j^{new}=(1-m_{eff})e_j^{old}+m_{eff}e_i.$$

## 11. Claims That Must Remain Placeholders or Be Removed

- Any numerical OVT-B TETA, LocA, AssocA/AssA, ClsA, base/novel, runtime, or memory result for UA-KF-OVT.
- Any claim that the reported results were produced with the configured checkpoint; the binary and logs are absent.
- Any claim that the 1,048 benchmark categories and 1,053 detector classes have been verified to map correctly; the annotation file is absent.
- Any claim that TETA's evaluator field is exactly `AssA` or exactly `AssocA` without an actual evaluator log.
- Any claim that UA-KF-OVT has an independently switchable or measured uncertainty-aware tentative-track ablation.
- Any claim that stored appearance embeddings are normalized before and after the EMA.
- Any claim of exact published OVTrack+ reproduction or an identified cause of the local baseline discrepancy.
- The prose values `w_max=0.75` and active uncertainty defaults `0.4/0.3/0.3` when describing the active `ovtrack_r50_ua.py` experiment. The executable active values are `0.30` and `0.60/0.25/0.15` respectively ([`ovtrack_r50_ua.py`](configs/ovtrack-teta/ovtb/ovtrack_r50_ua.py#L147-L172)).
- Any statement that the current ROI-head path is score-only; current source passes class-probability rows.

## 12. Exact Result Fields Available for a Results Table

From executable repository code, the following fields are available in principle:

- Dataset-returned evaluation dictionary: `Base_TETA50`, `Novel_TETA50` ([`ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L220-L275)).
- External TETA console/parser schema expected by the repository: `TETA`, `LocA`, `AssocA`, `ClsA`, `LocRe`, `LocPr`, `AssocRe`, `AssocPr`, `ClsRe`, `ClsPr` ([`tools/hparam_search.py`](tools/hparam_search.py#L91-L116)). These are **not verified as populated results** because no log exists.
- Per-image raw output structures: `bbox_results` and `track_results` ([`ovtrack/models/mot/ovtrack.py`](ovtrack/models/mot/ovtrack.py#L194-L207)); track JSON fields include `image_id`, `bbox`, `score`, `category_id`, `video_id`, and `track_id` ([`ovtrack/datasets/ovtb_dataset.py`](ovtrack/datasets/ovtb_dataset.py#L63-L83)).

No completed values for these fields are available in the workspace. The results table should therefore retain placeholders until the baseline, OVTrack+, and UA-KF-OVT commands are run with the checkpoint, annotation file, installed TETA version, and logs preserved.

## 13. Evidence Gaps

- `paper.tex`, `project_context.md`, and a Capstone draft are not present, so their exact conflicting statements cannot be line-audited here.
- `data/ovtb/ovtb_ann.json`, `saved_models/`, evaluation logs, result files, runtime measurements, and random-seed records are not present.
- `teta` is an external dependency; its source/version and rendered output are not vendored.
- The synthetic files under `scratch/` define tests and printed expectations, but they are not completed OVT-B experiment evidence. In particular, `scratch/test_ablation_matrix.py` uses a synthetic sequence and fixed seeds in code; it does not establish benchmark results ([`scratch/test_ablation_matrix.py`](scratch/test_ablation_matrix.py#L27-L107)).