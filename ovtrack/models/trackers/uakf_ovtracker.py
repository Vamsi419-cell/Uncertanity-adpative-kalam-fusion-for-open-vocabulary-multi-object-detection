import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

try:
    from mmdet.core import bbox_overlaps
except ImportError:
    def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        if bboxes1.numel() == 0 or bboxes2.numel() == 0:
            return torch.zeros((bboxes1.size(0), bboxes2.size(0)), device=bboxes1.device)
        b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1[:, 0:1], bboxes1[:, 1:2], bboxes1[:, 2:3], bboxes1[:, 3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2[:, 0:1].t(), bboxes2[:, 1:2].t(), bboxes2[:, 2:3].t(), bboxes2[:, 3:4].t()
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        area1 = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
        area2 = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)
        union_area = area1 + area2 - inter_area
        return inter_area / torch.clamp(union_area, min=eps)

try:
    from motmetrics.lap import linear_sum_assignment
except ImportError:
    from scipy.optimize import linear_sum_assignment

from ovtrack.core.bbox import bbox_xyxy_to_cxcyah, bbox_cxcyah_to_xyxy
from ..builder import TRACKERS
from .ovsort_tracker import OVSortTracker
from .uncertainty_estimator import UncertaintyEstimator
from .adaptive_kalman import AdaptiveKalmanFilter


@TRACKERS.register_module()
class UAKFOVTracker(OVSortTracker):
    """Uncertainty-Adaptive Kalman Fusion Open-Vocabulary Tracker (UA-KF-OVT).
    
    Research components:
    1. Semantic and detection uncertainty quantification (entropy, margin, score).
    2. Dynamic measurement noise covariance R(u) in Kalman update.
    3. Uncertainty-aware appearance vs. motion cost fusion: w(u) * IoU + (1 - w(u)) * ReID.
    4. Uncertainty-gated appearance memory update to protect against identity corruption.
    """

    def __init__(
        self,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        num_frames_retain=30,
        momentum_embed=0.2,
        motion_weight=0.03,
        # UA-KF research parameters:
        motion_weight_max=0.35,
        enable_adaptive_r=True,
        enable_adaptive_weight=True,
        enable_appearance_gating=True,
        kalman_mapping='linear',
        kalman_beta=3.0,
        kalman_max_scale=10.0,
        w_entropy=0.4,
        w_margin=0.3,
        w_score=0.3,
        ema_gamma=0.5,
        num_tentatives=1,
        appearance_gate_min=0.05,
        init_cfg=None,
        **kwargs
    ):
        super().__init__(
            obj_score_thr=obj_score_thr,
            match_score_thr=match_score_thr,
            num_frames_retain=num_frames_retain,
            momentum_embed=momentum_embed,
            motion_weight=motion_weight,
            num_tentatives=num_tentatives,
            init_cfg=init_cfg,
            **kwargs
        )
        self.motion_weight_base = motion_weight
        self.motion_weight_max = motion_weight_max
        self.enable_adaptive_r = enable_adaptive_r
        self.enable_adaptive_weight = enable_adaptive_weight
        self.enable_appearance_gating = enable_appearance_gating
        self.kalman_mapping = kalman_mapping
        self.kalman_beta = kalman_beta
        self.kalman_max_scale = kalman_max_scale
        self.appearance_gate_min = appearance_gate_min

        self.uncertainty_estimator = UncertaintyEstimator(
            w_entropy=w_entropy,
            w_margin=w_margin,
            w_score=w_score,
            ema_gamma=ema_gamma,
        )
        self._matched_uncertainties = {}

    def pop_invalid_tracks(self, frame_id):
        """Uncertainty-aware track invalidation and pruning.
        
        Confidently initiated tracks (low historical uncertainty) are protected with a
        grace period during brief 1-frame occlusions to prevent premature deletion.
        """
        invalid_ids = []
        for k, v in self.tracks.items():
            disappeared = frame_id - v['frame_ids'][-1]
            case1 = disappeared >= self.num_frames_retain

            if v.tentative and v['frame_ids'][-1] != frame_id:
                # Check if track was confident
                smoothed_u = self.uncertainty_estimator.track_uncertainties.get(k, 1.0)
                if smoothed_u < 0.4 and disappeared <= 1:
                    case2 = False
                else:
                    case2 = True
            else:
                case2 = False

            if case1 or case2:
                invalid_ids.append(k)

        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def reset(self):
        super().reset()
        if hasattr(self, 'uncertainty_estimator'):
            self.uncertainty_estimator.reset()
        self._matched_uncertainties = {}

    def update_track(self, id, obj):
        """Update track with uncertainty-gated appearance and adaptive Kalman."""
        super().update_track(id, obj)

        # Retrieve matched uncertainty for this track in the current frame
        u = self._matched_uncertainties.get(id, None)

        # Apply uncertainty-gated appearance update if enabled
        if self.enable_appearance_gating and u is not None:
            # Embeddings are stored under 'embeds' in obj
            embed_idx = self.memo_items.index('embeds') if 'embeds' in self.memo_items else -1
            if embed_idx >= 0:
                new_embed = obj[embed_idx][None]
                base_m = self.momentums.get('embeds', 0.2)
                # Gate learning rate: confident -> high alpha; uncertain -> low alpha
                alpha = base_m * max(self.appearance_gate_min, 1.0 - float(u))
                # Re-smooth embedding with gated alpha
                self.tracks[id]['embeds'] = (1.0 - alpha) * self.tracks[id]['embeds'] + alpha * new_embed

        # Adaptive Kalman update
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()

        if self.enable_adaptive_r and hasattr(self.kf, 'update'):
            # Check if self.kf supports the uncertainty parameter
            try:
                self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
                    self.tracks[id].mean, self.tracks[id].covariance, bbox, uncertainty=u)
            except TypeError:
                # Standard KalmanFilter fallback
                self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
                    self.tracks[id].mean, self.tracks[id].covariance, bbox)
        else:
            self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
                self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def track(
        self,
        model,
        bboxes,
        labels,
        embeds,
        cls_embeds,
        frame_id,
        uncertainties=None,
        cls_probs=None,
        **kwargs
    ):
        """Track forward with uncertainty-adaptive matching."""
        if not hasattr(self, 'kf'):
            if isinstance(model.motion, AdaptiveKalmanFilter):
                self.kf = model.motion
            elif self.enable_adaptive_r:
                # Wrap or instantiate AdaptiveKalmanFilter
                self.kf = AdaptiveKalmanFilter(
                    mapping=self.kalman_mapping,
                    beta=self.kalman_beta,
                    max_scale=self.kalman_max_scale,
                )
            else:
                self.kf = model.motion

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        # Compute uncertainties if not supplied externally
        if uncertainties is None:
            if cls_probs is not None:
                uncertainties = self.uncertainty_estimator.estimate(
                    cls_probs=cls_probs, scores=bboxes[:, -1]
                )
            else:
                # Fallback: score-derived uncertainty
                uncertainties = self.uncertainty_estimator.estimate(
                    scores=bboxes[:, -1], num_dets=bboxes.size(0)
                )

        if isinstance(uncertainties, np.ndarray):
            uncertainties = torch.from_numpy(uncertainties).to(bboxes.device)

        # Distractor removal (preserves alignment with uncertainties)
        valid_inds = labels > -1
        low_inds = torch.nonzero(
            bboxes[:, -1] < 0.5, as_tuple=False
        ).squeeze(1)
        ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > 0.3).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = embeds[valid_inds]
        if uncertainties is not None and uncertainties.numel() > 0:
            uncertainties = uncertainties[valid_inds]

        if bboxes.size(0) > 0 and not self.empty:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            active_ids = [track_id for track_id, _ in self.tracks.items()]

            pred_bbox = torch.zeros((len(self.tracks), 4), device=bboxes.device)
            for i, (track_id, track) in enumerate(self.tracks.items()):
                predict_mean, _ = self.kf.predict(track['mean'], track['covariance'])
                pred_bbox[i] = torch.tensor(
                    predict_mean[:4], dtype=torch.float32, device=bboxes.device)

            pred_bbox_xyxy = bbox_cxcyah_to_xyxy(pred_bbox)
            iou_dists = bbox_overlaps(bboxes[:, :4], pred_bbox_xyxy)

            track_embeds = self.get('embeds', active_ids)
            sims = torch.mm(embeds, track_embeds.t())
            # Numerically stable bi-directional softmax
            sims_clamped = torch.clamp(sims, -30.0, 30.0)
            exps = torch.exp(sims_clamped)
            d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
            t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
            scores = (d2t_scores + t2d_scores) / 2

            cos_embeds = F.normalize(embeds, p=2, dim=1)
            cos_track_embeds = F.normalize(track_embeds, p=2, dim=1)
            cos = torch.mm(cos_embeds, cos_track_embeds.t())
            cos = (1.0 + cos) / 2
            reid_dists = (scores + cos) / 2

            # Dynamic uncertainty-aware fusion weight w(u)
            if self.enable_adaptive_weight and uncertainties is not None and uncertainties.numel() > 0:
                u_col = uncertainties.view(-1, 1)  # (N_dets, 1)
                dyn_w = self.motion_weight_base + (
                    self.motion_weight_max - self.motion_weight_base
                ) * u_col
                match_dists = (1.0 - dyn_w) * reid_dists + dyn_w * iou_dists
            else:
                match_dists = (
                    (1.0 - self.motion_weight_base) * reid_dists
                    + self.motion_weight_base * iou_dists
                )

            # Cost matrix: tracks (rows) x detections (cols)
            match_dists = 1.0 - match_dists.T
            # Defensively clean any NaN/Inf
            match_dists = torch.nan_to_num(match_dists, nan=1.0, posinf=1.0, neginf=0.0)
            row, col = linear_sum_assignment(match_dists.cpu().numpy())

            self._matched_uncertainties.clear()
            for r, c in zip(row, col):
                dist = match_dists[r, c]
                if dist <= self.match_score_thr:
                    matched_track_id = active_ids[r]
                    ids[c] = matched_track_id
                    u_val = float(uncertainties[c].item()) if (uncertainties is not None and uncertainties.numel() > 0) else 0.0
                    smoothed_u = self.uncertainty_estimator.update_track_ema(
                        matched_track_id, u_val)
                    self._matched_uncertainties[matched_track_id] = smoothed_u

            ids = ids.to(bboxes.device)
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long,
            ).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        else:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long,
            ).to(bboxes.device)
            self.num_tracks += num_new_tracks
            self._matched_uncertainties.clear()

        # Update tracker state
        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            scores=bboxes[:, -1],
            labels=labels,
            embeds=embeds,
            frame_ids=frame_id,
        )

        # Prune dead tracks from uncertainty memory
        self.uncertainty_estimator.prune_tracks(self.tracks.keys())

        return bboxes, labels, ids
