import torch
import torch.nn.functional as F
import numpy as np


class UncertaintyEstimator:
    """Semantic and detection uncertainty estimator for Open-Vocabulary MOT.
    
    Quantifies uncertainty from:
    1. Normalized Shannon Entropy over open-vocabulary class probabilities:
       H(p) = - (1 / ln(C)) * sum_c p_c ln(p_c + eps) in [0, 1]
    2. Top-1 vs Top-2 margin uncertainty:
       M(p) = 1 - (p_top1 - p_top2) in [0, 1]
    3. Detection score uncertainty:
       S(p) = 1 - score in [0, 1]
    
    Composite Uncertainty:
       u = w_entropy * H + w_margin * M + w_score * S
       
    Also maintains track-specific temporal EMA without leakage across IDs.
    """

    def __init__(
        self,
        w_entropy=0.4,
        w_margin=0.3,
        w_score=0.3,
        ema_gamma=0.5,
        eps=1e-7,
    ):
        self.w_entropy = w_entropy
        self.w_margin = w_margin
        self.w_score = w_score
        total_w = w_entropy + w_margin + w_score
        assert total_w > 0, "Total uncertainty weights must be positive"
        self.w_entropy /= total_w
        self.w_margin /= total_w
        self.w_score /= total_w

        self.ema_gamma = float(ema_gamma)
        self.eps = eps
        self.track_uncertainties = dict()

    def reset(self):
        """Reset internal track-specific memory."""
        self.track_uncertainties.clear()

    def compute_entropy(self, probs):
        """Compute normalized Shannon entropy in [0, 1].
        
        Args:
            probs (torch.Tensor or np.ndarray): Shape (N, C) probability distribution.
            
        Returns:
            torch.Tensor or np.ndarray: Shape (N,) normalized entropy.
        """
        if isinstance(probs, torch.Tensor):
            num_classes = probs.size(1)
            if num_classes <= 1:
                return torch.zeros((probs.size(0),), dtype=probs.dtype, device=probs.device)
            # Ensure probabilities sum to 1
            p_sum = probs.sum(dim=-1, keepdim=True) + self.eps
            p_norm = probs / p_sum
            log_p = torch.log(p_norm + self.eps)
            entropy = -torch.sum(p_norm * log_p, dim=-1)
            max_entropy = np.log(num_classes)
            norm_entropy = torch.clamp(entropy / max_entropy, 0.0, 1.0)
            return norm_entropy
        else:
            probs = np.asarray(probs, dtype=np.float32)
            num_classes = probs.shape[1]
            if num_classes <= 1:
                return np.zeros((probs.shape[0],), dtype=np.float32)
            p_norm = probs / (np.sum(probs, axis=-1, keepdims=True) + self.eps)
            log_p = np.log(p_norm + self.eps)
            entropy = -np.sum(p_norm * log_p, axis=-1)
            max_entropy = np.log(num_classes)
            return np.clip(entropy / max_entropy, 0.0, 1.0)

    def compute_margin(self, probs):
        """Compute top-1 vs top-2 margin uncertainty in [0, 1].
        
        Small margin indicates ambiguous identity -> uncertainty approaches 1.
        Large margin indicates high confidence -> uncertainty approaches 0.
        """
        if isinstance(probs, torch.Tensor):
            if probs.size(1) < 2:
                return torch.zeros((probs.size(0),), dtype=probs.dtype, device=probs.device)
            topk, _ = torch.topk(probs, k=2, dim=-1)
            margin = topk[:, 0] - topk[:, 1]
            return torch.clamp(1.0 - margin, 0.0, 1.0)
        else:
            probs = np.asarray(probs, dtype=np.float32)
            if probs.shape[1] < 2:
                return np.zeros((probs.shape[0],), dtype=np.float32)
            sorted_p = np.sort(probs, axis=-1)[:, ::-1]
            margin = sorted_p[:, 0] - sorted_p[:, 1]
            return np.clip(1.0 - margin, 0.0, 1.0)

    def estimate(self, cls_probs=None, scores=None, num_dets=None):
        """Estimate composite uncertainty for a set of detections.
        
        Args:
            cls_probs (torch.Tensor or np.ndarray, optional): (N, C) class probabilities.
            scores (torch.Tensor or np.ndarray, optional): (N,) detection scores.
            num_dets (int, optional): Number of detections if both above are None.
            
        Returns:
            torch.Tensor or np.ndarray: (N,) composite uncertainty in [0, 1].
        """
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                score_unc = torch.clamp(1.0 - scores, 0.0, 1.0)
            else:
                scores = np.asarray(scores, dtype=np.float32)
                score_unc = np.clip(1.0 - scores, 0.0, 1.0)
        else:
            score_unc = None

        if cls_probs is not None and cls_probs.shape[0] > 0:
            entropy_unc = self.compute_entropy(cls_probs)
            margin_unc = self.compute_margin(cls_probs)

            if score_unc is not None:
                composite = (
                    self.w_entropy * entropy_unc
                    + self.w_margin * margin_unc
                    + self.w_score * score_unc
                )
            else:
                norm_factor = self.w_entropy + self.w_margin
                composite = (self.w_entropy * entropy_unc + self.w_margin * margin_unc) / norm_factor
            
            if isinstance(composite, torch.Tensor):
                return torch.clamp(composite, 0.0, 1.0)
            return np.clip(composite, 0.0, 1.0)

        elif score_unc is not None:
            # Fallback when full class distributions are unavailable: strictly score-based
            return score_unc
        else:
            n = num_dets if num_dets is not None else 0
            return np.zeros((n,), dtype=np.float32)

    def update_track_ema(self, track_id, raw_uncertainty):
        """Update track-specific smoothed uncertainty using EMA.
        
        Args:
            track_id (int): Track ID.
            raw_uncertainty (float): Instantaneous uncertainty for this frame.
            
        Returns:
            float: Smoothed uncertainty for track.
        """
        u = float(raw_uncertainty)
        if track_id not in self.track_uncertainties:
            self.track_uncertainties[track_id] = u
        else:
            prev_u = self.track_uncertainties[track_id]
            self.track_uncertainties[track_id] = (
                (1.0 - self.ema_gamma) * prev_u + self.ema_gamma * u
            )
        return self.track_uncertainties[track_id]

    def prune_tracks(self, active_track_ids):
        """Prune removed tracks from memory to prevent leakage and unbounded growth."""
        active_set = set(int(t) for t in active_track_ids)
        stale_ids = [t for t in self.track_uncertainties if t not in active_set]
        for t in stale_ids:
            del self.track_uncertainties[t]
