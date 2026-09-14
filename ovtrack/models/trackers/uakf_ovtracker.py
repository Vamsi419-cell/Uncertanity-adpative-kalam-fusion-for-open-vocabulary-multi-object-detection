
import torch

from mmcv.utils import Registry

from .ovtracker import OVTracker
from .uncertainty_estimator import (
    UncertaintyEstimator
)


class UAKFOVTracker(OVTracker):
    """
    UA-KF-OVT tracker.

    Original OVTracker is NOT modified.

    Adds:
      - semantic uncertainty
      - temporal EMA
      - adaptive R scaling
      - adaptive appearance/motion weight
    """

    def __init__(
        self,
        *args,
        uncertainty=None,
        adaptive_r=None,
        adaptive_w=None,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        uncertainty = (
            uncertainty
            if uncertainty is not None
            else {}
        )

        adaptive_r = (
            adaptive_r
            if adaptive_r is not None
            else {}
        )

        adaptive_w = (
            adaptive_w
            if adaptive_w is not None
            else {}
        )

        self.uncertainty_cfg = uncertainty
        self.adaptive_r_cfg = adaptive_r
        self.adaptive_w_cfg = adaptive_w

        self.uncertainty_estimator = (
            UncertaintyEstimator(
                alpha=uncertainty.get(
                    "alpha",
                    1.0
                ),
                beta=uncertainty.get(
                    "beta",
                    1.0
                ),
                ema_mu=uncertainty.get(
                    "ema_mu",
                    0.8
                )
            )
        )

        self.last_uncertainties = {}
        self.last_fusion_weights = {}

    def reset_uncertainty(self):

        self.uncertainty_estimator.reset()

        self.last_uncertainties.clear()
        self.last_fusion_weights.clear()

    def compute_uncertainty(
        self,
        similarities,
        track_ids=None
    ):

        uncertainty = (
            self.uncertainty_estimator.compute(
                similarities
            )
        )

        if track_ids is not None:

            uncertainty = (
                self.uncertainty_estimator.update(
                    track_ids,
                    uncertainty
                )
            )

        return uncertainty

    def adaptive_r_scale(
        self,
        uncertainty
    ):

        if not self.adaptive_r_cfg.get(
            "enabled",
            True
        ):
            return 1.0

        gamma = self.adaptive_r_cfg.get(
            "gamma",
            1.0
        )

        u = float(
            torch.as_tensor(
                uncertainty
            )
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
        )

        mapping = self.adaptive_r_cfg.get(
            "mapping",
            "linear"
        )

        if mapping == "exponential":
            return float(
                torch.exp(
                    torch.tensor(
                        gamma * u
                    )
                )
            )

        return 1.0 + gamma * u

    def adaptive_fusion_weight(
        self,
        uncertainty
    ):

        w_base = self.adaptive_w_cfg.get(
            "w_base",
            0.03
        )

        if not self.adaptive_w_cfg.get(
            "enabled",
            True
        ):
            return w_base

        delta = self.adaptive_w_cfg.get(
            "delta",
            0.10
        )

        u = float(
            torch.as_tensor(
                uncertainty
            )
            .detach()
            .cpu()
            .clamp(0.0, 1.0)
        )

        w = w_base + delta * u

        return max(
            0.0,
            min(1.0, w)
        )

    def get_adaptive_parameters(
        self,
        uncertainty
    ):

        return {
            "uncertainty": float(
                torch.as_tensor(
                    uncertainty
                )
                .detach()
                .cpu()
            ),

            "r_scale": self.adaptive_r_scale(
                uncertainty
            ),

            "fusion_weight": (
                self.adaptive_fusion_weight(
                    uncertainty
                )
            )
        }
