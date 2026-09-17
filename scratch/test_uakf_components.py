"""Unit tests for UA-KF-OVT research components.

Tests:
1. UncertaintyEstimator: mathematical validity, entropy, margin, bounds [0, 1], EMA smoothing.
2. AdaptiveKalmanFilter: R(u) scaling, baseline Kalman invariance, numerical stability.
3. UAKFOVTracker: tracking loop, dynamic association weighting, appearance gating, state management.
4. Baseline Invariance: verifying baseline compatibility.
"""

import sys
import os
import numpy as np
import torch

# Ensure repository root is in python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ovtrack.models.trackers.uncertainty_estimator import UncertaintyEstimator
from ovtrack.models.trackers.adaptive_kalman import AdaptiveKalmanFilter
from ovtrack.models.motions.kalman_filter import KalmanFilter


def test_uncertainty_estimator():
    print("--- [TEST 1] UncertaintyEstimator ---")
    estimator = UncertaintyEstimator(w_entropy=0.4, w_margin=0.3, w_score=0.3, ema_gamma=0.5)

    # 1. Peaked distribution (confident) vs Uniform distribution (maximum uncertainty)
    peaked = torch.zeros((1, 10))
    peaked[0, 0] = 1.0
    uniform = torch.ones((1, 10)) / 10.0

    h_peaked = estimator.compute_entropy(peaked).item()
    h_uniform = estimator.compute_entropy(uniform).item()
    assert 0.0 <= h_peaked < 0.05, f"Expected near-zero entropy for peaked, got {h_peaked}"
    assert 0.95 < h_uniform <= 1.0, f"Expected near-one entropy for uniform, got {h_uniform}"
    print(f"  Entropy: peaked={h_peaked:.4f}, uniform={h_uniform:.4f} [PASS]")

    # 2. Margin
    m_peaked = estimator.compute_margin(peaked).item()
    m_uniform = estimator.compute_margin(uniform).item()
    assert m_peaked < 0.05, f"Expected near-zero margin uncertainty for peaked, got {m_peaked}"
    assert m_uniform > 0.95, f"Expected near-one margin uncertainty for uniform, got {m_uniform}"
    print(f"  Margin: peaked={m_peaked:.4f}, uniform={m_uniform:.4f} [PASS]")

    # 3. Composite uncertainty bounds
    scores = torch.tensor([0.95, 0.1])
    probs = torch.cat([peaked, uniform], dim=0)
    u = estimator.estimate(cls_probs=probs, scores=scores)
    assert u[0] < u[1], f"Expected u[0] < u[1], got {u[0]} vs {u[1]}"
    assert (u >= 0.0).all() and (u <= 1.0).all(), "Uncertainty outside [0, 1]"
    print(f"  Composite: confident={u[0].item():.4f}, uncertain={u[1].item():.4f} [PASS]")

    # 4. Track-specific temporal EMA
    track_id = 42
    u1 = estimator.update_track_ema(track_id, 0.8)
    assert u1 == 0.8, "First observation should set EMA directly"
    u2 = estimator.update_track_ema(track_id, 0.2)
    assert abs(u2 - 0.5) < 1e-5, f"Expected 0.5 EMA, got {u2}"
    print(f"  Track EMA smoothing: step 1={u1}, step 2={u2} [PASS]")

    # 5. Prune tracks
    estimator.prune_tracks(active_track_ids=[99])
    assert track_id not in estimator.track_uncertainties, "Dead track was not pruned"
    print("  Track memory pruning: [PASS]")
    print("[TEST 1 PASSED]\n")


def test_adaptive_kalman():
    print("--- [TEST 2] AdaptiveKalmanFilter ---")
    base_kf = KalmanFilter()
    akf_linear = AdaptiveKalmanFilter(mapping="linear", beta=3.0, min_scale=1.0, max_scale=10.0)
    akf_exp = AdaptiveKalmanFilter(mapping="exponential", beta=1.5, min_scale=1.0, max_scale=10.0)

    # 1. Baseline invariance when u = 0 or None
    meas = np.array([100.0, 100.0, 1.5, 50.0])
    mean_base, cov_base = base_kf.initiate(meas)
    mean_akf, cov_akf = akf_linear.initiate(meas)
    assert np.allclose(mean_base, mean_akf), "Initiate mean mismatch"
    assert np.allclose(cov_base, cov_akf), "Initiate cov mismatch"

    # Project and update with u=None
    p_mean_base, p_cov_base = base_kf.project(mean_base, cov_base)
    p_mean_akf, p_cov_akf = akf_linear.project(mean_akf, cov_akf, uncertainty=None)
    assert np.allclose(p_mean_base, p_mean_akf), "Project mean mismatch on baseline"
    assert np.allclose(p_cov_base, p_cov_akf), "Project cov mismatch on baseline"
    print("  Baseline exact equivalence (u=None): [PASS]")

    # 2. Monotonic R scaling with uncertainty
    r_factor_0 = akf_linear.compute_r_factor(0.0)
    r_factor_half = akf_linear.compute_r_factor(0.5)
    r_factor_1 = akf_linear.compute_r_factor(1.0)
    assert r_factor_0 == 1.0
    assert r_factor_half == 2.5
    assert r_factor_1 == 4.0
    print(f"  Linear R-factors: u=0 -> {r_factor_0}, u=0.5 -> {r_factor_half}, u=1.0 -> {r_factor_1} [PASS]")

    # 3. Kalman Gain sensitivity
    # Low uncertainty update
    new_m_low, new_c_low = akf_linear.update(mean_akf, cov_akf, meas + np.array([10.0, 10.0, 0, 0]), uncertainty=0.0)
    # High uncertainty update
    new_m_high, new_c_high = akf_linear.update(mean_akf, cov_akf, meas + np.array([10.0, 10.0, 0, 0]), uncertainty=1.0)

    # When uncertainty is high, Kalman gain is lower -> state shifts LESS toward measurement!
    shift_low = np.linalg.norm(new_m_low[:2] - mean_akf[:2])
    shift_high = np.linalg.norm(new_m_high[:2] - mean_akf[:2])
    assert shift_low > shift_high, f"Expected shift_low ({shift_low}) > shift_high ({shift_high})"
    print(f"  Trust calibration: low_u shift={shift_low:.4f} > high_u shift={shift_high:.4f} [PASS]")

    # 4. Exponential mapping
    r_exp_0 = akf_exp.compute_r_factor(0.0)
    r_exp_1 = akf_exp.compute_r_factor(1.0)
    assert abs(r_exp_0 - 1.0) < 1e-4
    assert abs(r_exp_1 - np.exp(1.5)) < 1e-4
    print(f"  Exponential R-factors: u=0 -> {r_exp_0:.4f}, u=1.0 -> {r_exp_1:.4f} [PASS]")
    print("[TEST 2 PASSED]\n")


def test_uakf_tracker_loop():
    print("--- [TEST 3] UAKFOVTracker Simulation Loop ---")
    from ovtrack.models.trackers.uakf_ovtracker import UAKFOVTracker

    tracker = UAKFOVTracker(
        motion_weight=0.03,
        motion_weight_max=0.35,
        enable_adaptive_r=True,
        enable_adaptive_weight=True,
        enable_appearance_gating=True,
    )

    class DummyModel:
        motion = AdaptiveKalmanFilter()

    model = DummyModel()

    # Frame 0: Initialize two objects
    bboxes_f0 = torch.tensor([
        [100.0, 100.0, 150.0, 180.0, 0.95],
        [300.0, 300.0, 350.0, 380.0, 0.85],
    ])
    labels_f0 = torch.tensor([1, 2])
    embeds_f0 = torch.nn.functional.normalize(torch.randn(2, 256), p=2, dim=1)
    cls_embeds_f0 = torch.nn.functional.normalize(torch.randn(2, 256), p=2, dim=1)

    b0, l0, ids0 = tracker.track(
        model=model,
        bboxes=bboxes_f0,
        labels=labels_f0,
        embeds=embeds_f0,
        cls_embeds=cls_embeds_f0,
        frame_id=0,
    )
    assert len(ids0) == 2
    assert ids0[0] == 0 and ids0[1] == 1
    print(f"  Frame 0: Spawned tracks with IDs {ids0.tolist()} [PASS]")

    # Frame 1: Small displacement, high confidence
    bboxes_f1 = torch.tensor([
        [102.0, 101.0, 152.0, 181.0, 0.90],
        [301.0, 303.0, 351.0, 383.0, 0.88],
    ])
    labels_f1 = torch.tensor([1, 2])
    embeds_f1 = torch.nn.functional.normalize(embeds_f0 + 0.05 * torch.randn(2, 256), p=2, dim=1)
    cls_embeds_f1 = cls_embeds_f0

    b1, l1, ids1 = tracker.track(
        model=model,
        bboxes=bboxes_f1,
        labels=labels_f1,
        embeds=embeds_f1,
        cls_embeds=cls_embeds_f1,
        frame_id=1,
    )
    assert ids1[0] == 0 and ids1[1] == 1, f"Expected IDs [0, 1], got {ids1.tolist()}"
    print(f"  Frame 1: Continuous association preserved IDs {ids1.tolist()} [PASS]")

    # Frame 2: Ambiguous detection with explicit high uncertainty
    uncertainties_f2 = torch.tensor([0.9, 0.1])
    bboxes_f2 = torch.tensor([
        [104.0, 103.0, 154.0, 183.0, 0.40],
        [303.0, 305.0, 353.0, 385.0, 0.92],
    ])
    labels_f2 = torch.tensor([1, 2])
    embeds_f2 = embeds_f1
    cls_embeds_f2 = cls_embeds_f1

    b2, l2, ids2 = tracker.track(
        model=model,
        bboxes=bboxes_f2,
        labels=labels_f2,
        embeds=embeds_f2,
        cls_embeds=cls_embeds_f2,
        frame_id=2,
        uncertainties=uncertainties_f2,
    )
    assert ids2[0] == 0 and ids2[1] == 1, f"Expected IDs [0, 1], got {ids2.tolist()}"
    print(f"  Frame 2: High uncertainty handled correctly, IDs={ids2.tolist()} [PASS]")
    print("[TEST 3 PASSED]\n")


def run_all_tests():
    print("==================================================")
    print("RUNNING UA-KF-OVT RESEARCH SUITE VALIDATION TESTS")
    print("==================================================")
    test_uncertainty_estimator()
    test_adaptive_kalman()
    test_uakf_tracker_loop()
    print("ALL TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
