"""Simulate and compare all ablations in the UA-KF-OVT experiment matrix.

Demonstrates the effect of:
1. EXP-0: Baseline OVTrack+ (fixed w=0.03, fixed R)
2. EXP-1: UA Association Only (dynamic w(u))
3. EXP-2: Adaptive Kalman Only (Linear R(u))
4. EXP-3: Adaptive Kalman Only (Exponential R(u))
5. EXP-4: Full UA-KF-OVT (R(u) + w(u))
6. EXP-5: Full UA-KF-OVT + Appearance Gating
"""

import sys
import os
import torch
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ovtrack.models.trackers.ovsort_tracker import OVSortTracker
from ovtrack.models.trackers.uakf_ovtracker import UAKFOVTracker
from ovtrack.models.motions.kalman_filter import KalmanFilter
from ovtrack.models.trackers.adaptive_kalman import AdaptiveKalmanFilter


def run_experiment(exp_name, tracker_cfg, motion_cfg, test_sequence):
    torch.manual_seed(42)
    np.random.seed(42)

    tracker = UAKFOVTracker(**tracker_cfg)
    
    class ModelWrapper:
        def __init__(self):
            if motion_cfg.get('type') == 'AdaptiveKalmanFilter':
                cfg = motion_cfg.copy()
                cfg.pop('type')
                self.motion = AdaptiveKalmanFilter(**cfg)
            else:
                self.motion = KalmanFilter()

    model = ModelWrapper()
    results = []

    for frame_id, frame_data in enumerate(test_sequence):
        bboxes = frame_data['bboxes'].clone()
        labels = frame_data['labels'].clone()
        embeds = frame_data['embeds'].clone()
        cls_embeds = frame_data['cls_embeds'].clone()
        uncertainties = frame_data.get('uncertainties', None)

        b, l, ids = tracker.track(
            model=model,
            bboxes=bboxes,
            labels=labels,
            embeds=embeds,
            cls_embeds=cls_embeds,
            frame_id=frame_id,
            uncertainties=uncertainties,
        )
        results.append({
            'frame_id': frame_id,
            'ids': ids.tolist(),
            'bboxes': b.cpu().numpy(),
            'num_tracks': len(tracker.tracks),
        })

    return results


def build_challenging_sequence():
    """Builds a sequence with an occluded frame and high semantic ambiguity."""
    frames = []
    # Ground truth: 2 objects moving horizontally
    # Frame 0: Both visible and confident
    f0_bboxes = torch.tensor([
        [100.0, 100.0, 150.0, 180.0, 0.95],
        [120.0, 100.0, 170.0, 180.0, 0.90],
    ])
    f0_embeds = torch.tensor([
        [1.0] * 128 + [0.0] * 128,
        [0.0] * 128 + [1.0] * 128,
    ], dtype=torch.float32)
    f0_embeds = torch.nn.functional.normalize(f0_embeds, p=2, dim=1)

    frames.append({
        'bboxes': f0_bboxes,
        'labels': torch.tensor([1, 2]),
        'embeds': f0_embeds,
        'cls_embeds': f0_embeds,
        'uncertainties': torch.tensor([0.05, 0.08]),
    })

    # Frame 1: Moving slightly
    f1_bboxes = torch.tensor([
        [105.0, 100.0, 155.0, 180.0, 0.92],
        [125.0, 100.0, 175.0, 180.0, 0.88],
    ])
    frames.append({
        'bboxes': f1_bboxes,
        'labels': torch.tensor([1, 2]),
        'embeds': f0_embeds + 0.02 * torch.randn(2, 256),
        'cls_embeds': f0_embeds,
        'uncertainties': torch.tensor([0.06, 0.10]),
    })

    # Frame 2: Crossing / Occlusion event — high semantic ambiguity (u=0.9)
    # The detector is confused and shifts positions slightly
    f2_bboxes = torch.tensor([
        [110.0, 100.0, 160.0, 180.0, 0.45],
        [130.0, 100.0, 180.0, 180.0, 0.40],
    ])
    # Ambiguous appearance during occlusion
    mixed_embed = (f0_embeds[0] + f0_embeds[1]) / 2.0
    f2_embeds = torch.stack([mixed_embed, mixed_embed], dim=0)
    frames.append({
        'bboxes': f2_bboxes,
        'labels': torch.tensor([1, 2]),
        'embeds': f2_embeds,
        'cls_embeds': f2_embeds,
        'uncertainties': torch.tensor([0.88, 0.92]),  # Highly uncertain!
    })

    # Frame 3: Disentangled objects reappearing
    f3_bboxes = torch.tensor([
        [115.0, 100.0, 165.0, 180.0, 0.91],
        [135.0, 100.0, 185.0, 180.0, 0.89],
    ])
    frames.append({
        'bboxes': f3_bboxes,
        'labels': torch.tensor([1, 2]),
        'embeds': f0_embeds,
        'cls_embeds': f0_embeds,
        'uncertainties': torch.tensor([0.08, 0.07]),
    })

    return frames


def run_ablation_suite():
    print("==================================================")
    print("UA-KF-OVT ABLATION MATRIX EMPIRICAL SIMULATION")
    print("==================================================")
    sequence = build_challenging_sequence()

    configs = {
        "EXP-0 (OVTrack+ Baseline)": {
            'tracker': dict(
                motion_weight=0.03,
                enable_adaptive_r=False,
                enable_adaptive_weight=False,
                enable_appearance_gating=False,
            ),
            'motion': dict(type='KalmanFilter'),
        },
        "EXP-1 (UA Association Only)": {
            'tracker': dict(
                motion_weight=0.03,
                motion_weight_max=0.35,
                enable_adaptive_r=False,
                enable_adaptive_weight=True,
                enable_appearance_gating=False,
            ),
            'motion': dict(type='KalmanFilter'),
        },
        "EXP-2 (Adaptive Kalman Linear Only)": {
            'tracker': dict(
                motion_weight=0.03,
                enable_adaptive_r=True,
                enable_adaptive_weight=False,
                enable_appearance_gating=False,
                kalman_mapping='linear',
                kalman_beta=3.0,
            ),
            'motion': dict(type='AdaptiveKalmanFilter', mapping='linear', beta=3.0),
        },
        "EXP-3 (Adaptive Kalman Exp Only)": {
            'tracker': dict(
                motion_weight=0.03,
                enable_adaptive_r=True,
                enable_adaptive_weight=False,
                enable_appearance_gating=False,
                kalman_mapping='exponential',
                kalman_beta=1.5,
            ),
            'motion': dict(type='AdaptiveKalmanFilter', mapping='exponential', beta=1.5),
        },
        "EXP-4 (Joint UA-KF: Dyn R + Dyn w)": {
            'tracker': dict(
                motion_weight=0.03,
                motion_weight_max=0.35,
                enable_adaptive_r=True,
                enable_adaptive_weight=True,
                enable_appearance_gating=False,
                kalman_mapping='linear',
                kalman_beta=3.0,
            ),
            'motion': dict(type='AdaptiveKalmanFilter', mapping='linear', beta=3.0),
        },
        "EXP-5 (Full UA-KF-OVT + App Gating)": {
            'tracker': dict(
                motion_weight=0.03,
                motion_weight_max=0.35,
                enable_adaptive_r=True,
                enable_adaptive_weight=True,
                enable_appearance_gating=True,
                appearance_gate_min=0.05,
                kalman_mapping='linear',
                kalman_beta=3.0,
            ),
            'motion': dict(type='AdaptiveKalmanFilter', mapping='linear', beta=3.0),
        },
    }

    summary = {}
    for exp_id, cfg in configs.items():
        res = run_experiment(exp_id, cfg['tracker'], cfg['motion'], sequence)
        final_ids = res[-1]['ids']
        f0_ids = res[0]['ids']
        id_switches = 0
        for f in range(1, len(res)):
            if res[f]['ids'] != res[f-1]['ids']:
                id_switches += 1
        summary[exp_id] = {
            'final_ids': final_ids,
            'id_consistency': (final_ids == f0_ids),
            'num_tracks': res[-1]['num_tracks'],
        }
        print(f"[{exp_id}]")
        print(f"  Frame 0 IDs: {f0_ids}")
        print(f"  Frame 3 IDs: {final_ids} | ID Consistent: {final_ids == f0_ids}")
        print(f"  Total Tracks Spawned: {res[-1]['num_tracks']}")
        print()

    print("==================================================")
    print("SUMMARY COMPARISON")
    print("==================================================")
    for exp_id, s in summary.items():
        status = "PRESERVED IDENTITY" if s['id_consistency'] else "ID SWITCH OCCURRED"
        print(f"{exp_id:<38} | {status} | Tracks: {s['num_tracks']}")


if __name__ == "__main__":
    run_ablation_suite()
