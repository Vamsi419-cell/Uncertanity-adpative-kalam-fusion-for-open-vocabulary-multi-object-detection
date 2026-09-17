# Changelog

All notable changes, bug fixes, audits, and research additions to the UA-KF-OVT codebase are documented here in reverse chronological order.

## [Unreleased] - 2026-09-17

### Initial Audit & Workspace Setup
- Cloned official OVT-B repository (`Coo1Sea/OVT-B-Dataset`) into workspace.
- Completed comprehensive 17-point repository audit across model entry points, RoI heads, Kalman filter, trackers, and evaluation pipeline.
- Audited baseline Kalman filter in `ovtrack/models/motions/kalman_filter.py` and traced coupling in `ovtrack/models/trackers/ovsort_tracker.py`.
- Identified critical information bottleneck in `OVTrackRoIHead`: category probability distribution $P(c|r)$ was discarded post-NMS.
- Established strict behavioral protections for original OVTrack and OVTrack+ baseline files.
- Formulated mathematical definitions for Normalized Shannon Entropy, Top-1/Top-2 Margin, Adaptive Measurement Covariance $R(u)$, Dynamic Association Weight $w(u)$, and Appearance Memory Gating.
- Initialized `RESEARCH_REPORT.md` and `implementation_plan.md`.
