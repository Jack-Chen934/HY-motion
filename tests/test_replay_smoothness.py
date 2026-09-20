"""Regression checks for the kinematic replay command report.

Run after a smoke replay with:
    python -m pytest tests/test_replay_smoothness.py
"""

import json
from pathlib import Path


REPORT = Path(__file__).parents[1] / "reports/final_smoothness_full.json"


def test_replay_does_not_reuse_source_frames():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["source_frame_reuse_steps"] == 0


def test_root_command_step_is_bounded():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["max_commanded_root_step_m"] <= 0.025
    assert report["max_applied_root_step_m"] <= 0.03


def test_feet_do_not_sink_below_ground():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert max(report["max_foot_anchor_below_ground_by_foot_m"].values()) <= 0.005
