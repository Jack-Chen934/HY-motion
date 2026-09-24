"""Pure-data tests for the first isolated HRI task contract."""

import numpy as np

from generate_human_mjcf import BODY_NAMES
from run_hri_walking_approach import (
    GRASP_DISTANCE_TOLERANCE,
    GRASP_OFFSET_M,
    detect_pause_window,
    handover_schedule,
    interaction_target,
    interpolate_trajectory,
    load_trajectory_cache,
    robot_side_target,
    save_trajectory_cache,
    should_update_viewer,
    step_interval_for_rate,
    object_position_in_hand,
    object_position_in_palm,
    palm_pose_in_hand,
    PALM_GATE_BLEND,
)


def test_detect_pause_window_matches_current_motion_tail():
    data = np.load("motions/prepared/final_full_cooperation_seed2026.npz", allow_pickle=False)
    points = data["keypoints_hy"].astype(np.float32)
    conversion = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    points = points @ conversion.T
    start, end = detect_pause_window(points, 30.0, BODY_NAMES.index("left_wrist"), 0.12, 0.5)
    assert 3.8 <= start / 30.0 <= 4.0
    assert 4.9 <= end / 30.0 <= 5.0


def test_interaction_target_is_ten_centimeters_away_from_robot():
    hand = np.array((0.0, -2.82, 1.1), dtype=np.float32)
    robot = np.array((0.0, -2.95, 0.7), dtype=np.float32)
    target = interaction_target(hand, robot)
    assert np.isclose(np.linalg.norm(target - hand), 0.1, atol=1e-6)
    assert target[1] > hand[1]


def test_rate_limits_are_converted_to_stable_simulation_intervals():
    assert step_interval_for_rate(20.0, 0.01) == 5
    assert step_interval_for_rate(100.0, 0.01) == 1


def test_viewer_refresh_keeps_first_and_final_frames_visible():
    assert should_update_viewer(0, 5, 50)
    assert should_update_viewer(50, 5, 50)
    assert should_update_viewer(10, 5, 50)
    assert not should_update_viewer(11, 5, 50)


def test_trajectory_cache_roundtrip_and_interpolation(tmp_path):
    path = tmp_path / "episode.npz"
    times = [0.0, 1.0, 2.0]
    human = [np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([2.0, 4.0])]
    robot = [np.array([10.0]), np.array([20.0]), np.array([30.0])]
    save_trajectory_cache(path, times, human, robot, {"source_report": "report.json"})
    loaded_times, loaded_human, loaded_robot, metadata = load_trajectory_cache(path)
    assert np.allclose(loaded_times, times)
    assert np.allclose(interpolate_trajectory(loaded_times, loaded_human, 0.5), [0.5, 1.0])
    assert np.allclose(interpolate_trajectory(loaded_times, loaded_robot, 1.5), [25.0])
    assert metadata["source_report"] == "report.json"


def test_handover_schedule_has_ordered_safety_phases():
    schedule = handover_schedule(4.9666667, 3.9, 4.0)
    assert schedule["stable_start"] < schedule["pregrasp_end"]
    assert schedule["pregrasp_end"] < schedule["slow_end"]
    assert schedule["slow_end"] < schedule["grip_end"] < schedule["verify_end"]
    assert schedule["verify_end"] < schedule["retreat_end"] <= schedule["task_end"]


def test_handover_schedule_includes_post_motion_tail():
    schedule = handover_schedule(4.9666667, 3.9, 4.0)
    assert schedule["task_end"] > 35.0
    assert schedule["task_end"] < 38.0


def test_grasp_target_is_object_center_and_tolerance_is_positive():
    object_position = np.array((0.1, -2.8, 1.1), dtype=np.float32)
    robot_position = np.array((-0.5, -2.8, 0.0), dtype=np.float32)
    assert np.allclose(robot_side_target(object_position, robot_position, GRASP_OFFSET_M), object_position)
    assert GRASP_DISTANCE_TOLERANCE > 0.0


def test_object_is_offset_from_wrist_toward_the_palm_side():
    points = np.zeros((3, 3), dtype=np.float32)
    points[0] = (0.0, 0.0, 0.0)
    points[1] = (0.0, 0.0, 1.0)
    points[2] = (0.0, 0.0, 2.0)
    result = object_position_in_hand(points, hand_index=2, parent_index=1, forward_offset=0.08)
    assert np.allclose(result, (0.0, 0.0, 2.08))


def test_object_is_placed_between_wrist_and_wooden_palm_roots():
    points = np.zeros((35, 3), dtype=np.float32)
    points[20] = (0.0, 0.0, 1.0)
    for index, position in zip((22, 25, 28, 31, 34), ((0.2, 0.0, 1.0), (0.4, 0.0, 1.0),
                                                       (0.6, 0.0, 1.0), (0.8, 0.0, 1.0),
                                                       (1.0, 0.0, 1.0))):
        points[index] = position
    result = object_position_in_palm(points, palm_fraction=0.5)
    assert np.allclose(result, (0.3, 0.0, 1.0))


def test_palm_pose_is_inside_thumb_index_gate_and_orthonormal():
    points = np.zeros((35, 3), dtype=np.float32)
    points[20] = (0.0, 0.0, 0.0)  # wrist
    points[22] = (0.08, 0.10, 0.0)  # index root
    points[25] = (0.02, 0.13, 0.0)  # middle root
    points[28] = (-0.06, 0.11, 0.0)  # pinky root
    points[31] = (-0.02, 0.12, 0.0)  # ring root
    points[34] = (0.04, 0.04, 0.0)  # thumb root
    position, quat, diagnostics = palm_pose_in_hand(points)
    gate = 0.5 * (points[22] + points[34])
    palm = np.mean(points[[22, 25, 28, 31]], axis=0)
    assert np.allclose(position, (1.0 - PALM_GATE_BLEND) * gate + PALM_GATE_BLEND * palm)
    assert diagnostics["object_to_index_m"] > 0.0
    assert diagnostics["object_to_thumb_m"] > 0.0
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
