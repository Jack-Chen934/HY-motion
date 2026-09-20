"""Regression checks for MJCF keypoint-to-ball-joint retargeting."""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Genesis 1.3.3 is installed in a read-only environment on this server and
# otherwise attempts to cache Numba functions beside its package files.
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hy_genesis_numba")

from generate_human_mjcf import BODY_NAMES, PARENTS, rotation_from_basis
from replay_human_mjcf import prepare_points, qpos_for_frame


MOTION = Path(__file__).parents[1] / "motions/prepared/final_full_cooperation_seed2026.npz"
LEG_BODIES = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
)


@dataclass
class _Joint:
    q_start: int


class _FakeHuman:
    """Only the qpos layout needed by qpos_for_frame."""

    n_qs = 91

    @staticmethod
    def get_joint(name: str) -> _Joint:
        if name == "pelvis_free":
            return _Joint(0)
        body_name = name.removesuffix("_ball")
        body_index = BODY_NAMES.index(body_name)
        return _Joint(7 + 4 * (body_index - 1))


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()


def _reconstruct_positions(
    qpos: np.ndarray, points: np.ndarray, rest: np.ndarray, rest_basis: np.ndarray
) -> np.ndarray:
    positions = np.zeros_like(points)
    rotations: list[np.ndarray | None] = [None] * len(BODY_NAMES)
    positions[0] = qpos[:3]
    rotations[0] = _rotation_from_wxyz(qpos[3:7])

    for child in range(1, len(BODY_NAMES)):
        parent = int(PARENTS[child])
        parent_rotation = rotations[parent]
        assert parent_rotation is not None
        offset = rest_basis.T @ (rest[child] - rest[parent])
        positions[child] = positions[parent] + parent_rotation @ offset
        child_q_start = 7 + 4 * (child - 1)
        local_rotation = _rotation_from_wxyz(qpos[child_q_start : child_q_start + 4])
        rotations[child] = parent_rotation @ local_rotation
    return positions


def test_leg_bone_directions_follow_keypoints_without_one_level_lag():
    points, _ = prepare_points(MOTION)
    rest = points[0]
    rest_basis = rotation_from_basis(rest)
    fake_human = _FakeHuman()
    max_errors = {body_name: 0.0 for body_name in LEG_BODIES}

    for frame_points in points:
        qpos = qpos_for_frame(frame_points, rest, rest_basis, fake_human)
        reconstructed = _reconstruct_positions(qpos, frame_points, rest, rest_basis)
        for body_name in LEG_BODIES:
            body_index = BODY_NAMES.index(body_name)
            parent_index = int(PARENTS[body_index])
            expected_bone = frame_points[body_index] - frame_points[parent_index]
            actual_bone = reconstructed[body_index] - reconstructed[parent_index]
            error = float(np.linalg.norm(actual_bone - expected_bone))
            max_errors[body_name] = max(max_errors[body_name], error)

    assert max(max_errors.values()) < 1e-5, max_errors
