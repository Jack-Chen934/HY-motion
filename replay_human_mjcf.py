"""Replay HY-Motion keypoints through a 22-joint MJCF ball-joint human in Genesis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import RotationSpline

from generate_human_mjcf import BODY_NAMES, PARENTS, HY_TO_GENESIS, rotation_from_basis


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
DEFAULT_MODEL = WORKSPACE / "genesis-human-experiment/assets/human_22ball.xml"
DEFAULT_REPORT = WORKSPACE / "genesis-human-experiment/reports/final_smoothness_full.json"
SIM_DT = 0.01
FOOT_NAMES = ("left_foot", "right_foot")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{segment}_ball"
    for leg in ("left", "right")
    for segment in ("hip", "knee", "ankle")
)
FOOT_GEOM_RADIUS_M = 0.045
MAX_LEG_FRAME_STEP_RAD = 0.08
CONTACT_CONFIRM_FRAMES = 2
CONTACT_RELEASE_CONFIRM_FRAMES = 3
CONTACT_RAMP_SECONDS = 0.12
CONTACT_ENTER_SPEED_MPS = 0.15
CONTACT_RELEASE_SPEED_MPS = 0.25
MAX_ROOT_CORRECTION_STEP_M = 0.002
MAX_CONTACT_SWITCH_CORRECTION_STEP_M = 0.002
MAX_SUPPORT_JOINT_STEP_RAD = 0.025
MAX_SUPPORT_TOTAL_JOINT_STEP_RAD = 0.04
ROOT_CORRECTION_RELEASE_STEP_M = 0.001
MAX_GROUND_CLAMP_STEP_M = 0.02
MAX_ACCEPTABLE_SUPPORT_IK_ERROR_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Keep the viewer camera fixed at the initial world position instead of following the human.",
    )
    parser.add_argument("--collision", action="store_true")
    parser.add_argument(
        "--foot-lock",
        action="store_true",
        help="Lock detected foot contacts in XY and keep the foot anchor above the ground.",
    )
    parser.add_argument(
        "--foot-contact-height",
        type=float,
        default=0.055,
        help="Raw foot-anchor height below which a contact can start (meters).",
    )
    parser.add_argument(
        "--foot-clearance",
        type=float,
        default=FOOT_GEOM_RADIUS_M + 0.01,
        help="Target foot-anchor height above the ground during contact (meters).",
    )
    parser.add_argument(
        "--foot-release-height",
        type=float,
        default=0.085,
        help="A locked foot is released only above this raw height (meters).",
    )
    parser.add_argument(
        "--leg-ik",
        action="store_true",
        help="Use Genesis multi-link IK on hip/knee/ankle joints for locked feet.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("cubic", "linear"),
        default="cubic",
        help="Interpolation from source FPS to simulation FPS (default: cubic).",
    )
    parser.add_argument(
        "--ik-weight",
        type=float,
        default=0.2,
        help="Fraction of each support/swing IK correction applied per step (0..1).",
    )
    return parser.parse_args()


def quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-8)
    target = target / max(np.linalg.norm(target), 1e-8)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -1.0 + 1e-7:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, source))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)
    axis = np.cross(source, target)
    quaternion = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(np.float32)


def slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two normalized wxyz quaternions along the shortest arc."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 1.0 - 1e-8:
        result = first + float(alpha) * (second - first)
        return (result / max(np.linalg.norm(result), 1e-12)).astype(np.float32)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    first_weight = np.sin((1.0 - alpha) * theta) / sin_theta
    second_weight = np.sin(alpha * theta) / sin_theta
    return (first_weight * first + second_weight * second).astype(np.float32)


def interpolate_qpos(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate a Genesis free-root + ball-joint qpos continuously."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    result = np.empty_like(first, dtype=np.float32)
    result[:3] = (1.0 - alpha) * first[:3] + alpha * second[:3]
    result[3:7] = slerp_wxyz(first[3:7], second[3:7], alpha)
    for start in range(7, len(result), 4):
        result[start : start + 4] = slerp_wxyz(
            first[start : start + 4], second[start : start + 4], alpha
        )
    return result


def build_cubic_qpos_interpolator(qposes: np.ndarray, fps: float):
    """Build a C2-continuous interpolator for root positions and quaternions."""
    qposes = make_quaternion_sequence_continuous(qposes)
    times = np.arange(len(qposes), dtype=np.float64) / float(fps)
    root_spline = CubicSpline(times, qposes[:, :3], axis=0)
    quaternion_starts = [3] + list(range(7, qposes.shape[1], 4))
    rotation_splines = []
    for start in quaternion_starts:
        values = qposes[:, start : start + 4]
        rotations = Rotation.from_quat(values[:, [1, 2, 3, 0]])
        rotation_splines.append(RotationSpline(times, rotations))

    def sample(time_seconds: float) -> np.ndarray:
        time_seconds = float(np.clip(time_seconds, times[0], times[-1]))
        result = np.empty(qposes.shape[1], dtype=np.float32)
        result[:3] = np.asarray(root_spline(time_seconds), dtype=np.float32)
        for start, spline in zip(quaternion_starts, rotation_splines):
            xyzw = np.asarray(spline(time_seconds).as_quat(), dtype=np.float32)
            result[start : start + 4] = xyzw[[3, 0, 1, 2]]
        return result

    return sample


def interpolate_points(points: np.ndarray, source_position: float) -> tuple[np.ndarray, int]:
    """Linearly interpolate keypoints at a fractional source-frame position."""
    left = int(np.floor(source_position))
    left = min(max(left, 0), len(points) - 1)
    right = min(left + 1, len(points) - 1)
    alpha = float(source_position - left) if right != left else 0.0
    return ((1.0 - alpha) * points[left] + alpha * points[right]).astype(np.float32), left


def make_quaternion_sequence_continuous(qposes: np.ndarray) -> np.ndarray:
    """Remove q/-q sign flips before temporal interpolation."""
    result = np.asarray(qposes, dtype=np.float32).copy()
    quaternion_starts = [3] + list(range(7, result.shape[1], 4))
    for frame in range(1, len(result)):
        for start in quaternion_starts:
            if float(np.dot(result[frame - 1, start : start + 4], result[frame, start : start + 4])) < 0.0:
                result[frame, start : start + 4] *= -1.0
    return result


def vector_motion_metrics(values: np.ndarray, dt: float) -> dict[str, float]:
    """Summarize velocity, acceleration, and jerk for vector samples."""
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) < 2:
        return {
            "max_velocity": 0.0,
            "max_acceleration": 0.0,
            "max_jerk": 0.0,
        }
    velocity = np.diff(samples, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) >= 2 else np.empty((0, samples.shape[1]))
    jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) >= 2 else np.empty((0, samples.shape[1]))

    def maximum(array: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(array, axis=1))) if len(array) else 0.0

    return {
        "max_velocity": maximum(velocity),
        "max_acceleration": maximum(acceleration),
        "max_jerk": maximum(jerk),
    }


def quaternion_motion_metrics(qposes: np.ndarray, joint_starts: list[int], dt: float) -> dict[str, float]:
    """Summarize angular velocity, acceleration, and jerk for qpos joints."""
    samples = np.asarray(qposes, dtype=np.float64)
    if len(samples) < 2:
        return {
            "max_joint_velocity_rad_s": 0.0,
            "max_joint_acceleration_rad_s2": 0.0,
            "max_joint_jerk_rad_s3": 0.0,
        }
    angular_steps = np.zeros((len(samples) - 1, len(joint_starts)), dtype=np.float64)
    for frame in range(1, len(samples)):
        for joint_index, start in enumerate(joint_starts):
            previous = samples[frame - 1, start : start + 4]
            current = samples[frame, start : start + 4]
            previous_rotation = Rotation.from_quat(
                [previous[1], previous[2], previous[3], previous[0]]
            )
            current_rotation = Rotation.from_quat(
                [current[1], current[2], current[3], current[0]]
            )
            angular_steps[frame - 1, joint_index] = (
                previous_rotation.inv() * current_rotation
            ).magnitude()
    velocity = angular_steps / dt
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) >= 2 else np.empty((0, len(joint_starts)))
    jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) >= 2 else np.empty((0, len(joint_starts)))

    def maximum(array: np.ndarray) -> float:
        return float(np.max(array)) if array.size else 0.0

    return {
        "max_joint_velocity_rad_s": maximum(velocity),
        "max_joint_acceleration_rad_s2": maximum(acceleration),
        "max_joint_jerk_rad_s3": maximum(jerk),
    }


def prepare_points(motion_path: Path) -> tuple[np.ndarray, float]:
    data = np.load(motion_path, allow_pickle=False)
    points_hy = data["keypoints_hy"].astype(np.float32)
    points = points_hy @ HY_TO_GENESIS.T
    ground_lift = max(0.0, 0.02 - float(points[..., 2].min()))
    points[..., 2] += ground_lift
    return points[:, : len(BODY_NAMES)], float(data["fps"][0])


def _foot_index(name: str) -> int:
    return BODY_NAMES.index(name)


def apply_foot_lock(
    points: np.ndarray,
    fps: float,
    contact_height: float,
    clearance: float,
    release_height: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply a translation-only foot lock to a keypoint sequence.

    This is deliberately a kinematic preprocessing step. It does not solve IK or
    generate contact forces; it only modifies the root translation so the
    retargeted skeleton has a stable support point during detected contacts.
    """
    if release_height < contact_height:
        raise ValueError("foot release height must be >= foot contact height")
    if clearance < 0.0:
        raise ValueError("foot clearance must be non-negative")

    foot_indices = np.array([_foot_index(name) for name in FOOT_NAMES], dtype=np.int32)
    raw = points.copy()
    locked = np.zeros(2, dtype=bool)
    enter_counts = np.zeros(2, dtype=np.int32)
    release_counts = np.zeros(2, dtype=np.int32)
    lock_xy = np.zeros((2, 2), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    adjusted = np.empty_like(raw)
    contact_mask = np.zeros((len(raw), 2), dtype=bool)
    translations = np.zeros((len(raw), 3), dtype=np.float32)

    for frame, frame_points in enumerate(raw):
        raw_foot = frame_points[foot_indices]
        for foot in range(2):
            z = float(raw_foot[foot, 2])
            horizontal_speed = 0.0
            if frame > 0:
                horizontal_speed = float(
                    np.linalg.norm(raw_foot[foot, :2] - raw[frame - 1, foot_indices[foot], :2])
                    * fps
                )
            if locked[foot]:
                if z > release_height or horizontal_speed > CONTACT_RELEASE_SPEED_MPS:
                    release_counts[foot] += 1
                    if release_counts[foot] >= CONTACT_RELEASE_CONFIRM_FRAMES:
                        locked[foot] = False
                        release_counts[foot] = 0
                else:
                    release_counts[foot] = 0
                enter_counts[foot] = 0
            elif z <= contact_height and horizontal_speed <= CONTACT_ENTER_SPEED_MPS:
                enter_counts[foot] += 1
                if enter_counts[foot] >= CONTACT_CONFIRM_FRAMES:
                    locked[foot] = True
                    lock_xy[foot] = raw_foot[foot, :2] + translation[:2]
                    enter_counts[foot] = 0
            else:
                enter_counts[foot] = 0
                release_counts[foot] = 0

        active = np.flatnonzero(locked)
        if len(active) > 0:
            xy_constraints = lock_xy[active] - raw_foot[active, :2]
            z_constraints = clearance - raw_foot[active, 2]
            desired = np.array(
                [*np.mean(xy_constraints, axis=0), float(np.mean(z_constraints))],
                dtype=np.float32,
            )
        else:
            # Return smoothly to the source trajectory after the last support foot
            # leaves the ground, avoiding a visible root teleport.
            desired = np.zeros(3, dtype=np.float32)

        delta = desired - translation
        xy_norm = float(np.linalg.norm(delta[:2]))
        if xy_norm > 0.035:
            delta[:2] *= 0.035 / xy_norm
        delta[2] = float(np.clip(delta[2], -0.025, 0.025))
        translation += delta

        adjusted[frame] = frame_points + translation
        translations[frame] = translation
        contact_mask[frame] = locked

    return adjusted, {"contact_mask": contact_mask, "translations": translations}


def as_position(value: object) -> np.ndarray:
    return np.asarray(value).reshape(-1)[:3].astype(np.float32)


def joint_position(human, body_name: str) -> np.ndarray:
    if body_name == "pelvis":
        return as_position(human.get_link(name="pelvis").get_pos())
    return as_position(human.get_joint(name=f"{body_name}_ball").get_anchor_pos())


def skeleton_bone_lengths(human) -> np.ndarray:
    """Return all fixed parent-child anchor distances in BODY_NAMES order."""
    positions = [joint_position(human, name) for name in BODY_NAMES]
    return np.asarray(
        [np.linalg.norm(positions[child] - positions[int(PARENTS[child])]) for child in range(1, len(BODY_NAMES))],
        dtype=np.float32,
    )


def set_ball_joint_delta(qpos: np.ndarray, joint, axis: np.ndarray, angle: float) -> None:
    """Apply a small parent-frame rotation to a ball-joint quaternion in qpos."""
    current = qpos[joint.q_start : joint.q_start + 4]
    current_rotation = Rotation.from_quat([current[1], current[2], current[3], current[0]])
    delta_rotation = Rotation.from_rotvec(axis * angle)
    updated = delta_rotation * current_rotation
    qpos[joint.q_start : joint.q_start + 4] = quat_wxyz(updated.as_matrix())


def limit_ball_joint_change(
    qpos: np.ndarray,
    reference_qpos: np.ndarray,
    joint,
    max_angle: float,
) -> None:
    """Limit a ball joint's total change from the beginning of this frame."""
    reference = reference_qpos[joint.q_start : joint.q_start + 4]
    current = qpos[joint.q_start : joint.q_start + 4]
    reference_rotation = Rotation.from_quat(
        [reference[1], reference[2], reference[3], reference[0]]
    )
    current_rotation = Rotation.from_quat(
        [current[1], current[2], current[3], current[0]]
    )
    relative = reference_rotation.inv() * current_rotation
    angle = float(relative.magnitude())
    if angle <= max_angle:
        return
    limited = reference_rotation * Rotation.from_rotvec(
        relative.as_rotvec() * (max_angle / angle)
    )
    qpos[joint.q_start : joint.q_start + 4] = quat_wxyz(limited.as_matrix())


def custom_leg_ik(
    human,
    qpos: np.ndarray,
    active_contacts: list[tuple[int, str]],
    targets: dict[str, np.ndarray],
    iterations: int = 6,
    damping: float = 0.04,
    finite_difference_step: float = 1e-3,
    max_joint_step: float = 0.08,
    max_total_joint_step: float = 0.12,
    include_root: bool = False,
    max_root_step: float = 0.02,
    root_regularization: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Numerical damped-least-squares IK for the ball-joint human legs."""
    active_legs = [foot_name.split("_")[0] for _, foot_name in active_contacts]
    joints = []
    for leg in active_legs:
        joints.extend(
            human.get_joint(name=f"{leg}_{segment}_ball")
            for segment in ("hip", "knee", "ankle")
        )
    variables = []
    regularization = []
    if include_root:
        variables.extend((None, axis) for axis in np.eye(3, dtype=np.float32))
        regularization.extend([root_regularization] * 3)
    variables.extend((joint, axis) for joint in joints for axis in np.eye(3, dtype=np.float32))
    regularization.extend([damping] * (len(variables) - len(regularization)))
    target_vector = np.concatenate([targets[name] for _, name in active_contacts]).astype(np.float32)
    initial_qpos = qpos.copy()
    free_start = human.get_joint(name="pelvis_free").q_start

    for _ in range(iterations):
        human.set_qpos(qpos, zero_velocity=True)
        current_vector = np.concatenate(
            [joint_position(human, name) for _, name in active_contacts]
        ).astype(np.float32)
        error = target_vector - current_vector
        if float(np.max(np.abs(error))) < 1e-4:
            break

        jacobian = np.zeros((len(target_vector), len(variables)), dtype=np.float32)
        for column, (joint, axis) in enumerate(variables):
            trial_qpos = qpos.copy()
            if joint is None:
                trial_qpos[human.get_joint(name="pelvis_free").q_start : human.get_joint(name="pelvis_free").q_start + 3] += (
                    axis * finite_difference_step
                )
            else:
                set_ball_joint_delta(trial_qpos, joint, axis, finite_difference_step)
            human.set_qpos(trial_qpos, zero_velocity=True)
            trial_vector = np.concatenate(
                [joint_position(human, name) for _, name in active_contacts]
            ).astype(np.float32)
            jacobian[:, column] = (trial_vector - current_vector) / finite_difference_step

        regularized = jacobian.T @ jacobian + np.diag(
            np.square(np.asarray(regularization, dtype=np.float32))
        )
        delta = np.linalg.solve(regularized, jacobian.T @ error)
        delta_offset = 0
        if include_root:
            root_delta = delta[:3]
            root_norm = float(np.linalg.norm(root_delta))
            if root_norm > max_root_step:
                root_delta *= max_root_step / max(root_norm, 1e-8)
            qpos[free_start : free_start + 3] += root_delta
            root_offset = qpos[free_start : free_start + 3] - initial_qpos[free_start : free_start + 3]
            root_offset_norm = float(np.linalg.norm(root_offset))
            if root_offset_norm > max_root_step:
                qpos[free_start : free_start + 3] = initial_qpos[
                    free_start : free_start + 3
                ] + root_offset * (max_root_step / max(root_offset_norm, 1e-8))
            delta_offset = 3
        for joint_index, joint in enumerate(joints):
            joint_delta = delta[delta_offset + joint_index * 3 : delta_offset + joint_index * 3 + 3]
            norm = float(np.linalg.norm(joint_delta))
            if norm > max_joint_step:
                joint_delta *= max_joint_step / norm
            angle = float(np.linalg.norm(joint_delta))
            if angle > 1e-8:
                set_ball_joint_delta(qpos, joint, joint_delta / angle, angle)
            limit_ball_joint_change(qpos, initial_qpos, joint, max_total_joint_step)

    human.set_qpos(qpos, zero_velocity=True)
    final_vector = np.concatenate(
        [joint_position(human, name) for _, name in active_contacts]
    ).astype(np.float32)
    final_error = float(np.max(np.abs(target_vector - final_vector)))
    return qpos, final_error


def qpos_for_frame(points: np.ndarray, rest: np.ndarray, rest_basis: np.ndarray, human) -> np.ndarray:
    root_target_basis = rotation_from_basis(points)
    root_q = quat_wxyz(root_target_basis @ rest_basis.T)
    global_rotations = [root_target_basis]
    qpos = np.zeros(human.n_qs, dtype=np.float32)
    free = human.get_joint(name="pelvis_free")
    qpos[free.q_start : free.q_start + 7] = np.concatenate([points[0], root_q])
    for child in range(1, len(BODY_NAMES)):
        parent = int(PARENTS[child])
        rest_vector = rest_basis.T @ (rest[child] - rest[parent])
        target_vector = global_rotations[parent].T @ (points[child] - points[parent])
        local_q = quat_from_two_vectors(rest_vector, target_vector)
        local_rotation = Rotation.from_quat([local_q[1], local_q[2], local_q[3], local_q[0]]).as_matrix()
        global_rotations.append(global_rotations[parent] @ local_rotation)
        joint = human.get_joint(name=f"{BODY_NAMES[child]}_ball")
        qpos[joint.q_start : joint.q_start + 4] = local_q
    return qpos


def run(
    motion_path: Path,
    model_path: Path,
    report_path: Path,
    seconds: float | None,
    show_viewer: bool,
    fixed_camera: bool,
    collision: bool,
    foot_lock: bool,
    foot_contact_height: float,
    foot_clearance: float,
    foot_release_height: float,
    leg_ik: bool,
    interpolation: str,
    ik_weight: float,
) -> None:
    if leg_ik and not foot_lock:
        raise ValueError("--leg-ik requires --foot-lock")
    if not 0.0 <= ik_weight <= 1.0:
        raise ValueError("--ik-weight must be between 0 and 1")
    points, fps = prepare_points(motion_path.resolve())
    contact_info = {
        "contact_mask": np.zeros((len(points), 2), dtype=bool),
        "translations": np.zeros((len(points), 3), dtype=np.float32),
    }
    if foot_lock:
        points, contact_info = apply_foot_lock(
            points,
            fps=fps,
            contact_height=foot_contact_height,
            clearance=foot_clearance,
            release_height=foot_release_height,
        )
    rest = points[0]
    rest_basis = rotation_from_basis(rest)
    requested_duration = (len(points) - 1) / fps if seconds is None else min(seconds, (len(points) - 1) / fps)
    steps = int(np.ceil(requested_duration / SIM_DT))

    gs.init(backend=gs.cpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=collision, box_box_detection=True),
        viewer_options=gs.options.ViewerOptions(
            res=(1100, 700),
            camera_pos=(3.4, -3.4, 2.2),
            camera_lookat=(0.0, 0.0, 0.95),
            camera_fov=50,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(model_path.resolve()), requires_jac_and_IK=False),
        name="hymotion_human_22ball",
    )
    scene.build()
    camera_follow_enabled = bool(show_viewer and not fixed_camera)
    if camera_follow_enabled:
        # The clip travels about 2.2 m along -Y. Follow the human root so the
        # full walk remains visible instead of leaving the fixed origin view.
        scene.viewer.follow_entity(
            human,
            fixed_axis=(None, None, None),
            smoothing=0.92,
            fix_orientation=False,
        )

    # Retarget each source frame once, then interpolate at the simulation rate.
    # This removes the visible 30 FPS -> 100 Hz frame plateaus.
    source_qposes = np.stack(
        [qpos_for_frame(frame_points, rest, rest_basis, human) for frame_points in points]
    ).astype(np.float32)
    source_qposes = make_quaternion_sequence_continuous(source_qposes)
    cubic_qpos_at = (
        build_cubic_qpos_interpolator(source_qposes, fps)
        if interpolation == "cubic"
        else None
    )
    free_start = human.get_joint(name="pelvis_free").q_start
    source_ground_offset = 0.0
    for source_qpos in source_qposes:
        human.set_qpos(source_qpos, zero_velocity=True)
        source_foot_heights = [
            float(joint_position(human, foot_name)[2]) for foot_name in FOOT_NAMES
        ]
        source_ground_offset = max(
            source_ground_offset,
            foot_clearance - min(source_foot_heights),
        )
    source_ground_offset = max(0.0, float(source_ground_offset))
    source_qposes[:, free_start + 2] += source_ground_offset
    source_min_foot_height = float("inf")
    for source_qpos in source_qposes:
        human.set_qpos(source_qpos, zero_velocity=True)
        source_min_foot_height = min(
            source_min_foot_height,
            min(float(joint_position(human, foot_name)[2]) for foot_name in FOOT_NAMES),
        )

    max_root_error = 0.0
    max_joint_error = 0.0
    max_joint_error_name = None
    max_joint_error_frame = None
    max_foot_penetration = 0.0
    max_foot_anchor_below_ground = 0.0
    max_foot_anchor_below_ground_by_foot = {name: 0.0 for name in FOOT_NAMES}
    max_foot_anchor_below_ground_frame = {name: None for name in FOOT_NAMES}
    max_foot_anchor_below_ground_context = {name: None for name in FOOT_NAMES}
    max_foot_height_error = 0.0
    foot_contact_counts = {name: 0 for name in FOOT_NAMES}
    foot_sliding_distance = {name: 0.0 for name in FOOT_NAMES}
    previous_foot_positions = {name: None for name in FOOT_NAMES}
    locked_foot_xy = {name: None for name in FOOT_NAMES}
    max_root_correction = 0.0
    max_ground_clamp = 0.0
    max_ik_position_error = 0.0
    max_ik_position_error_frame = None
    max_ik_position_error_foot = None
    max_ik_qpos_delta = 0.0
    support_ik_rejections = 0
    support_ik_rejection_frames = []
    max_leg_joint_step_rad = 0.0
    max_leg_joint_step_frame = None
    max_leg_joint_step_joint = None
    previous_applied_leg_qpos = None
    previous_active_leg_names = set()
    previous_runtime_contact_names = set()
    runtime_contact_disabled = {name: False for name in FOOT_NAMES}
    contact_ramp = {name: 0.0 for name in FOOT_NAMES}
    expected_bone_lengths = np.linalg.norm(
        rest[1:] - rest[PARENTS[1:]], axis=1
    ).astype(np.float32)
    max_bone_length_error = 0.0
    max_post_step_bone_length_error = 0.0
    max_post_step_qpos_drift = 0.0
    previous_applied_qpos = None
    ik_frames = 0
    source_frame_reuse_steps = 0
    max_commanded_root_step_m = 0.0
    previous_commanded_root = None
    previous_applied_root = None
    max_applied_root_step_m = 0.0
    max_root_correction_step = 0.0
    max_root_correction_state = 0.0
    root_correction_state = np.zeros(3, dtype=np.float32)
    ground_lift_state = 0.0
    commanded_qpos_history = []
    applied_qpos_history = []
    commanded_root_history = []
    applied_root_history = []
    foot_position_history = []
    root_correction_history = []
    contact_transition_frames = []
    pre_post_ik_jump_metrics = []
    previous_metric_contact_names = set()
    previous_interpolated_qpos = None
    started = time.perf_counter()
    for step in range(steps + 1):
        source_position = min(step * SIM_DT, requested_duration) * fps
        frame_points, frame = interpolate_points(points, source_position)
        left_frame = min(int(np.floor(source_position)), len(points) - 1)
        right_frame = min(left_frame + 1, len(points) - 1)
        interpolation_alpha = source_position - left_frame if right_frame != left_frame else 0.0
        if cubic_qpos_at is not None:
            qpos = cubic_qpos_at(min(step * SIM_DT, requested_duration))
        else:
            qpos = interpolate_qpos(
                source_qposes[left_frame], source_qposes[right_frame], interpolation_alpha
            )
        if previous_interpolated_qpos is not None and np.array_equal(
            qpos, previous_interpolated_qpos
        ):
            source_frame_reuse_steps += 1
        previous_interpolated_qpos = qpos.copy()
        commanded_qpos_history.append(qpos.copy())
        commanded_root = qpos[human.get_joint(name="pelvis_free").q_start :]
        commanded_root = commanded_root[:3].copy()
        commanded_root_history.append(commanded_root.copy())
        if previous_commanded_root is not None:
            max_commanded_root_step_m = max(
                max_commanded_root_step_m,
                float(np.linalg.norm(commanded_root - previous_commanded_root)),
            )
        previous_commanded_root = commanded_root
        frame_points = frame_points.copy()
        frame_points[:, 2] += source_ground_offset
        # Preserve only the previous IK solution for the leg ball joints.  The
        # rest of qpos must still come from this frame so that the torso and
        # root follow the motion instead of freezing at the previous frame.
        # Likewise, preserve only legs that are currently supporting the body;
        # the swing leg must continue following the source motion.
        active_leg_names = {
            FOOT_NAMES[foot_index].split("_")[0]
            for foot_index in range(2)
            if foot_lock
            and leg_ik
            and contact_info["contact_mask"][frame, foot_index]
        }
        if previous_applied_leg_qpos is not None:
            for joint_name in LEG_JOINT_NAMES:
                joint = human.get_joint(name=joint_name)
                leg_name = joint_name.split("_")[0]
                if leg_name in active_leg_names and leg_name in previous_active_leg_names:
                    qpos[joint.q_start : joint.q_start + 4] = slerp_wxyz(
                        previous_applied_leg_qpos[joint.q_start : joint.q_start + 4],
                        qpos[joint.q_start : joint.q_start + 4],
                        0.15,
                    )
                else:
                    # Retargeting with two-vector quaternions can select a
                    # different equivalent branch at a frame boundary. Keep
                    # swing legs responsive while carrying their previous
                    # solution forward, so clearance IK does not restart from
                    # a different source branch every simulation step.
                    qpos[joint.q_start : joint.q_start + 4] = slerp_wxyz(
                        previous_applied_leg_qpos[joint.q_start : joint.q_start + 4],
                        qpos[joint.q_start : joint.q_start + 4],
                        0.35,
                    )
                    limit_ball_joint_change(
                        qpos, previous_applied_leg_qpos, joint, max_angle=MAX_LEG_FRAME_STEP_RAD
                    )
        # Root correction is stateful. Without this, the foot constraint is
        # recomputed from a fresh source pose every step and can never converge.
        qpos[free_start : free_start + 3] += root_correction_state
        qpos[free_start + 2] += ground_lift_state
        human.set_qpos(qpos, zero_velocity=True)
        correction = np.zeros(3, dtype=np.float32)
        ik_targets = {}
        solved_leg_names = set()
        accepted_ik_target_names = set()

        # Correct the free-root translation from the actual Genesis anchors. The
        # input keypoints and the fixed-length skeleton are not identical, so a
        # keypoint-only lock can still leave the collision spheres underground.
        active_contacts = []
        ik_contacts = []
        if foot_lock:
            for foot_index, foot_name in enumerate(FOOT_NAMES):
                if contact_info["contact_mask"][frame, foot_index]:
                    if runtime_contact_disabled[foot_name]:
                        continue
                    contact_ramp[foot_name] = min(
                        1.0, contact_ramp[foot_name] + SIM_DT / CONTACT_RAMP_SECONDS
                    )
                    active_contacts.append((foot_index, foot_name))
                    actual = joint_position(human, foot_name)
                    if locked_foot_xy[foot_name] is None:
                        locked_foot_xy[foot_name] = actual[:2].copy()
                    anchor = np.array(
                        [locked_foot_xy[foot_name][0], locked_foot_xy[foot_name][1], foot_clearance],
                        dtype=np.float32,
                    )
                    blend = contact_ramp[foot_name]
                    ik_targets[foot_name] = actual + blend * (anchor - actual)
                else:
                    runtime_contact_disabled[foot_name] = False
                    locked_foot_xy[foot_name] = None
                    contact_ramp[foot_name] = max(
                        0.0, contact_ramp[foot_name] - SIM_DT / CONTACT_RAMP_SECONDS
                    )
            # Only a foot detected as being in contact is an IK constraint.
            # Swing-foot lift is deliberately handled by the source motion and
            # root ground clamp; constraining it here causes discontinuities.
            if len(active_contacts) > 1:
                persistent_contacts = [
                    contact
                    for contact in active_contacts
                    if contact[1] in previous_runtime_contact_names
                ]
                # Prefer the foot that was already supporting the body. The
                # newly entering foot remains collision-safe but is not added
                # to the hard IK closure until the old support is released.
                ik_contacts = persistent_contacts[:1] or active_contacts[:1]
            else:
                ik_contacts = list(active_contacts)
            ik_targets = {
                foot_name: ik_targets[foot_name]
                for _, foot_name in ik_contacts
                if foot_name in ik_targets
            }

            # A single support foot can be satisfied exactly by translating
            # the free root below. Solving a redundant 18-DOF leg IK in that
            # case only changes the leg's posture and can select another ball
            # joint branch. Use leg IK for the genuinely closed-chain case.
            if leg_ik and len(ik_contacts) >= 1:
                qpos_before_ik = qpos.copy()
                qpos, support_ik_error = custom_leg_ik(
                    human,
                    qpos,
                    ik_contacts,
                    ik_targets,
                    iterations=4,
                    root_regularization=0.05,
                    include_root=True,
                    max_joint_step=MAX_SUPPORT_JOINT_STEP_RAD,
                    max_total_joint_step=MAX_SUPPORT_TOTAL_JOINT_STEP_RAD,
                    max_root_step=MAX_CONTACT_SWITCH_CORRECTION_STEP_M
                    if {name for _, name in active_contacts} != previous_runtime_contact_names
                    else MAX_ROOT_CORRECTION_STEP_M,
                )
                qpos = np.asarray(qpos, dtype=np.float32)
                qpos = interpolate_qpos(qpos_before_ik, qpos, ik_weight)
                human.set_qpos(qpos, zero_velocity=True)
                support_error_vector = np.concatenate(
                    [
                        ik_targets[foot_name] - joint_position(human, foot_name)
                        for _, foot_name in ik_contacts
                    ]
                )
                support_ik_error = float(np.max(np.abs(support_error_vector)))
                pre_post_ik_jump_metrics.append(
                    {
                        "step": int(step),
                        "source_frame": int(frame),
                        "kind": "support",
                        "max_abs_qpos_delta": float(np.max(np.abs(qpos - qpos_before_ik))),
                        "root_translation_delta_m": float(
                            np.linalg.norm(
                                qpos[free_start : free_start + 3]
                                - qpos_before_ik[free_start : free_start + 3]
                            )
                        ),
                    }
                )
                if support_ik_error > MAX_ACCEPTABLE_SUPPORT_IK_ERROR_M:
                    # Reject an unreachable support solution rather than
                    # forcing the leg into a visibly unstable configuration.
                    qpos = qpos_before_ik
                    support_ik_rejections += 1
                    support_ik_rejection_frames.append(int(frame))
                    for _, foot_name in ik_contacts:
                        runtime_contact_disabled[foot_name] = True
                        locked_foot_xy[foot_name] = None
                        contact_ramp[foot_name] = 0.0
                        ik_targets.pop(foot_name, None)
                    # Do not leave another hard target active after rejecting
                    # the primary closure: it was not solved against the
                    # fallback pose either.
                    active_contacts = []
                    ik_targets.clear()
                else:
                    solved_leg_names = {
                        foot_name.split("_")[0] for _, foot_name in ik_contacts
                    }
                    accepted_ik_target_names = set(ik_targets)
                max_ik_qpos_delta = max(max_ik_qpos_delta, float(np.max(np.abs(qpos - qpos_before_ik))))
                root_delta = qpos[free_start : free_start + 3] - qpos_before_ik[free_start : free_start + 3]
                root_delta_norm = float(np.linalg.norm(root_delta))
                max_root_correction = max(max_root_correction, root_delta_norm)
                max_root_correction_step = max(max_root_correction_step, root_delta_norm)
                human.set_qpos(qpos, zero_velocity=True)
                ik_frames += 1

            # Enforce the frame-to-frame joint limit before root translation
            # correction. The root may still translate to satisfy a foot
            # target, but no later operation may rotate the leg again.
            if previous_applied_leg_qpos is not None:
                for joint_name in LEG_JOINT_NAMES:
                    joint = human.get_joint(name=joint_name)
                    if joint_name.split("_")[0] in solved_leg_names:
                        continue
                    limit_ball_joint_change(
                        qpos,
                        previous_applied_leg_qpos,
                        joint,
                        max_angle=MAX_LEG_FRAME_STEP_RAD,
                    )
                human.set_qpos(qpos, zero_velocity=True)

            runtime_contact_names = {name for _, name in active_contacts}
            previous_runtime_contact_names = runtime_contact_names
        if not active_contacts:
            previous_runtime_contact_names = set()
        runtime_contact_names = {name for _, name in active_contacts}
        if runtime_contact_names != previous_metric_contact_names:
            contact_transition_frames.append(
                {
                    "step": int(step),
                    "source_frame": int(frame),
                    "from": sorted(previous_metric_contact_names),
                    "to": sorted(runtime_contact_names),
                }
            )
        previous_metric_contact_names = runtime_contact_names.copy()

        # Keep swing feet above the ground with their own leg IK. A root-Z
        # clamp would lift the support foot as well and fight support IK.
        actual_feet = [joint_position(human, foot_name) for foot_name in FOOT_NAMES]
        active_contact_names = {foot_name for _, foot_name in active_contacts}
        if leg_ik and foot_lock:
            for foot_index, foot_name in enumerate(FOOT_NAMES):
                if foot_name in active_contact_names:
                    continue
                actual = joint_position(human, foot_name)
                if float(actual[2]) >= foot_clearance:
                    continue
                swing_target = actual.copy()
                # Solve the whole clearance error in the current IK call. The
                # pose entering this block is already warm-started from the
                # previous applied leg solution, so limiting the joint step
                # controls continuity without allowing a foot to remain below
                # the ground for dozens of simulation steps.
                swing_target[2] = foot_clearance
                qpos_before_swing_ik = qpos.copy()
                qpos, _ = custom_leg_ik(
                    human,
                    qpos,
                    [(foot_index, foot_name)],
                    {foot_name: swing_target},
                    iterations=8,
                    max_joint_step=0.06,
                    max_total_joint_step=0.12,
                )
                qpos = np.asarray(qpos, dtype=np.float32)
                qpos = interpolate_qpos(qpos_before_swing_ik, qpos, ik_weight)
                pre_post_ik_jump_metrics.append(
                    {
                        "step": int(step),
                        "source_frame": int(frame),
                        "kind": "swing",
                        "foot": foot_name,
                        "max_abs_qpos_delta": float(np.max(np.abs(qpos - qpos_before_swing_ik))),
                        "root_translation_delta_m": float(
                            np.linalg.norm(
                                qpos[free_start : free_start + 3]
                                - qpos_before_swing_ik[free_start : free_start + 3]
                            )
                        ),
                    }
                )
                human.set_qpos(qpos, zero_velocity=True)
                swing_delta = float(np.max(np.abs(qpos - qpos_before_swing_ik)))
                max_ik_qpos_delta = max(max_ik_qpos_delta, swing_delta)
        # Last-resort geometric safety: if a limited swing/support IK solution
        # still leaves an anchor below the clearance plane, lift the free root
        # by exactly the residual. This path is only for an infeasible IK
        # result (normally it remains zero); it prevents collision geometry
        # from visibly entering the ground without changing bone lengths.
        human.set_qpos(qpos, zero_velocity=True)
        actual_feet_after_ik = [joint_position(human, foot_name) for foot_name in FOOT_NAMES]
        minimum_foot_z = min(float(foot[2]) for foot in actual_feet_after_ik)
        desired_ground_lift = max(0.0, foot_clearance - minimum_foot_z)
        ground_lift_delta = desired_ground_lift - ground_lift_state
        if ground_lift_delta > 0.0:
            ground_lift_state = desired_ground_lift
            qpos[free_start + 2] += ground_lift_delta
            max_ground_clamp = max(max_ground_clamp, abs(ground_lift_delta))
            human.set_qpos(qpos, zero_velocity=True)
        elif ground_lift_delta < 0.0:
            release = min(-ground_lift_delta, ROOT_CORRECTION_RELEASE_STEP_M)
            ground_lift_state -= release
            qpos[free_start + 2] -= release
            max_ground_clamp = max(max_ground_clamp, release)
            human.set_qpos(qpos, zero_velocity=True)
        correction = root_correction_state.copy()
        if leg_ik:
            for foot_name, target in ik_targets.items():
                if foot_name not in accepted_ik_target_names:
                    continue
                foot_error = float(np.linalg.norm(joint_position(human, foot_name) - target))
                if foot_error > max_ik_position_error:
                    max_ik_position_error = foot_error
                    max_ik_position_error_frame = frame
                    max_ik_position_error_foot = foot_name
        combined_root_state = root_correction_state.copy()
        combined_root_state[2] += ground_lift_state
        max_root_correction_state = max(
            max_root_correction_state, float(np.linalg.norm(combined_root_state))
        )
        if previous_applied_leg_qpos is not None:
            frame_leg_step = 0.0
            frame_leg_step_joint = None
            for joint_name in LEG_JOINT_NAMES:
                joint = human.get_joint(name=joint_name)
                old_q = previous_applied_leg_qpos[joint.q_start : joint.q_start + 4]
                new_q = qpos[joint.q_start : joint.q_start + 4]
                old_rotation = Rotation.from_quat([old_q[1], old_q[2], old_q[3], old_q[0]])
                new_rotation = Rotation.from_quat([new_q[1], new_q[2], new_q[3], new_q[0]])
                joint_step = float((old_rotation.inv() * new_rotation).magnitude())
                if joint_step > frame_leg_step:
                    frame_leg_step = joint_step
                    frame_leg_step_joint = joint_name
            if frame_leg_step > max_leg_joint_step_rad:
                max_leg_joint_step_rad = frame_leg_step
                max_leg_joint_step_frame = frame
                max_leg_joint_step_joint = frame_leg_step_joint
        previous_applied_leg_qpos = qpos.copy()
        previous_active_leg_names = active_leg_names
        previous_applied_qpos = qpos.copy()
        applied_root = qpos[human.get_joint(name="pelvis_free").q_start :][:3].copy()
        if leg_ik and foot_lock and accepted_ik_target_names:
            # Carry the actual support correction into the next source pose.
            # This prevents the IK solver from paying the same root correction
            # again from scratch on every 100 Hz step.
            root_correction_state = applied_root - commanded_root
        elif leg_ik and foot_lock:
            correction_norm = float(np.linalg.norm(root_correction_state))
            if correction_norm <= ROOT_CORRECTION_RELEASE_STEP_M:
                root_correction_state[:] = 0.0
            else:
                root_correction_state *= (
                    correction_norm - ROOT_CORRECTION_RELEASE_STEP_M
                ) / correction_norm
        applied_qpos_history.append(qpos.copy())
        applied_root_history.append(applied_root.copy())
        root_correction_history.append(applied_root - commanded_root)
        if previous_applied_root is not None:
            max_applied_root_step_m = max(
                max_applied_root_step_m,
                float(np.linalg.norm(applied_root - previous_applied_root)),
            )
        previous_applied_root = applied_root
        current_bone_lengths = skeleton_bone_lengths(human)
        max_bone_length_error = max(
            max_bone_length_error,
            float(np.max(np.abs(current_bone_lengths - expected_bone_lengths))),
        )
        evaluation_points = frame_points + correction if np.any(correction) else frame_points
        actual_root = joint_position(human, "pelvis")
        max_root_error = max(max_root_error, float(np.linalg.norm(actual_root - evaluation_points[0])))
        frame_joint_error = 0.0
        frame_joint_name = None
        for joint_index, body_name in enumerate(BODY_NAMES):
            actual = joint_position(human, body_name)
            error = float(np.linalg.norm(actual - evaluation_points[joint_index]))
            if error > frame_joint_error:
                frame_joint_error = error
                frame_joint_name = body_name
        if frame_joint_error > max_joint_error:
            max_joint_error = frame_joint_error
            max_joint_error_name = frame_joint_name
            max_joint_error_frame = frame
        for foot_index, foot_name in enumerate(FOOT_NAMES):
            actual_foot = joint_position(human, foot_name)
            is_contact = foot_name in accepted_ik_target_names
            if is_contact:
                foot_contact_counts[foot_name] += 1
                if previous_foot_positions[foot_name] is not None:
                    foot_sliding_distance[foot_name] += float(
                        np.linalg.norm(actual_foot[:2] - previous_foot_positions[foot_name][:2])
                    )
                max_foot_height_error = max(max_foot_height_error, abs(float(actual_foot[2] - foot_clearance)))
            below_ground = max(0.0, -float(actual_foot[2]))
            max_foot_anchor_below_ground = max(max_foot_anchor_below_ground, below_ground)
            if below_ground > max_foot_anchor_below_ground_by_foot[foot_name]:
                max_foot_anchor_below_ground_by_foot[foot_name] = below_ground
                max_foot_anchor_below_ground_frame[foot_name] = frame
                max_foot_anchor_below_ground_context[foot_name] = {
                    "step": step,
                    "active_contacts": sorted(active_contact_names),
                    "ground_lift_state_m": ground_lift_state,
                    "foot_z_m": float(actual_foot[2]),
                }
            max_foot_penetration = max(
                max_foot_penetration,
                max(0.0, FOOT_GEOM_RADIUS_M - float(actual_foot[2])),
            )
            previous_foot_positions[foot_name] = actual_foot if is_contact else None
        foot_position_history.append(
            np.asarray([joint_position(human, foot_name) for foot_name in FOOT_NAMES])
        )
        if show_viewer:
            scene.step()
        else:
            scene.step(update_visualizer=False)
        post_step_lengths = skeleton_bone_lengths(human)
        max_post_step_bone_length_error = max(
            max_post_step_bone_length_error,
            float(np.max(np.abs(post_step_lengths - expected_bone_lengths))),
        )
        if collision:
            post_step_qpos = np.asarray(human.get_qpos(), dtype=np.float32).reshape(-1)
            max_post_step_qpos_drift = max(
                max_post_step_qpos_drift,
                float(np.max(np.abs(post_step_qpos - qpos))),
            )
            # This is a kinematic replay. Collision response must not be
            # allowed to overwrite the commanded pose and visually separate
            # the chain. Real contact dynamics require a PD/torque controller.
            human.set_qpos(qpos, zero_velocity=True)

    joint_starts = [3] + list(range(7, source_qposes.shape[1], 4))
    source_root_metrics = vector_motion_metrics(
        source_qposes[:, free_start : free_start + 3], 1.0 / fps
    )
    commanded_root_metrics = vector_motion_metrics(
        np.asarray(commanded_root_history), SIM_DT
    )
    applied_root_metrics = vector_motion_metrics(
        np.asarray(applied_root_history), SIM_DT
    )
    source_joint_metrics = quaternion_motion_metrics(
        source_qposes, joint_starts, 1.0 / fps
    )
    commanded_joint_metrics = quaternion_motion_metrics(
        np.asarray(commanded_qpos_history), joint_starts, SIM_DT
    )
    applied_joint_metrics = quaternion_motion_metrics(
        np.asarray(applied_qpos_history), joint_starts, SIM_DT
    )
    root_correction_metrics = vector_motion_metrics(
        np.asarray(root_correction_history), SIM_DT
    )
    applied_foot_metrics = {
        foot_name: vector_motion_metrics(
            np.asarray(foot_position_history)[:, foot_index, :], SIM_DT
        )
        for foot_index, foot_name in enumerate(FOOT_NAMES)
    }
    report = {
        "status": "success",
        "motion": str(motion_path.resolve()),
        "model": str(model_path.resolve()),
        "frames": int(len(points)),
        "motion_fps": fps,
        "sim_dt": SIM_DT,
        "interpolation": interpolation,
        "ik_weight": ik_weight,
        "steps": steps,
        "source_frame_reuse_steps": int(source_frame_reuse_steps),
        "max_commanded_root_step_m": max_commanded_root_step_m,
        "max_applied_root_step_m": max_applied_root_step_m,
        "source_ground_offset_m": source_ground_offset,
        "source_min_foot_height_m": source_min_foot_height,
        "duration_seconds": requested_duration,
        "joints": len(BODY_NAMES),
        "qpos_size": int(human.n_qs),
        "human_mode": "mjcf_ball_joint_kinematic_replay",
        "collision_geometry": "capsule_and_sphere",
        "collision_enabled": collision,
        "foot_lock_enabled": foot_lock,
        "foot_contact_height_m": foot_contact_height,
        "foot_clearance_m": foot_clearance,
        "foot_release_height_m": foot_release_height,
        "leg_ik_enabled": leg_ik,
        "leg_ik_dofs": 18,
        "leg_ik_frames": int(ik_frames),
        "max_leg_ik_position_error_m": max_ik_position_error,
        "max_leg_ik_position_error_frame": max_ik_position_error_frame,
        "max_leg_ik_position_error_foot": max_ik_position_error_foot,
        "max_leg_ik_qpos_delta": max_ik_qpos_delta,
        "support_ik_rejections": int(support_ik_rejections),
        "support_ik_rejection_frames": support_ik_rejection_frames,
        "max_acceptable_support_ik_error_m": MAX_ACCEPTABLE_SUPPORT_IK_ERROR_M,
        "max_leg_joint_step_rad": max_leg_joint_step_rad,
        "max_leg_frame_step_limit_rad": MAX_LEG_FRAME_STEP_RAD,
        "max_leg_joint_step_frame": max_leg_joint_step_frame,
        "max_leg_joint_step_joint": max_leg_joint_step_joint,
        "max_bone_length_error_m": max_bone_length_error,
        "max_post_step_bone_length_error_m": max_post_step_bone_length_error,
        "max_post_step_qpos_drift": max_post_step_qpos_drift,
        "foot_contact_frames": {
            name: int(count) for name, count in foot_contact_counts.items()
        },
        "max_foot_penetration_m": max_foot_penetration,
        "max_foot_anchor_below_ground_m": max_foot_anchor_below_ground,
        "max_foot_anchor_below_ground_by_foot_m": max_foot_anchor_below_ground_by_foot,
        "max_foot_anchor_below_ground_frame_by_foot": max_foot_anchor_below_ground_frame,
        "max_foot_anchor_below_ground_context_by_foot": max_foot_anchor_below_ground_context,
        "max_foot_height_error_m": max_foot_height_error,
        "foot_sliding_distance_m": foot_sliding_distance,
        "max_root_correction_m": max_root_correction,
        "max_root_correction_step_m": max_root_correction_step,
        "max_root_correction_state_m": max_root_correction_state,
        "max_ground_clamp_m": max_ground_clamp,
        "max_root_tracking_error_m": max_root_error,
        "max_joint_position_error_m": max_joint_error,
        "max_joint_position_error_joint": max_joint_error_name,
        "max_joint_position_error_frame": max_joint_error_frame,
        "max_root_velocity_m_s": applied_root_metrics["max_velocity"],
        "max_root_acceleration_m_s2": applied_root_metrics["max_acceleration"],
        "max_root_jerk_m_s3": applied_root_metrics["max_jerk"],
        "max_joint_velocity_rad_s": applied_joint_metrics["max_joint_velocity_rad_s"],
        "max_joint_acceleration_rad_s2": applied_joint_metrics["max_joint_acceleration_rad_s2"],
        "max_joint_jerk_rad_s3": applied_joint_metrics["max_joint_jerk_rad_s3"],
        "smoothness_metrics": {
            "source_30fps": {
                "root": source_root_metrics,
                "joints": source_joint_metrics,
            },
            "commanded_100hz": {
                "root": commanded_root_metrics,
                "joints": commanded_joint_metrics,
            },
            "applied_100hz": {
                "root": applied_root_metrics,
                "joints": applied_joint_metrics,
                "root_correction": root_correction_metrics,
                "feet": applied_foot_metrics,
            },
        },
        "contact_transition_frames": contact_transition_frames,
        "pre_post_ik_jump_metrics": sorted(
            pre_post_ik_jump_metrics,
            key=lambda item: item["max_abs_qpos_delta"],
            reverse=True,
        )[:20],
        "elapsed_wall_seconds": time.perf_counter() - started,
        "viewer": show_viewer,
        "camera_follow_enabled": camera_follow_enabled,
        "camera_mode": "follow_human" if camera_follow_enabled else "fixed",
        "kinematic_pose_reapplied_after_collision_step": bool(collision),
        "note": "Joint rotations are retargeted from keypoint bone directions; twist is underdetermined. MJCF fixes parent-child bone lengths. Foot lock is kinematic preprocessing. Optional leg IK constrains hip/knee/ankle DOFs only and is not dynamics/PD control.",
    }
    report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    report_path.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run(
        args.motion,
        args.model,
        args.report,
        args.seconds,
        args.viewer,
        args.fixed_camera,
        args.collision,
        args.foot_lock,
        args.foot_contact_height,
        args.foot_clearance,
        args.foot_release_height,
        args.leg_ik,
        args.interpolation,
        args.ik_weight,
    )
