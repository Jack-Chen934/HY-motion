"""Replay HY-Motion keypoints through a 22-joint MJCF ball-joint human in Genesis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation

from generate_human_mjcf import BODY_NAMES, PARENTS, HY_TO_GENESIS, rotation_from_basis


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
DEFAULT_MODEL = WORKSPACE / "genesis-human-experiment/assets/human_22ball.xml"
DEFAULT_REPORT = WORKSPACE / "genesis-human-experiment/reports/stage_b_mjcf_replay.json"
SIM_DT = 0.01
FOOT_NAMES = ("left_foot", "right_foot")
FOOT_GEOM_RADIUS_M = 0.045


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
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
        default=FOOT_GEOM_RADIUS_M,
        help="Target foot-anchor height above the ground during contact (meters).",
    )
    parser.add_argument(
        "--foot-release-height",
        type=float,
        default=0.085,
        help="A locked foot is released only above this raw height (meters).",
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
    lock_xy = np.zeros((2, 2), dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    adjusted = np.empty_like(raw)
    contact_mask = np.zeros((len(raw), 2), dtype=bool)
    translations = np.zeros((len(raw), 3), dtype=np.float32)

    for frame, frame_points in enumerate(raw):
        raw_foot = frame_points[foot_indices]
        for foot in range(2):
            z = float(raw_foot[foot, 2])
            if locked[foot] and z > release_height:
                locked[foot] = False
            elif not locked[foot] and z <= contact_height:
                locked[foot] = True
                lock_xy[foot] = raw_foot[foot, :2] + translation[:2]

        active = np.flatnonzero(locked)
        if len(active) > 0:
            xy_constraints = lock_xy[active] - raw_foot[active, :2]
            z_constraints = clearance - raw_foot[active, 2]
            translation[:2] = np.mean(xy_constraints, axis=0)
            translation[2] = float(np.mean(z_constraints))
        else:
            # Return smoothly to the source trajectory after the last support foot
            # leaves the ground, avoiding a visible root teleport.
            translation *= 0.85

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
    collision: bool,
    foot_lock: bool,
    foot_contact_height: float,
    foot_clearance: float,
    foot_release_height: float,
) -> None:
    points, fps = prepare_points(motion_path.resolve())
    contact_info = {
        "contact_mask": np.zeros((len(points), 2), dtype=bool),
        "translations": np.zeros((len(points), 3), dtype=np.float32),
    }
    if foot_lock:
        points, contact_info = apply_foot_lock(
            points,
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
            res=(960, 640), camera_pos=(2.8, -2.8, 1.8), camera_lookat=(0.0, 0.0, 0.9), camera_fov=40
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(gs.morphs.MJCF(file=str(model_path.resolve())), name="hymotion_human_22ball")
    scene.build()

    max_root_error = 0.0
    max_joint_error = 0.0
    max_joint_error_name = None
    max_joint_error_frame = None
    max_foot_penetration = 0.0
    max_foot_anchor_below_ground = 0.0
    max_foot_height_error = 0.0
    foot_contact_counts = {name: 0 for name in FOOT_NAMES}
    foot_sliding_distance = {name: 0.0 for name in FOOT_NAMES}
    previous_foot_positions = {name: None for name in FOOT_NAMES}
    locked_foot_xy = {name: None for name in FOOT_NAMES}
    max_root_correction = 0.0
    max_ground_clamp = 0.0
    started = time.perf_counter()
    for step in range(steps + 1):
        frame = min(int(round(min(step * SIM_DT, requested_duration) * fps)), len(points) - 1)
        frame_points = points[frame]
        qpos = qpos_for_frame(frame_points, rest, rest_basis, human)
        human.set_qpos(qpos, zero_velocity=True)
        correction = np.zeros(3, dtype=np.float32)

        # Correct the free-root translation from the actual Genesis anchors. The
        # input keypoints and the fixed-length skeleton are not identical, so a
        # keypoint-only lock can still leave the collision spheres underground.
        active_contacts = []
        if foot_lock:
            for foot_index, foot_name in enumerate(FOOT_NAMES):
                if contact_info["contact_mask"][frame, foot_index]:
                    active_contacts.append((foot_index, foot_name))
                    actual = joint_position(human, foot_name)
                    if locked_foot_xy[foot_name] is None:
                        locked_foot_xy[foot_name] = actual[:2].copy()
                else:
                    locked_foot_xy[foot_name] = None
            if active_contacts:
                free = human.get_joint(name="pelvis_free")
                for _ in range(3):
                    constraints = []
                    for _, foot_name in active_contacts:
                        actual = joint_position(human, foot_name)
                        constraints.append(
                            np.array(
                                [
                                    locked_foot_xy[foot_name][0] - actual[0],
                                    locked_foot_xy[foot_name][1] - actual[1],
                                    foot_clearance - actual[2],
                                ],
                                dtype=np.float32,
                            )
                        )
                    delta = np.mean(constraints, axis=0)
                    qpos[free.q_start : free.q_start + 3] += delta
                    correction += delta
                    human.set_qpos(qpos, zero_velocity=True)
                max_root_correction = max(max_root_correction, float(np.linalg.norm(correction)))
        else:
            active_contacts = []

        # Keep every foot collision sphere outside the plane, including swing
        # frames where no foot-lock constraint is active.
        actual_feet = [joint_position(human, foot_name) for foot_name in FOOT_NAMES]
        lowest_foot_z = min(float(position[2]) for position in actual_feet)
        if foot_lock and lowest_foot_z < foot_clearance:
            ground_correction = np.array([0.0, 0.0, foot_clearance - lowest_foot_z], dtype=np.float32)
            qpos[human.get_joint(name="pelvis_free").q_start : human.get_joint(name="pelvis_free").q_start + 3] += ground_correction
            human.set_qpos(qpos, zero_velocity=True)
            correction += ground_correction
            max_ground_clamp = max(max_ground_clamp, float(ground_correction[2]))
            max_root_correction = max(max_root_correction, float(np.linalg.norm(correction)))
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
            is_contact = bool(contact_info["contact_mask"][frame, foot_index])
            if is_contact:
                foot_contact_counts[foot_name] += 1
                if previous_foot_positions[foot_name] is not None:
                    foot_sliding_distance[foot_name] += float(
                        np.linalg.norm(actual_foot[:2] - previous_foot_positions[foot_name][:2])
                    )
                max_foot_height_error = max(max_foot_height_error, abs(float(actual_foot[2] - foot_clearance)))
            max_foot_anchor_below_ground = max(max_foot_anchor_below_ground, max(0.0, -float(actual_foot[2])))
            max_foot_penetration = max(
                max_foot_penetration,
                max(0.0, FOOT_GEOM_RADIUS_M - float(actual_foot[2])),
            )
            previous_foot_positions[foot_name] = actual_foot if is_contact else None
        if show_viewer:
            scene.step()
        else:
            scene.step(update_visualizer=False)

    report = {
        "status": "success",
        "motion": str(motion_path.resolve()),
        "model": str(model_path.resolve()),
        "frames": int(len(points)),
        "motion_fps": fps,
        "sim_dt": SIM_DT,
        "steps": steps,
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
        "foot_contact_frames": {
            name: int(count) for name, count in foot_contact_counts.items()
        },
        "max_foot_penetration_m": max_foot_penetration,
        "max_foot_anchor_below_ground_m": max_foot_anchor_below_ground,
        "max_foot_height_error_m": max_foot_height_error,
        "foot_sliding_distance_m": foot_sliding_distance,
        "max_root_correction_m": max_root_correction,
        "max_ground_clamp_m": max_ground_clamp,
        "max_root_tracking_error_m": max_root_error,
        "max_joint_position_error_m": max_joint_error,
        "max_joint_position_error_joint": max_joint_error_name,
        "max_joint_position_error_frame": max_joint_error_frame,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "viewer": show_viewer,
        "note": "Joint rotations are retargeted from keypoint bone directions; twist is underdetermined and not fitted. Foot lock is translation-only preprocessing and is not a dynamics/IK contact solve.",
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
        args.collision,
        args.foot_lock,
        args.foot_contact_height,
        args.foot_clearance,
        args.foot_release_height,
    )
