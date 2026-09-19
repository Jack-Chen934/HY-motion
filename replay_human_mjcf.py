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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--collision", action="store_true")
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


def run(motion_path: Path, model_path: Path, report_path: Path, seconds: float | None, show_viewer: bool, collision: bool) -> None:
    points, fps = prepare_points(motion_path.resolve())
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
    started = time.perf_counter()
    for step in range(steps + 1):
        frame = min(int(round(min(step * SIM_DT, requested_duration) * fps)), len(points) - 1)
        qpos = qpos_for_frame(points[frame], rest, rest_basis, human)
        human.set_qpos(qpos, zero_velocity=True)
        actual_root = np.asarray(human.get_link(name="pelvis").get_pos()).reshape(-1)[:3]
        max_root_error = max(max_root_error, float(np.linalg.norm(actual_root - points[frame, 0])))
        frame_joint_error = 0.0
        frame_joint_name = None
        for joint_index, body_name in enumerate(BODY_NAMES):
            actual = np.asarray(human.get_link(name=body_name).get_pos()).reshape(-1)[:3]
            error = float(np.linalg.norm(actual - points[frame, joint_index]))
            if error > frame_joint_error:
                frame_joint_error = error
                frame_joint_name = body_name
        if frame_joint_error > max_joint_error:
            max_joint_error = frame_joint_error
            max_joint_error_name = frame_joint_name
            max_joint_error_frame = frame
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
        "max_root_tracking_error_m": max_root_error,
        "max_joint_position_error_m": max_joint_error,
        "max_joint_position_error_joint": max_joint_error_name,
        "max_joint_position_error_frame": max_joint_error_frame,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "viewer": show_viewer,
        "note": "Joint rotations are retargeted from keypoint bone directions; twist is underdetermined and not fitted in this stage.",
    }
    report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    report_path.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run(args.motion, args.model, args.report, args.seconds, args.viewer, args.collision)
