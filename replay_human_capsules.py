"""Replay a prepared HY-Motion clip as a 22-joint kinematic Genesis human."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import genesis as gs
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
DEFAULT_REPORT = WORKSPACE / "genesis-human-experiment/reports/stage_a_replay.json"
SIM_DT = 0.01
BODY_COUNT = 22
BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)
BODY_EDGES = [(child, int(parent)) for child, parent in enumerate(BODY_PARENTS) if parent >= 0]

# HY-Motion is Y-up. This proper rotation maps HY y-up to Genesis z-up.
HY_TO_GENESIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def convert_points(points_hy: np.ndarray, ground_lift: float) -> np.ndarray:
    points_genesis = points_hy @ HY_TO_GENESIS.T
    points_genesis = points_genesis.copy()
    points_genesis[..., 2] += ground_lift
    return points_genesis


def quat_z_to_vector(vector: np.ndarray) -> np.ndarray:
    target = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(target)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    target /= norm
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(source, target))
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * np.arccos(np.clip(dot, -1.0, 1.0))
    sin_half = np.sin(half_angle)
    return np.array(
        [np.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float32,
    )


def segment_pose(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    vector = end - start
    length = float(np.linalg.norm(vector))
    return (start + end) * 0.5, max(length, 0.02), quat_z_to_vector(vector)


def interpolate_points(points: np.ndarray, frame_float: float) -> np.ndarray:
    frame_float = float(np.clip(frame_float, 0.0, len(points) - 1))
    left = int(np.floor(frame_float))
    right = min(left + 1, len(points) - 1)
    alpha = frame_float - left
    return (1.0 - alpha) * points[left] + alpha * points[right]


def add_human(scene: gs.Scene, first_frame: np.ndarray):
    body_material = gs.materials.Kinematic()
    body_surface = gs.surfaces.Default(color=(0.72, 0.58, 0.38, 1.0), roughness=0.8)
    entities = []
    lengths = {}

    for child, parent in BODY_EDGES:
        pos, length, quat = segment_pose(first_frame[parent], first_frame[child])
        lengths[(parent, child)] = length
        entities.append(
            scene.add_entity(
                gs.morphs.Cylinder(
                    height=length,
                    radius=0.045,
                    pos=tuple(pos),
                    quat=tuple(quat),
                    fixed=True,
                ),
                material=body_material,
                surface=body_surface,
                name=f"human_bone_{parent}_{child}",
            )
        )

    for joint_index in range(BODY_COUNT):
        entities.append(
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.065,
                    pos=tuple(first_frame[joint_index]),
                    fixed=True,
                ),
                material=body_material,
                surface=body_surface,
                name=f"human_joint_{joint_index}",
            )
        )
    return entities, lengths


def update_human(entities, points: np.ndarray) -> None:
    bone_count = len(BODY_EDGES)
    for entity_index, (child, parent) in enumerate(BODY_EDGES):
        pos, _, quat = segment_pose(points[parent], points[child])
        entities[entity_index].set_pos(pos, relative=False, zero_velocity=True)
        entities[entity_index].set_quat(quat, relative=False, zero_velocity=True)
    for joint_index in range(BODY_COUNT):
        entities[bone_count + joint_index].set_pos(
            points[joint_index], relative=False, zero_velocity=True
        )


def run(motion_path: Path, report_path: Path, seconds: float | None, show_viewer: bool) -> None:
    motion_path = motion_path.resolve()
    report_path = report_path.resolve()
    motion = np.load(motion_path, allow_pickle=False)
    points_hy = motion["keypoints_hy"].astype(np.float32)
    fps = float(motion["fps"][0])
    if points_hy.ndim != 3 or points_hy.shape[1:] != (52, 3):
        raise ValueError(f"Expected keypoints_hy shape (T, 52, 3), got {points_hy.shape}")
    if not np.isfinite(points_hy).all():
        raise FloatingPointError("Motion cache contains NaN or Inf")

    points_candidate = points_hy @ HY_TO_GENESIS.T
    ground_lift = max(0.0, 0.02 - float(points_candidate[..., 2].min()))
    points = convert_points(points_hy, ground_lift)
    total_duration = (len(points) - 1) / fps
    requested_duration = total_duration if seconds is None else min(seconds, total_duration)
    steps = int(np.ceil(requested_duration / SIM_DT))

    gs.init(backend=gs.cpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=True, box_box_detection=True),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(3.0, -3.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.8),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human_entities, _ = add_human(scene, points[0, :BODY_COUNT])
    scene.build()

    max_entity_error = 0.0
    sampled_root = []
    started = time.perf_counter()
    for step in range(steps + 1):
        time_seconds = min(step * SIM_DT, requested_duration)
        target = interpolate_points(points, time_seconds * fps)
        update_human(human_entities, target)
        actual_root = as_numpy(human_entities[len(BODY_EDGES)].get_pos()).reshape(-1)[:3]
        max_entity_error = max(max_entity_error, float(np.linalg.norm(actual_root - target[0])))
        sampled_root.append(target[0].tolist())
        scene.step(update_visualizer=show_viewer)

    report = {
        "status": "success",
        "motion": str(motion_path),
        "frames": int(len(points)),
        "motion_fps": fps,
        "sim_dt": SIM_DT,
        "sim_hz": 1.0 / SIM_DT,
        "requested_duration_seconds": requested_duration,
        "steps": steps,
        "coordinate_conversion": "HY Y-up -> Genesis Z-up via proper rotation",
        "ground_lift_m": ground_lift,
        "human_mode": "kinematic_capsule_skeleton",
        "body_joints": BODY_COUNT,
        "max_root_tracking_error_m": max_entity_error,
        "root_start_genesis": sampled_root[0],
        "root_end_genesis": sampled_root[-1],
        "root_net_displacement_m": float(np.linalg.norm(np.asarray(sampled_root[-1]) - np.asarray(sampled_root[0]))),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "viewer": show_viewer,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run(args.motion, args.report, args.seconds, args.viewer)
