"""Measure fixed-base R1 Pro clearance against the animated human proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import genesis as gs
import numpy as np

from generate_human_mjcf import rotation_from_basis
from r1pro_adapter import load_r1pro
from replay_human_mjcf import (
    SIM_DT,
    build_cubic_qpos_interpolator,
    make_quaternion_sequence_continuous,
    prepare_points,
    qpos_for_frame,
)
from run_hri_walking_approach import (
    DEFAULT_HUMAN_MODEL,
    DEFAULT_MOTION,
    DEFAULT_ROBOT_URDF,
    HUMAN_OFFSET,
    LEFT_ARM_START,
    RIGHT_ARM_START,
    TORSO_START,
    as_numpy,
    contact_count,
    joint_indices,
)


DEFAULT_REPORT = Path(__file__).resolve().parent / "reports/r1pro_layout_diagnostic.json"
MIN_PROXY_CLEARANCE_M = 0.25
MIN_BASE_ROOT_DISTANCE_M = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--human-model", type=Path, default=DEFAULT_HUMAN_MODEL)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--base-x", type=float, default=-0.55)
    parser.add_argument("--base-y", type=float, default=-3.00)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-clearance", type=float, default=MIN_PROXY_CLEARANCE_M)
    parser.add_argument("--sample-hz", type=float, default=10.0, help="Pose samples per simulated second.")
    return parser.parse_args()


def nearest_geom_aabb_pair(first: object, second: object) -> tuple[float, tuple[int, int]]:
    best = float("inf")
    best_pair = (-1, -1)
    for first_index, first_geom in enumerate(first.geoms):
        first_aabb = as_numpy(first_geom.get_AABB())
        for second_index, second_geom in enumerate(second.geoms):
            distance = float(np.linalg.norm(np.maximum(
                np.maximum(first_aabb[0] - as_numpy(second_geom.get_AABB())[1],
                           as_numpy(second_geom.get_AABB())[0] - first_aabb[1]),
                0.0,
            )))
            if distance < best:
                best = distance
                best_pair = (first_index, second_index)
    return best, best_pair


def run(args: argparse.Namespace) -> dict[str, object]:
    motion_path = args.motion.resolve()
    human_path = args.human_model.resolve()
    model = load_r1pro(args.robot_urdf)
    for path in (motion_path, human_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.min_clearance < 0.0 or args.sample_hz <= 0.0:
        raise ValueError("clearance must be non-negative and sample rate positive")

    points, fps = prepare_points(motion_path)
    duration = (len(points) - 1) / fps if args.seconds is None else min(args.seconds, (len(points) - 1) / fps)
    if duration <= 0.0:
        raise ValueError("diagnostic duration must be positive")
    rest_basis = rotation_from_basis(points[0])
    base = np.asarray((args.base_x, args.base_y, 0.0), dtype=np.float32)

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=True, box_box_detection=True),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), camera_pos=(2.8, -5.4, 2.5),
            camera_lookat=(0.0, -1.5, 0.95), camera_fov=42,
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_path), requires_jac_and_IK=False),
        name="layout_human_proxy",
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(model.urdf_path), pos=tuple(base), fixed=True,
            merge_fixed_links=False, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0),
        name="layout_r1_pro",
    )
    scene.build()

    # Set the same neutral posture used by the handover task, then hold it for
    # the entire scan.  This is a layout check, not a controller or IK test.
    for names, target in (
        (tuple(f"left_arm_joint{i}" for i in range(1, 8)), LEFT_ARM_START),
        (tuple(f"right_arm_joint{i}" for i in range(1, 8)), RIGHT_ARM_START),
        (tuple(f"torso_joint{i}" for i in range(1, 5)), TORSO_START),
    ):
        dofs, _ = joint_indices(robot, names)
        robot.set_dofs_position(target, dofs, zero_velocity=True)

    free_start = human.get_joint(name="pelvis_free").q_start
    qposes = np.stack([
        qpos_for_frame(frame, points[0], rest_basis, human) for frame in points
    ]).astype(np.float32)
    qposes = make_quaternion_sequence_continuous(qposes)
    qposes[:, free_start : free_start + 3] += HUMAN_OFFSET
    qpos_at = build_cubic_qpos_interpolator(qposes, fps)
    steps = int(np.ceil(duration * args.sample_hz))
    min_proxy_clearance = float("inf")
    min_root_distance = float("inf")
    min_distance_time = 0.0
    closest_pair = (None, None)
    max_contacts = 0
    started = time.perf_counter()
    for step in range(steps + 1):
        t = min(step / args.sample_hz, duration)
        qpos = qpos_at(t)
        human.set_qpos(qpos, zero_velocity=True)
        scene.step(update_visualizer=args.viewer)
        clearance, pair = nearest_geom_aabb_pair(human, robot)
        if clearance < min_proxy_clearance:
            min_proxy_clearance = clearance
            min_distance_time = t
            closest_pair = pair
        human_root = qpos[free_start : free_start + 3]
        root_distance = float(np.linalg.norm(human_root[:2] - base[:2]))
        min_root_distance = min(min_root_distance, root_distance)
        max_contacts = max(max_contacts, contact_count(scene, human, robot))

    open_jaw = model.jaw_geometry((0.05, -0.05))
    closed_jaw = model.jaw_geometry((0.0, 0.0))
    report: dict[str, object] = {
        "task": "r1pro_human_proxy_layout_diagnostic",
        "status": "success" if min_proxy_clearance >= args.min_clearance and max_contacts == 0 else "failure",
        "robot_urdf": str(model.urdf_path),
        "human_model": str(human_path),
        "motion": str(motion_path),
        "robot_base_position_m": base.tolist(),
        "duration_s": duration,
        "pose_sample_count": steps + 1,
        "pose_sample_rate_hz": args.sample_hz,
        "max_sample_gap_s": 1.0 / args.sample_hz,
        "closest_geom_indices": closest_pair,
        "min_human_proxy_aabb_clearance_m": min_proxy_clearance,
        "min_required_proxy_clearance_m": args.min_clearance,
        "time_of_min_clearance_s": min_distance_time,
        "min_base_to_human_root_distance_m": min_root_distance,
        "min_required_base_to_human_root_distance_m": MIN_BASE_ROOT_DISTANCE_M,
        "contact_samples_max": max_contacts,
        "robot_links": len(model.links),
        "robot_joints": len(model.joints),
        "left_gripper": {
            "tcp_local_m": model.tcp_local.tolist(),
            "finger_collision_boxes": [
                {"link": finger.link_name, "joint": finger.joint_name,
                 "size_m": finger.size.tolist(), "collision_origin_m": finger.collision_origin.tolist(),
                 "joint_origin_m": finger.joint_origin.tolist(), "axis": finger.joint_axis.tolist(),
                 "limits_m": [finger.lower, finger.upper]}
                for finger in model.left_fingers
            ],
            "open": open_jaw,
            "closed": closed_jaw,
        },
        "layout_pass": min_proxy_clearance >= args.min_clearance and max_contacts == 0,
        "elapsed_wall_s": time.perf_counter() - started,
        "note": "AABB clearance is a conservative broad-phase geometry diagnostic. Robot remains fixed in neutral posture; no arm reachability, handover, or grasp is claimed.",
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        raise SystemExit(f"layout diagnostic failed: {type(exc).__name__}: {exc}") from exc
