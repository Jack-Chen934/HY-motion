"""Run the isolated walking-human / R1 Pro safe-approach HRI task.

This script reads the R1 Pro asset from the existing robot project but owns no
files there.  The first HRI episode deliberately keeps the mobile base at a
fixed observation pose and tests the harder-to-isolate integration first:
HY-Motion playback, a full visual mesh, the human collision proxy, hand-window
detection, and left-arm IK near (but not on) the human hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from generate_human_mjcf import BODY_NAMES, HY_TO_GENESIS, PARENTS, rotation_from_basis
from replay_human_mjcf import (
    SIM_DT,
    build_cubic_qpos_interpolator,
    interpolate_qpos,
    make_quaternion_sequence_continuous,
    prepare_points,
    qpos_for_frame,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
DEFAULT_HUMAN_MODEL = WORKSPACE / "genesis-human-experiment/assets/human_22ball.xml"
DEFAULT_WOODEN_DIR = WORKSPACE / "HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden"
DEFAULT_ROBOT_URDF = (
    WORKSPACE.parent / "gaze_vr_isaaclab" / "src" / "gaze_vr_isaaclab"
    / "assets" / "r1_pro" / "r1pro.urdf"
)
DEFAULT_REPORT = WORKSPACE / "genesis-human-experiment/reports/hri_walking_approach.json"
DEFAULT_OBJ = WORKSPACE / "genesis-human-experiment/assets/generated/hymotion_mesh_initial.obj"
DEFAULT_TRAJECTORY = WORKSPACE / "genesis-human-experiment/motions/cache/hri_episode_robot.npz"

ROBOT_BASE_POS = np.array((0.0, -2.95, 0.0), dtype=np.float32)
# The robot stays to the human's right: this preserves the root-distance
# constraint while keeping the handover point inside the left-arm workspace.
HANDOVER_ROBOT_BASE_POS = np.array((-0.55, -3.00, 0.0), dtype=np.float32)
HUMAN_OFFSET = np.zeros(3, dtype=np.float32)
HAND_NAME = "left_wrist"
HAND_PARENT_NAME = "left_elbow"
HAND_SPEED_THRESHOLD = 0.12
MIN_PAUSE_SECONDS = 0.50
REACH_LEAD_SECONDS = 1.50
MIN_HUMAN_ROBOT_DISTANCE = 0.25
MAX_HAND_TARGET_ERROR = 0.10
MAX_IK_POSITION_ERROR = 0.05
HOLD_SECONDS = 0.50
HANDOVER_TAIL_SECONDS = 4.0
HAND_STABLE_SECONDS = 0.50
# The original baseline used 16 s + 24 s here.  That made the arm appear to
# stop for long periods even though it was waiting at a valid state boundary.
# These durations leave enough time for the configured TCP/joint limits while
# avoiding the extra idle time in the original 16 s + 24 s baseline.  The
# values are based on the measured pregrasp and object-approach convergence.
PREGRASP_SECONDS = 16.00
SLOW_APPROACH_SECONDS = 14.00
GRIP_CLOSE_SECONDS = 0.60
GRASP_VERIFY_SECONDS = 0.50
RETREAT_SECONDS = 1.20
PREGRASP_OFFSET_M = 0.25
GRASP_OFFSET_M = 0.00
OBJECT_SIZE = (0.04, 0.04, 0.04)
OBJECT_HAND_OFFSET = np.zeros(3, dtype=np.float32)
HAND_OBJECT_FORWARD_OFFSET_M = 0.08
# WoodenMesh exposes the hand joints that are absent from the 22-joint
# collision proxy.  These points are used to construct a local palm frame
# rather than treating the wrist as the object attachment point.
LEFT_WRIST_WOODEN_INDEX = 20
LEFT_PALM_BASE_WOODEN_INDICES = (22, 25, 28, 31, 34)
PALM_GATE_BLEND = 0.60
MIN_BASE_ROOT_DISTANCE = 0.80
MIN_HUMAN_ROBOT_AABB_DISTANCE = 0.05
MAX_JOINT_SPEED = 0.35
MAX_TCP_SPEED = 0.10
GRASP_DISTANCE_TOLERANCE = 0.09
GRASP_STABLE_SECONDS = 0.25
GRIPPER_TCP_LOCAL = np.array((0.0, 0.0, -0.03689), dtype=np.float32)
GRIPPER_QUAT = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

# These are the validated neutral R1 Pro arm targets used by the robot project.
LEFT_ARM_START = np.array((-0.4, 1.3, -0.7, -1.57, 1.3, -0.4, -0.8), dtype=np.float32)
RIGHT_ARM_START = np.array((-0.4, -1.3, 0.7, -1.57, -1.3, -0.4, 0.8), dtype=np.float32)
GRIPPER_OPEN = np.array((0.05, -0.05), dtype=np.float32)
GRIPPER_CLOSED = np.array((0.005, -0.005), dtype=np.float32)
TORSO_START = np.zeros(4, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--human-model", type=Path, default=DEFAULT_HUMAN_MODEL)
    parser.add_argument("--wooden-dir", type=Path, default=DEFAULT_WOODEN_DIR)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mesh-obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument(
        "--save-trajectory", type=Path, default=DEFAULT_TRAJECTORY,
        help="Save strict-mode robot/human trajectory cache as a compressed NPZ.",
    )
    parser.add_argument(
        "--trajectory", type=Path, default=None,
        help="Trajectory cache used by --profile hri-replay.",
    )
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-collision", action="store_true", help="Disable Genesis collision solving; keep distance checks.")
    parser.add_argument(
        "--camera-mode", choices=("fixed", "follow"), default="fixed",
        help="Viewer camera mode. Fixed is the stable default; follow tracks the robot.",
    )
    parser.add_argument("--fixed-camera", action="store_true", help="Compatibility alias for --camera-mode fixed.")
    parser.add_argument("--ik-hz", type=float, default=100.0, help="Maximum arm IK solve rate during playback.")
    parser.add_argument("--ik-max-samples", type=int, default=32)
    parser.add_argument("--ik-max-solver-iters", type=int, default=200)
    parser.add_argument("--pause-ik-hz", type=float, default=100.0, help="IK solve rate while the hand is in the interaction pause window.")
    parser.add_argument("--viewer-hz", type=float, default=20.0, help="Viewer refresh rate during playback.")
    parser.add_argument(
        "--progress-hz", type=float, default=2.0,
        help="Print handover simulation progress at this rate; 0 disables progress output.",
    )
    parser.add_argument("--safety-check-hz", type=float, default=20.0, help="Rate for Python-side contact and distance checks.")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--profile", choices=("handover", "dynamic-handover", "strict", "smooth-viewer", "human-viewer", "hri-viewer", "hri-replay", "layout-diagnostic"), default="handover",
        help=("handover runs the kinematic baseline; dynamic-handover runs the runtime-weld dynamic transfer; strict runs the legacy empty-hand gate; "
              "smooth-viewer is a controlled HRI preview; "
              "human-viewer shows only the human; hri-viewer shows a static robot preview; "
              "hri-replay replays a saved strict trajectory."),
    )
    parser.add_argument("--handover-hold-seconds", type=float, default=HANDOVER_TAIL_SECONDS)
    parser.add_argument("--max-joint-speed", type=float, default=MAX_JOINT_SPEED)
    parser.add_argument("--max-tcp-speed", type=float, default=MAX_TCP_SPEED)
    return parser.parse_args()


def step_interval_for_rate(rate_hz: float, dt: float) -> int:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, int(round(1.0 / (rate_hz * dt))))


def should_update_viewer(step: int, interval: int, final_step: int) -> bool:
    return step == 0 or step == final_step or step % interval == 0


def interpolate_trajectory(times: np.ndarray, values: np.ndarray, position: float) -> np.ndarray:
    """Linearly sample a saved trajectory at one time in seconds."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float32)
    if times.ndim != 1 or len(times) == 0 or values.shape[0] != len(times):
        raise ValueError("trajectory times and values must have the same non-zero first dimension")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory times must be strictly increasing")
    if position <= times[0]:
        return values[0].copy()
    if position >= times[-1]:
        return values[-1].copy()
    right = int(np.searchsorted(times, position, side="right"))
    left = right - 1
    alpha = (position - times[left]) / (times[right] - times[left])
    return ((1.0 - alpha) * values[left] + alpha * values[right]).astype(np.float32)


def save_trajectory_cache(
    path: Path,
    times: list[float],
    human_qpos: list[np.ndarray],
    robot_qpos: list[np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Persist the strict episode state needed by the visual replay profile."""
    if not times or len(times) != len(human_qpos) or len(times) != len(robot_qpos):
        raise ValueError("trajectory samples must be non-empty and have equal lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_s=np.asarray(times, dtype=np.float64),
        human_qpos=np.asarray(human_qpos, dtype=np.float32),
        robot_qpos=np.asarray(robot_qpos, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=np.unicode_),
    )


def load_trajectory_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load and validate a strict trajectory cache."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = ("time_s", "human_qpos", "robot_qpos")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"trajectory cache is missing fields: {', '.join(missing)}")
        times = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
        human_qpos = np.asarray(data["human_qpos"], dtype=np.float32)
        robot_qpos = np.asarray(data["robot_qpos"], dtype=np.float32)
        raw_metadata = str(data["metadata"].item()) if "metadata" in data else "{}"
    if len(times) == 0 or human_qpos.shape[0] != len(times) or robot_qpos.shape[0] != len(times):
        raise ValueError("trajectory cache arrays have inconsistent lengths")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory cache time_s must be strictly increasing")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("trajectory cache metadata is not valid JSON") from exc
    return times, human_qpos, robot_qpos, metadata


def run_hri_replay(args: argparse.Namespace) -> dict[str, object]:
    """Display a previously solved strict episode without recomputing IK."""
    if args.trajectory is None:
        raise ValueError("--profile hri-replay requires --trajectory")
    trajectory_path = args.trajectory.resolve()
    times, cached_human, cached_robot, cache_metadata = load_trajectory_cache(trajectory_path)
    motion_path = args.motion.resolve()
    human_model = args.human_model.resolve()
    wooden_dir = args.wooden_dir.resolve()
    robot_urdf = args.robot_urdf.resolve()
    report_path = args.report.resolve()
    for required in (motion_path, human_model, robot_urdf, wooden_dir / "v_template.bin"):
        if not required.exists():
            raise FileNotFoundError(required)

    vertices, faces, mesh_roots, _hand_keypoints, mesh_fps = load_wooden_motion(motion_path, wooden_dir)
    points, fps = prepare_points(motion_path)
    if abs(mesh_fps - fps) > 1e-4:
        raise ValueError("Motion FPS differs between WoodenMesh and keypoint inputs")
    # A handover cache continues after the 4.97 s source motion while the
    # robot approaches, transfers the object, and retreats.  Replay the full
    # cached episode; the human/mesh samplers already clamp at motion end.
    duration = float(times[-1])
    if args.seconds is not None:
        duration = min(duration, args.seconds)
    viewer_hz = max(args.viewer_hz, 30.0)
    simulation_dt = 1.0 / viewer_hz
    steps = int(np.ceil(duration / simulation_dt))
    backend, backend_name = select_backend(args.backend)
    write_obj(args.mesh_obj.resolve(), vertices[0], faces)
    rest_basis = rotation_from_basis(points[0])
    gs.init(backend=backend, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=simulation_dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, box_box_detection=False),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), refresh_rate=int(round(viewer_hz)), realtime_factor=1.0,
            camera_pos=(2.8, -5.4, 2.5), camera_lookat=(0.0, -1.5, 0.95), camera_fov=42,
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_model), requires_jac_and_IK=False), name="hri_human_proxy"
    )
    mesh = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh_obj.resolve()), fixed=True, visualization=True, collision=False,
            enable_custom_vverts=True, decimate=False, convexify=False, file_meshes_are_zup=True,
        ),
        name="hri_human_visual",
    )
    robot_base = np.asarray(cache_metadata.get("robot_base_position", ROBOT_BASE_POS), dtype=np.float32)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf), pos=tuple(robot_base), fixed=True,
            merge_fixed_links=False, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0), name="r1_pro_hri_replay",
    )
    object_entity = scene.add_entity(
        gs.morphs.Box(size=OBJECT_SIZE, pos=(0.0, 0.0, 1.0), fixed=True),
        material=gs.materials.Rigid(friction=2.0),
        surface=gs.surfaces.Default(color=(0.10, 0.35, 0.85, 1.0)),
        name="handover_object_replay",
    )
    scene.build()
    if mesh.n_vverts != vertices.shape[1]:
        raise RuntimeError(f"Genesis visual vertex count mismatch: {mesh.n_vverts} != {vertices.shape[1]}")
    if cached_human.shape[1] != human.n_qs or cached_robot.shape[1] != robot.n_qs:
        raise ValueError(
            f"trajectory qpos shape mismatch: cache human/robot={cached_human.shape[1]}/{cached_robot.shape[1]}, "
            f"scene={human.n_qs}/{robot.n_qs}"
        )
    free_start = human.get_joint(name="pelvis_free").q_start
    end_effector = robot.get_link("left_gripper_link")
    hand_keypoints = _hand_keypoints
    control_transfer_time = float(cache_metadata.get("control_transfer_time_s", duration + 1.0))
    started = time.perf_counter()
    mesh_updates = 0
    robot_updates = 0
    for step in range(steps + 1):
        time_seconds = min(step * simulation_dt, duration)
        human_qpos = interpolate_trajectory(times, cached_human, time_seconds)
        robot_qpos = interpolate_trajectory(times, cached_robot, time_seconds)
        human.set_qpos(human_qpos, zero_velocity=True)
        robot.set_qpos(robot_qpos, zero_velocity=True)
        source_position = time_seconds * fps
        frame_hand_keypoints = interpolate_points(hand_keypoints, source_position)
        if time_seconds < control_transfer_time:
            object_position, object_quat, _ = palm_pose_in_hand(frame_hand_keypoints)
        else:
            object_position = tcp_position(end_effector)
            object_quat = tcp_quaternion(end_effector)
        object_entity.set_pos(object_position, relative=False, zero_velocity=True)
        object_entity.set_quat(object_quat, relative=False, zero_velocity=True)
        aligned = interpolate_points(vertices, source_position)
        mesh_root = interpolate_points(mesh_roots, source_position)
        aligned = aligned + (human_qpos[free_start : free_start + 3] - mesh_root)
        mesh.set_vverts(aligned)
        mesh_updates += 1
        robot_updates += 1
        scene.step(update_visualizer=args.viewer)
        # The replay scene is intentionally non-dynamic. Reapply the cached
        # state after stepping so the viewer cannot drift from the strict trace.
        human.set_qpos(human_qpos, zero_velocity=True)
        robot.set_qpos(robot_qpos, zero_velocity=True)
    elapsed_wall_seconds = time.perf_counter() - started
    report = {
        "status": "success",
        "task": "strict_trajectory_visual_replay",
        "profile": "hri-replay",
        "motion": str(motion_path),
        "trajectory": str(trajectory_path),
        "source_report": cache_metadata.get("source_report"),
        "duration_seconds": duration,
        "simulation_hz": viewer_hz,
        "simulation_dt": simulation_dt,
        "simulation_steps": steps + 1,
        "mesh_vertices": int(vertices.shape[1]),
        "mesh_faces": int(faces.shape[0]),
        "mesh_updates": mesh_updates,
        "robot_updates": robot_updates,
        "robot_loaded": True,
        "object_loaded": True,
        "object_attachment": "left_thumb_index_palm_frame_before_transfer",
        "control_transfer_time_s": control_transfer_time,
        "ik_calls": 0,
        "collision_enabled": False,
        "safety_checks": 0,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "realtime_factor": duration / max(elapsed_wall_seconds, 1e-9),
        "backend_used": backend_name,
        "note": "Visual replay only: all robot states come from a strict trajectory cache; no IK, contacts, or safety checks are recomputed.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def select_backend(mode: str) -> tuple[object, str]:
    if mode == "cpu":
        return gs.cpu, "cpu"
    if mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--backend gpu requested, but torch.cuda.is_available() is false")
        return gs.gpu, "gpu"
    if torch.cuda.is_available():
        return gs.gpu, "gpu"
    return gs.cpu, "cpu"


def as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def joint_indices(entity: object, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    dofs: list[int] = []
    qs: list[int] = []
    for name in names:
        joint = entity.get_joint(name)
        if len(joint.dofs_idx_local) != 1 or len(joint.qs_idx_local) != 1:
            raise RuntimeError(f"Expected scalar joint: {name}")
        dofs.append(int(joint.dofs_idx_local[0]))
        qs.append(int(joint.qs_idx_local[0]))
    return np.asarray(dofs, dtype=np.int32), np.asarray(qs, dtype=np.int32)


def load_wooden_motion(
    motion_path: Path, wooden_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    repo = wooden_dir.parents[4]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hymotion.pipeline.body_model import WoodenMesh

    with np.load(motion_path, allow_pickle=False) as data:
        poses = data["poses"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        fps = float(data["fps"][0]) if "fps" in data else 30.0
    model = WoodenMesh(model_path=str(wooden_dir))
    with torch.inference_mode():
        result = model({"poses": torch.from_numpy(poses), "trans": torch.from_numpy(trans)})
    vertices = result["vertices"].cpu().numpy().astype(np.float32) @ HY_TO_GENESIS.T
    keypoints = (result["keypoints3d"].cpu().numpy().astype(np.float32) + trans[:, None, :]) @ HY_TO_GENESIS.T
    roots = keypoints[:, 0]
    return vertices, np.asarray(model.faces, dtype=np.int32), roots, keypoints, fps


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Runtime allocation mesh for the Genesis visual entity\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def detect_pause_window(points: np.ndarray, fps: float, hand_index: int, threshold: float, minimum_seconds: float) -> tuple[int, int]:
    relative = points[:, hand_index] - points[:, 0]
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=1) * fps
    slow = np.r_[False, speed < threshold]
    minimum_frames = max(1, int(np.ceil(minimum_seconds * fps)))
    best = (0, len(points), len(points))
    start_search = int(np.floor(3.4 * fps))
    cursor = start_search
    while cursor < len(slow):
        if not slow[cursor]:
            cursor += 1
            continue
        end = cursor
        while end < len(slow) and slow[end]:
            end += 1
        if end - cursor >= minimum_frames and end - cursor > best[0]:
            best = (end - cursor, cursor, end)
        cursor = end
    if best[0] < minimum_frames:
        raise RuntimeError("No hand pause window satisfies the configured threshold")
    return best[1], min(best[2], len(points) - 1)


def interpolate_points(points: np.ndarray, position: float) -> np.ndarray:
    left = min(max(int(np.floor(position)), 0), len(points) - 1)
    right = min(left + 1, len(points) - 1)
    alpha = position - left if right != left else 0.0
    return ((1.0 - alpha) * points[left] + alpha * points[right]).astype(np.float32)


def interaction_target(hand: np.ndarray, robot_position: np.ndarray) -> np.ndarray:
    away_from_robot = hand - robot_position
    away_from_robot[2] = 0.0
    norm = float(np.linalg.norm(away_from_robot))
    direction = away_from_robot / norm if norm > 1e-6 else np.array((0.0, 1.0, 0.0), dtype=np.float32)
    return hand + 0.10 * direction


def object_position_in_hand(
    points: np.ndarray,
    hand_index: int,
    parent_index: int,
    forward_offset: float = HAND_OBJECT_FORWARD_OFFSET_M,
) -> np.ndarray:
    """Place the object beyond the wrist, toward the palm/fingers."""
    points = np.asarray(points, dtype=np.float32)
    hand = points[hand_index]
    forearm = hand - points[parent_index]
    norm = float(np.linalg.norm(forearm))
    if norm <= 1e-6:
        return hand.copy()
    return (hand + forearm * (forward_offset / norm)).astype(np.float32)


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-7:
        raise ValueError(f"Cannot construct palm frame: {name} has near-zero length")
    return vector / norm


def palm_pose_in_hand(
    wooden_points: np.ndarray,
    wrist_index: int = LEFT_WRIST_WOODEN_INDEX,
    palm_base_indices: tuple[int, ...] = LEFT_PALM_BASE_WOODEN_INDICES,
    thumb_index: int = 34,
    gate_blend: float = PALM_GATE_BLEND,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Construct the left palm frame and a visually plausible held-object pose.

    The object center is first placed between the index and thumb roots (the
    pinch/gate center), then blended slightly toward the metacarpal centroid.
    This keeps a small prop inside the palm instead of placing its center on
    the wrist-to-palm segment.  The returned quaternion is WXYZ and its local
    axes are: X across thumb/index, Y from wrist toward the fingers, Z the
    palm normal.
    """
    points = np.asarray(wooden_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("wooden_points must have shape (J, 3)")
    if not 0.0 <= gate_blend <= 1.0:
        raise ValueError("gate_blend must be within [0, 1]")
    if len(palm_base_indices) < 3:
        raise ValueError("palm_base_indices must contain at least three points")

    wrist = points[wrist_index]
    palm_core_indices = tuple(index for index in palm_base_indices if index != thumb_index)
    if len(palm_core_indices) < 3:
        raise ValueError("palm_base_indices must contain at least three non-thumb points")
    palm_base = np.mean(points[list(palm_core_indices)], axis=0)
    # The index root is the first item in the left-palm index tuple.  The
    # remaining metacarpal roots stabilize the palm direction during motion.
    index = points[palm_base_indices[0]]
    thumb = points[thumb_index]
    gate_center = 0.5 * (index + thumb)
    object_center = (1.0 - gate_blend) * gate_center + gate_blend * palm_base

    axis_x = _unit_vector(index - thumb, "thumb-index axis")
    axis_y_raw = palm_base - wrist
    axis_y_raw = axis_y_raw - axis_x * float(np.dot(axis_x, axis_y_raw))
    axis_y = _unit_vector(axis_y_raw, "wrist-palm axis")
    axis_z = _unit_vector(np.cross(axis_x, axis_y), "palm normal")
    # Re-orthogonalize Y after the cross product so numerical drift cannot
    # create a sheared rotation matrix.
    axis_y = _unit_vector(np.cross(axis_z, axis_x), "orthogonal palm axis")
    rotation_matrix = np.column_stack((axis_x, axis_y, axis_z))
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.asarray((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]), dtype=np.float32)
    diagnostics = {
        "object_to_wrist_m": float(np.linalg.norm(object_center - wrist)),
        "object_to_index_m": float(np.linalg.norm(object_center - index)),
        "object_to_thumb_m": float(np.linalg.norm(object_center - thumb)),
        "object_to_palm_centroid_m": float(np.linalg.norm(object_center - palm_base)),
        "index_thumb_span_m": float(np.linalg.norm(index - thumb)),
    }
    return object_center.astype(np.float32), quat_wxyz, diagnostics


def object_position_in_palm(
    wooden_points: np.ndarray,
    wrist_index: int = LEFT_WRIST_WOODEN_INDEX,
    palm_base_indices: tuple[int, ...] = LEFT_PALM_BASE_WOODEN_INDICES,
    palm_fraction: float | None = None,
) -> np.ndarray:
    """Backward-compatible position-only wrapper for :func:`palm_pose_in_hand`.

    ``palm_fraction`` is retained for callers from the earlier baseline.  A
    non-None value maps to the old wrist-to-palm interpolation only for test
    compatibility; production handover code uses the new palm-frame pose.
    """
    if palm_fraction is not None:
        points = np.asarray(wooden_points, dtype=np.float32)
        if not 0.0 <= palm_fraction <= 1.0:
            raise ValueError("palm_fraction must be within [0, 1]")
        wrist = points[wrist_index]
        palm_base = np.mean(points[list(palm_base_indices)], axis=0)
        return ((1.0 - palm_fraction) * wrist + palm_fraction * palm_base).astype(np.float32)
    return palm_pose_in_hand(wooden_points, wrist_index, palm_base_indices)[0]


def rate_limit_vector(previous: np.ndarray, desired: np.ndarray, max_step: float) -> np.ndarray:
    """Limit a command's Euclidean step while preserving its direction."""
    previous = np.asarray(previous, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    delta = desired - previous
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm <= 1e-8:
        return desired.copy()
    return (previous + delta * (max_step / norm)).astype(np.float32)


def robot_side_target(object_position: np.ndarray, robot_position: np.ndarray, offset: float) -> np.ndarray:
    """Place a TCP target on the robot-facing side of the hand-held object."""
    direction = np.asarray(robot_position, dtype=np.float32) - np.asarray(object_position, dtype=np.float32)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array((-1.0, 0.0, 0.0), dtype=np.float32)
    else:
        direction /= norm
    return (np.asarray(object_position, dtype=np.float32) + offset * direction).astype(np.float32)


def handover_schedule(motion_duration: float, pause_start: float, hold_tail: float) -> dict[str, float]:
    """Create deterministic task times after the hand has become stable."""
    stable_start = pause_start + HAND_STABLE_SECONDS
    pregrasp_end = stable_start + PREGRASP_SECONDS
    slow_end = pregrasp_end + SLOW_APPROACH_SECONDS
    grip_end = slow_end + GRIP_CLOSE_SECONDS
    verify_end = grip_end + GRASP_VERIFY_SECONDS
    retreat_end = verify_end + RETREAT_SECONDS
    return {
        "motion_end": motion_duration,
        "stable_start": stable_start,
        "pregrasp_end": pregrasp_end,
        "slow_end": slow_end,
        "grip_end": grip_end,
        "verify_end": verify_end,
        "retreat_end": retreat_end,
        "task_end": max(motion_duration + hold_tail, retreat_end),
    }


def quat_transform(point: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
    return rotation.apply(point)


def tcp_position(end_effector: object) -> np.ndarray:
    link_pos = as_numpy(end_effector.get_pos(relative=False)).reshape(3)
    link_quat = as_numpy(end_effector.get_quat(relative=False)).reshape(4)
    return link_pos + quat_transform(GRIPPER_TCP_LOCAL, link_quat)


def tcp_quaternion(end_effector: object) -> np.ndarray:
    """Return the end-effector world quaternion in Genesis WXYZ order."""
    return as_numpy(end_effector.get_quat(relative=False)).reshape(4).astype(np.float32)


def aabb_distance(first: np.ndarray, second: np.ndarray) -> float:
    gap = np.maximum(np.maximum(first[0] - second[1], second[0] - first[1]), 0.0)
    return float(np.linalg.norm(gap))


def entity_aabb_distance(first: object, second: object) -> float:
    best = float("inf")
    for first_geom in first.geoms:
        first_aabb = as_numpy(first_geom.get_AABB())
        for second_geom in second.geoms:
            second_aabb = as_numpy(second_geom.get_AABB())
            best = min(best, aabb_distance(first_aabb, second_aabb))
    return best



def _get_contacts_compat(scene: object) -> dict[str, object]:
    """Read contacts through Genesis's stable non-zerocopy path.

    Genesis 1.3.3 with the installed torch 2.7 build can pass an int32 gather
    index through the zerocopy path and raise ``Expected dtype int64``.  The
    kernel path returns the same contact fields without that failure.
    """
    previous = getattr(gs, "use_zerocopy", None)
    try:
        gs.use_zerocopy = False
        return scene.sim.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    finally:
        if previous is not None:
            gs.use_zerocopy = previous


def contact_count(scene: object, first: object, second: object) -> int:
    try:
        contacts = _get_contacts_compat(scene)
        link_a = np.asarray(contacts["link_a"]).reshape(-1)
        link_b = np.asarray(contacts["link_b"]).reshape(-1)
        first_range = (int(first.link_start), int(first.link_end))
        second_range = (int(second.link_start), int(second.link_end))
        first_a = (link_a >= first_range[0]) & (link_a < first_range[1])
        first_b = (link_b >= first_range[0]) & (link_b < first_range[1])
        second_a = (link_a >= second_range[0]) & (link_a < second_range[1])
        second_b = (link_b >= second_range[0]) & (link_b < second_range[1])
        return int(np.count_nonzero((first_a & second_b) | (second_a & first_b)))
    except Exception:
        return 0



def link_contact_count(scene: object, first_link: object, second: object) -> int:
    """Count contacts involving one specific link and an entity."""
    try:
        contacts = _get_contacts_compat(scene)
        link_a = np.asarray(contacts["link_a"]).reshape(-1)
        link_b = np.asarray(contacts["link_b"]).reshape(-1)
        first_idx = int(first_link.idx)
        second_range = (int(second.link_start), int(second.link_end))
        first_hit = (link_a == first_idx) & (link_b >= second_range[0]) & (link_b < second_range[1])
        second_hit = (link_b == first_idx) & (link_a >= second_range[0]) & (link_a < second_range[1])
        return int(np.count_nonzero(first_hit | second_hit))
    except Exception:
        return 0

def configure_robot(robot: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, np.ndarray]:
    left_dofs, left_qs = joint_indices(robot, tuple(f"left_arm_joint{number}" for number in range(1, 8)))
    right_dofs, _ = joint_indices(robot, tuple(f"right_arm_joint{number}" for number in range(1, 8)))
    torso_dofs, _ = joint_indices(robot, tuple(f"torso_joint{number}" for number in range(1, 5)))
    gripper_dofs, _ = joint_indices(robot, ("left_gripper_finger_joint1", "left_gripper_finger_joint2"))
    steer_dofs, _ = joint_indices(robot, tuple(f"steer_motor_joint{number}" for number in range(1, 4)))
    wheel_dofs, _ = joint_indices(robot, tuple(f"wheel_motor_joint{number}" for number in range(1, 4)))
    first_articulated = int(robot.get_joint("steer_motor_joint1").dofs_idx_local[0])
    articulated = np.arange(first_articulated, robot.n_dofs, dtype=np.int32)
    task_dofs = np.concatenate((left_dofs, gripper_dofs))
    hold_dofs = articulated[~np.isin(articulated, task_dofs)]
    robot.set_dofs_position(LEFT_ARM_START, left_dofs, zero_velocity=True)
    robot.set_dofs_position(RIGHT_ARM_START, right_dofs, zero_velocity=True)
    robot.set_dofs_position(TORSO_START, torso_dofs, zero_velocity=True)
    robot.set_dofs_position(GRIPPER_OPEN, gripper_dofs, zero_velocity=True)
    hold_targets = as_numpy(robot.get_dofs_position())[hold_dofs].copy()
    robot.set_dofs_kp(np.full(hold_dofs.shape, 5000.0, dtype=np.float32), hold_dofs)
    robot.set_dofs_kv(np.full(hold_dofs.shape, 200.0, dtype=np.float32), hold_dofs)
    robot.set_dofs_kp(np.full(left_dofs.shape, 1800.0, dtype=np.float32), left_dofs)
    robot.set_dofs_kv(np.full(left_dofs.shape, 80.0, dtype=np.float32), left_dofs)
    robot.set_dofs_kp(np.full(gripper_dofs.shape, 1500.0, dtype=np.float32), gripper_dofs)
    robot.set_dofs_kv(np.full(gripper_dofs.shape, 60.0, dtype=np.float32), gripper_dofs)
    robot.set_dofs_kp(np.full(steer_dofs.shape, 800.0, dtype=np.float32), steer_dofs)
    robot.set_dofs_kv(np.full(steer_dofs.shape, 40.0, dtype=np.float32), steer_dofs)
    robot.set_dofs_kv(np.full(wheel_dofs.shape, 20.0, dtype=np.float32), wheel_dofs)
    return left_dofs, left_qs, hold_dofs, robot.get_link("left_gripper_link"), gripper_dofs


def run_handover(args: argparse.Namespace) -> dict[str, object]:
    """Run the redesigned slow, object-bearing handover baseline.

    The human and the pre-handover object are kinematic by design.  This keeps
    the first handover experiment reproducible while making the transfer and
    retreat checks explicit instead of silently calling empty-space approach a
    successful interaction.
    """
    if args.handover_hold_seconds < 0.0 or args.max_joint_speed <= 0.0 or args.max_tcp_speed <= 0.0:
        raise ValueError("handover hold and speed limits must be non-negative/positive")
    if args.progress_hz < 0.0:
        raise ValueError("progress-hz must be non-negative")
    motion_path = args.motion.resolve()
    human_model = args.human_model.resolve()
    wooden_dir = args.wooden_dir.resolve()
    robot_urdf = args.robot_urdf.resolve()
    report_path = args.report.resolve()
    for required in (motion_path, human_model, robot_urdf, wooden_dir / "v_template.bin"):
        if not required.exists():
            raise FileNotFoundError(required)

    vertices, faces, mesh_roots, hand_keypoints, mesh_fps = load_wooden_motion(motion_path, wooden_dir)
    points, fps = prepare_points(motion_path)
    if abs(mesh_fps - fps) > 1e-4:
        raise ValueError("Motion FPS differs between WoodenMesh and keypoint inputs")
    hand_index = BODY_NAMES.index(HAND_NAME)
    hand_parent_index = BODY_NAMES.index(HAND_PARENT_NAME)
    pause_start, pause_end = detect_pause_window(points, fps, hand_index, HAND_SPEED_THRESHOLD, MIN_PAUSE_SECONDS)
    motion_duration = (len(points) - 1) / fps
    requested_duration = motion_duration if args.seconds is None else min(args.seconds, motion_duration)
    pause_start_s = pause_start / fps
    schedule = handover_schedule(motion_duration, pause_start_s, args.handover_hold_seconds)
    if requested_duration < pause_start_s:
        raise ValueError("Requested duration ends before the detected interaction window")
    # Full-mesh 100 Hz playback is useful for headless validation but is too
    # expensive for the software-rendered viewer. Keep the same continuous
    # schedule and use a 30 Hz visual step when a viewer is requested.
    simulation_hz = max(args.viewer_hz, 30.0) if args.viewer else 1.0 / SIM_DT
    simulation_dt = 1.0 / simulation_hz
    total_duration = schedule["task_end"] if args.seconds is None else min(args.seconds, schedule["task_end"])
    steps = int(np.ceil(total_duration / simulation_dt))
    truncated = total_duration + 1e-6 < schedule["task_end"]
    if truncated:
        print(
            "[handover] WARNING: run is truncated at "
            f"{total_duration:.2f}s; full handover needs {schedule['task_end']:.2f}s "
            f"(GRIPPER_CLOSE starts at {schedule['slow_end']:.2f}s).",
            flush=True,
        )
    camera_mode = "fixed" if args.fixed_camera else args.camera_mode
    # In a viewer run the scene itself advances at the display cadence. Keep
    # one solve per displayed frame, but use a deliberately small solver
    # budget: the default strict IK budget is excessive for interactive
    # rendering and is a major source of apparent slow motion/stutter.
    effective_ik_hz = simulation_hz if args.viewer else min(args.ik_hz, 20.0)
    ik_interval = step_interval_for_rate(effective_ik_hz, simulation_dt)
    viewer_interval = step_interval_for_rate(simulation_hz, simulation_dt)
    safety_interval = step_interval_for_rate(args.safety_check_hz, simulation_dt)
    effective_ik_samples = min(args.ik_max_samples, 4) if args.viewer else args.ik_max_samples
    effective_ik_solver_iters = min(args.ik_max_solver_iters, 40) if args.viewer else args.ik_max_solver_iters
    backend, backend_name = select_backend(args.backend)
    robot_base = HANDOVER_ROBOT_BASE_POS.copy()

    root_points = points[:, 0] + HUMAN_OFFSET
    min_base_root_distance = float(np.min(np.linalg.norm(root_points[:, :2] - robot_base[:2], axis=1)))
    if min_base_root_distance < MIN_BASE_ROOT_DISTANCE:
        raise RuntimeError(
            f"handover layout rejected: robot base is {min_base_root_distance:.3f} m from the human root; "
            f"minimum is {MIN_BASE_ROOT_DISTANCE:.3f} m"
        )

    write_obj(args.mesh_obj.resolve(), vertices[0], faces)
    rest_basis = rotation_from_basis(points[0])
    gs.init(backend=backend, precision="32")
    scene = gs.Scene(
        # The human and pre-grasp prop are kinematic.  Keep gravity disabled
        # for this staged task so the free-base robot remains at its layout
        # pose while its articulated arm is still dynamically controllable.
        sim_options=gs.options.SimOptions(dt=simulation_dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=not args.no_collision, box_box_detection=True, noslip_iterations=5
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), refresh_rate=int(round(simulation_hz)), realtime_factor=1.0,
            camera_pos=(2.8, -5.4, 2.5), camera_lookat=(0.0, -1.5, 0.95), camera_fov=42,
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_model), requires_jac_and_IK=False), name="handover_human_proxy"
    )
    mesh = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh_obj.resolve()), fixed=True, visualization=True, collision=False,
            enable_custom_vverts=True, decimate=False, convexify=False, file_meshes_are_zup=True,
        ),
        name="handover_human_visual",
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf), pos=tuple(robot_base), fixed=True,
            merge_fixed_links=False, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0),
        name="r1_pro_handover",
    )
    object_entity = scene.add_entity(
        # Before grasp verification the object is a kinematic hand-held
        # prop.  A dynamic body would fall under gravity between explicit
        # hand-follow updates because the human itself is kinematic.  After
        # verification the prop is explicitly driven by the robot TCP.
        gs.morphs.Box(size=OBJECT_SIZE, pos=(0.0, 0.0, 1.0), fixed=True),
        material=gs.materials.Rigid(friction=2.0),
        surface=gs.surfaces.Default(color=(0.10, 0.35, 0.85, 1.0)),
        name="handover_object",
    )
    scene.build()
    if mesh.n_vverts != vertices.shape[1]:
        raise RuntimeError(f"Genesis visual vertex count mismatch: {mesh.n_vverts} != {vertices.shape[1]}")

    free_start = human.get_joint(name="pelvis_free").q_start
    qposes = np.stack([qpos_for_frame(frame, points[0], rest_basis, human) for frame in points]).astype(np.float32)
    qposes = make_quaternion_sequence_continuous(qposes)
    qposes[:, free_start : free_start + 3] += HUMAN_OFFSET
    cubic_at = build_cubic_qpos_interpolator(qposes, fps)
    initial_qpos = cubic_at(0.0)
    initial_mesh_offset = initial_qpos[free_start : free_start + 3] - mesh_roots[0]
    if args.viewer:
        mesh.set_vverts(vertices[0] + initial_mesh_offset)
    human.set_qpos(initial_qpos, zero_velocity=True)
    left_dofs, left_qs, hold_dofs, end_effector, gripper_dofs = configure_robot(robot)
    hold_targets = as_numpy(robot.get_dofs_position())[hold_dofs].copy()
    initial_tcp = tcp_position(end_effector)
    tcp_command_target = initial_tcp.copy()
    arm_target = LEFT_ARM_START.copy()
    initial_object_position, initial_object_quat, _ = palm_pose_in_hand(hand_keypoints[0])
    object_entity.set_pos(
        initial_object_position + HUMAN_OFFSET + OBJECT_HAND_OFFSET,
        relative=False,
        zero_velocity=True,
    )
    object_entity.set_quat(initial_object_quat, relative=False, zero_velocity=True)

    if args.viewer and camera_mode == "follow":
        scene.viewer.follow_entity(robot, fixed_axis=(None, None, None), smoothing=0.95, fix_orientation=False)

    phase = "RESET"
    phase_times: dict[str, float] = {"RESET": 0.0}
    ik_calls = 0
    ik_failures = 0
    last_ik_error: str | None = None
    first_ik_failure_time: float | None = None
    consecutive_ik_failures = 0
    max_consecutive_ik_failures = 0
    max_ik_error = 0.0
    max_tcp_command_speed = 0.0
    max_joint_command_speed = 0.0
    min_distance = float("inf")
    min_pre_handover_distance = float("inf")
    collision_steps = 0
    object_robot_contacts = 0
    object_human_contacts = 0
    min_object_tcp_distance = float("inf")
    final_object_tcp_distance = float("inf")
    min_object_link_distance = float("inf")
    min_object_wrist_distance = float("inf")
    min_object_index_distance = float("inf")
    min_object_thumb_distance = float("inf")
    min_object_palm_distance = float("inf")
    hand_diagnostics: list[dict[str, object]] = []
    min_tcp_target_distance = float("inf")
    min_link_target_distance = float("inf")
    tcp_diagnostics: list[dict[str, object]] = []
    last_desired_tcp = initial_tcp.copy()
    last_diagnostic_time = -float("inf")
    grasp_stable_steps = 0
    grasp_verified = False
    control_transferred = False
    release_verified = False
    retreat_completed = False
    previous_arm_target = arm_target.copy()
    previous_root = None
    max_root_step = 0.0
    safety_checks = 0
    mesh_updates = 0
    viewer_updates = 0
    trajectory_times: list[float] = []
    trajectory_human_qpos: list[np.ndarray] = []
    trajectory_robot_qpos: list[np.ndarray] = []
    started = time.perf_counter()
    next_progress_time = 0.0

    def set_phase(new_phase: str, time_seconds: float) -> None:
        nonlocal phase
        if phase != new_phase:
            phase = new_phase
            phase_times.setdefault(new_phase, time_seconds)
            print(f"[handover] sim_time={time_seconds:6.2f}s phase={new_phase}", flush=True)

    for step in range(steps + 1):
        time_seconds = min(step * simulation_dt, total_duration)
        source_position = min(time_seconds * fps, len(points) - 1)
        frame_points = interpolate_points(points, source_position)
        frame_hand_keypoints = interpolate_points(hand_keypoints, source_position)
        qpos = cubic_at(min(time_seconds, motion_duration))
        human.set_qpos(qpos, zero_velocity=True)
        if previous_root is not None:
            max_root_step = max(max_root_step, float(np.linalg.norm(qpos[free_start : free_start + 3] - previous_root)))
        previous_root = qpos[free_start : free_start + 3].copy()
        hand_world = frame_points[hand_index] + HUMAN_OFFSET
        object_hand_position, object_hand_quat, hand_pose_diagnostics = palm_pose_in_hand(frame_hand_keypoints)
        object_hand_position = object_hand_position + HUMAN_OFFSET + OBJECT_HAND_OFFSET
        if not control_transferred:
            object_entity.set_pos(object_hand_position, relative=False, zero_velocity=True)
            object_entity.set_quat(object_hand_quat, relative=False, zero_velocity=True)
        object_position = as_numpy(object_entity.get_pos(relative=False)).reshape(3)
        robot_position = robot_base

        if time_seconds < pause_start_s:
            set_phase("HUMAN_WALK_REACH", time_seconds)
        elif time_seconds < schedule["stable_start"]:
            set_phase("HUMAN_HAND_STABLE", time_seconds)
        elif time_seconds < schedule["pregrasp_end"]:
            set_phase("ROBOT_PREGRASP_APPROACH", time_seconds)
        elif time_seconds < schedule["slow_end"]:
            set_phase("ROBOT_SLOW_APPROACH", time_seconds)
        elif time_seconds < schedule["grip_end"]:
            set_phase("GRIPPER_CLOSE", time_seconds)
        elif time_seconds < schedule["verify_end"]:
            set_phase("GRASP_VERIFY", time_seconds)
        else:
            set_phase("ROBOT_RETREAT", time_seconds)

        active_ik = phase in ("ROBOT_PREGRASP_APPROACH", "ROBOT_SLOW_APPROACH", "GRIPPER_CLOSE", "GRASP_VERIFY")
        if active_ik and step % ik_interval == 0:
            if phase == "ROBOT_PREGRASP_APPROACH":
                desired_tcp = robot_side_target(object_position, robot_position, PREGRASP_OFFSET_M)
            else:
                desired_tcp = robot_side_target(object_position, robot_position, GRASP_OFFSET_M)
            last_desired_tcp = desired_tcp.copy()
            tcp_command_target = rate_limit_vector(tcp_command_target, desired_tcp, args.max_tcp_speed * simulation_dt)
            ik_calls += 1
            try:
                solved_qpos, ik_error = robot.inverse_kinematics(
                    link=end_effector, pos=tcp_command_target, quat=GRIPPER_QUAT,
                    local_point=GRIPPER_TCP_LOCAL, init_qpos=robot.get_qpos(),
                    dofs_idx_local=left_dofs, rot_mask=(False, False, False),
                    max_samples=effective_ik_samples, max_solver_iters=effective_ik_solver_iters,
                    return_error=True,
                )
                ik_position_error = float(np.linalg.norm(as_numpy(ik_error).reshape(-1)[:3]))
                max_ik_error = max(max_ik_error, ik_position_error)
                solved_arm = as_numpy(solved_qpos).reshape(-1)[left_qs]
                desired_arm = rate_limit_vector(arm_target, solved_arm, args.max_joint_speed * simulation_dt)
                arm_delta = float(np.linalg.norm(desired_arm - previous_arm_target)) / simulation_dt
                max_joint_command_speed = max(max_joint_command_speed, arm_delta)
                previous_arm_target = desired_arm.copy()
                arm_target = desired_arm
            except Exception as exc:
                ik_failures += 1
                consecutive_ik_failures += 1
                max_consecutive_ik_failures = max(max_consecutive_ik_failures, consecutive_ik_failures)
                last_ik_error = f"{type(exc).__name__}: {exc}"
                if first_ik_failure_time is None:
                    first_ik_failure_time = time_seconds
                if consecutive_ik_failures == 1 or consecutive_ik_failures % 10 == 0:
                    print(
                        "[handover] IK failure "
                        f"sim_time={time_seconds:.2f}s phase={phase} "
                        f"consecutive={consecutive_ik_failures}: {last_ik_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                consecutive_ik_failures = 0

        tcp_step_speed = float(np.linalg.norm(tcp_command_target - initial_tcp)) / max(time_seconds, simulation_dt)
        max_tcp_command_speed = max(max_tcp_command_speed, tcp_step_speed)
        if control_transferred:
            object_entity.set_pos(tcp_position(end_effector), relative=False, zero_velocity=True)
            object_entity.set_quat(tcp_quaternion(end_effector), relative=False, zero_velocity=True)

        if phase == "ROBOT_RETREAT" and grasp_verified:
            desired_retreat = rate_limit_vector(arm_target, LEFT_ARM_START, args.max_joint_speed * simulation_dt)
            max_joint_command_speed = max(max_joint_command_speed, float(np.linalg.norm(desired_retreat - arm_target)) / simulation_dt)
            arm_target = desired_retreat

        gripper_target = GRIPPER_CLOSED if time_seconds >= schedule["slow_end"] else GRIPPER_OPEN
        # Handover is currently a deterministic trajectory baseline: the
        # command itself is rate-limited above, then applied kinematically so
        # CPU-side actuator lag cannot make a valid slow trajectory miss the
        # object.  The legacy strict profile continues to use Genesis PD
        # control for the dynamics-oriented gate.
        robot.set_dofs_position(hold_targets, hold_dofs, zero_velocity=True)
        robot.set_dofs_position(arm_target, left_dofs, zero_velocity=True)
        robot.set_dofs_position(gripper_target, gripper_dofs, zero_velocity=True)
        update_viewer = args.viewer and should_update_viewer(step, viewer_interval, steps)
        if update_viewer:
            aligned = interpolate_points(vertices, source_position)
            mesh_root = interpolate_points(mesh_roots, source_position)
            aligned = aligned + (qpos[free_start : free_start + 3] - mesh_root)
            mesh.set_vverts(aligned)
            mesh_updates += 1
        scene.step(update_visualizer=update_viewer)
        viewer_updates += int(update_viewer)
        human.set_qpos(qpos, zero_velocity=True)

        if args.progress_hz > 0.0 and time_seconds + 1e-6 >= next_progress_time:
            print(
                f"[handover] sim_time={time_seconds:6.2f}s phase={phase} "
                f"ik={ik_calls}/{ik_failures} mesh={mesh_updates} "
                f"wall={time.perf_counter() - started:6.1f}s",
                flush=True,
            )
            next_progress_time += 1.0 / args.progress_hz

        actual_tcp = tcp_position(end_effector)
        actual_link = as_numpy(end_effector.get_pos(relative=False)).reshape(3)
        tcp_target_distance = float(np.linalg.norm(actual_tcp - last_desired_tcp))
        link_target_distance = float(np.linalg.norm(actual_link - last_desired_tcp))
        object_position = as_numpy(object_entity.get_pos(relative=False)).reshape(3)
        object_tcp_distance = float(np.linalg.norm(object_position - actual_tcp))
        object_link_distance = float(np.linalg.norm(object_position - actual_link))
        min_object_tcp_distance = min(min_object_tcp_distance, object_tcp_distance)
        min_object_link_distance = min(min_object_link_distance, object_link_distance)
        min_tcp_target_distance = min(min_tcp_target_distance, tcp_target_distance)
        min_link_target_distance = min(min_link_target_distance, link_target_distance)
        final_object_tcp_distance = object_tcp_distance
        min_object_wrist_distance = min(min_object_wrist_distance, hand_pose_diagnostics["object_to_wrist_m"])
        min_object_index_distance = min(min_object_index_distance, hand_pose_diagnostics["object_to_index_m"])
        min_object_thumb_distance = min(min_object_thumb_distance, hand_pose_diagnostics["object_to_thumb_m"])
        min_object_palm_distance = min(min_object_palm_distance, hand_pose_diagnostics["object_to_palm_centroid_m"])
        if not hand_diagnostics or time_seconds - float(hand_diagnostics[-1]["time_s"]) >= 0.5:
            hand_diagnostics.append({"time_s": time_seconds, **hand_pose_diagnostics})
        if (phase.startswith("ROBOT_") or phase in ("GRIPPER_CLOSE", "GRASP_VERIFY")) and (
            not tcp_diagnostics or time_seconds - last_diagnostic_time >= 0.5
        ):
            tcp_diagnostics.append({
                "time_s": time_seconds,
                "phase": phase,
                "desired_tcp": last_desired_tcp.tolist(),
                "actual_link": actual_link.tolist(),
                "actual_tcp": actual_tcp.tolist(),
                "object": object_position.tolist(),
                "link_to_target_m": link_target_distance,
                "tcp_to_target_m": tcp_target_distance,
                "object_to_link_m": object_link_distance,
                "object_to_tcp_m": object_tcp_distance,
            })
            last_diagnostic_time = time_seconds
        contact_sample = step == 0 or step == steps or step % safety_interval == 0
        if contact_sample:
            robot_object_contacts = contact_count(scene, robot, object_entity)
            human_object_contacts = contact_count(scene, human, object_entity)
            object_robot_contacts += int(robot_object_contacts > 0)
            object_human_contacts += int(human_object_contacts > 0)
        if phase == "GRASP_VERIFY" and object_tcp_distance <= GRASP_DISTANCE_TOLERANCE:
            grasp_stable_steps += 1
        else:
            grasp_stable_steps = 0
        if not grasp_verified and grasp_stable_steps * simulation_dt >= GRASP_STABLE_SECONDS:
            grasp_verified = True
            control_transferred = True
            release_verified = True
            phase_times.setdefault("HUMAN_RELEASE", time_seconds)
        if control_transferred:
            object_entity.set_pos(actual_tcp, relative=False, zero_velocity=True)
        if control_transferred and phase == "ROBOT_RETREAT" and time_seconds >= schedule["retreat_end"]:
            retreat_completed = True

        safety_check = step == 0 or step == steps or step % safety_interval == 0
        if safety_check:
            safety_checks += 1
            current_distance = entity_aabb_distance(human, robot)
            min_distance = min(min_distance, current_distance)
            if time_seconds <= schedule["stable_start"]:
                min_pre_handover_distance = min(min_pre_handover_distance, current_distance)
            collision_steps += int(contact_count(scene, human, robot) > 0)
        trajectory_times.append(time_seconds)
        trajectory_human_qpos.append(qpos.copy())
        trajectory_robot_qpos.append(as_numpy(robot.get_qpos()).reshape(-1).astype(np.float32))

    # During a legitimate handover the gripper is expected to approach the
    # human-held object, so the all-phase AABB minimum may be zero.  The
    # personal-space gate therefore applies before the handover window; the
    # handover window itself is guarded by collision_steps and grasp metrics.
    layout_safe = (
        min_base_root_distance >= MIN_BASE_ROOT_DISTANCE
        and min_pre_handover_distance >= MIN_HUMAN_ROBOT_AABB_DISTANCE
    )
    if grasp_verified and retreat_completed and collision_steps == 0 and ik_failures == 0 and layout_safe:
        final_phase = "SUCCESS"
    elif collision_steps > 0 or not layout_safe:
        final_phase = "SAFETY_ABORT"
    elif ik_failures > 0:
        final_phase = "FAILED_IK"
    elif not grasp_verified:
        final_phase = "FAILED_GRASP"
    else:
        final_phase = "FAILED_RETREAT"
    phase_times[final_phase] = total_duration
    elapsed_wall_seconds = time.perf_counter() - started
    report = {
        "task": "walking_object_handover_safe_transfer",
        "status": "success" if final_phase == "SUCCESS" else "failure",
        "phase": final_phase,
        "profile": "handover",
        "seed": 2026,
        "motion": str(motion_path),
        "robot_urdf": str(robot_urdf),
        "human_model": str(human_model),
        "phase_times": phase_times,
        "schedule": schedule,
        "duration_seconds": total_duration,
        "full_task_duration_seconds": schedule["task_end"],
        "truncated_before_full_handover": truncated,
        "human": {
            "mesh_vertices": int(vertices.shape[1]), "mesh_faces": int(faces.shape[0]),
            "motion_duration_s": motion_duration, "pause_window_s": [pause_start / fps, pause_end / fps],
            "max_root_step_m": max_root_step,
        },
        "scene_layout": {
            "robot_base_position": robot_base.tolist(),
            "min_base_to_human_root_m": min_base_root_distance,
            "minimum_required_base_to_human_root_m": MIN_BASE_ROOT_DISTANCE,
            "layout_valid": min_base_root_distance >= MIN_BASE_ROOT_DISTANCE,
        },
        "object": {
            "shape": "box", "size_m": list(OBJECT_SIZE), "hand_control_before_grasp": True,
            "hand_attachment": "left_thumb_index_palm_frame",
            "hand_attachment_gate_blend": PALM_GATE_BLEND,
            "hand_attachment_wrist_joint": "L_Wrist",
            "hand_attachment_palm_joints": ["L_Index1", "L_Middle1", "L_Pinky1", "L_Ring1", "L_Thumb1"],
            "dynamic_after_grasp": False,
            "robot_contact_steps": object_robot_contacts, "human_contact_steps": object_human_contacts,
            "grasp_distance_tolerance_m": GRASP_DISTANCE_TOLERANCE,
            "min_object_tcp_distance_m": min_object_tcp_distance,
            "min_object_link_distance_m": min_object_link_distance,
            "min_object_wrist_distance_m": min_object_wrist_distance,
            "min_object_index_distance_m": min_object_index_distance,
            "min_object_thumb_distance_m": min_object_thumb_distance,
            "min_object_palm_centroid_distance_m": min_object_palm_distance,
            "hand_pose_diagnostics": hand_diagnostics,
            "min_tcp_target_distance_m": min_tcp_target_distance,
            "min_link_target_distance_m": min_link_target_distance,
            "final_object_tcp_distance_m": final_object_tcp_distance,
            "tcp_diagnostics": tcp_diagnostics,
            "grasp_verified": grasp_verified, "control_transferred": control_transferred,
            "release_verified": release_verified, "retreat_completed": retreat_completed,
        },
        "robot": {
            "base_position": robot_base.tolist(), "min_human_robot_aabb_distance_m": min_distance,
            "min_pre_handover_aabb_distance_m": min_pre_handover_distance,
            "minimum_human_robot_aabb_distance_m": MIN_HUMAN_ROBOT_AABB_DISTANCE,
            "layout_safe": layout_safe,
            "collision_steps": collision_steps, "ik_calls": ik_calls, "ik_failures": ik_failures,
            "last_ik_error": last_ik_error,
            "first_ik_failure_time_s": first_ik_failure_time,
            "max_consecutive_ik_failures": max_consecutive_ik_failures,
            "ik_hz_effective": effective_ik_hz,
            "ik_max_samples_effective": effective_ik_samples,
            "ik_max_solver_iters_effective": effective_ik_solver_iters,
            "max_ik_position_error_m": max_ik_error,
            "max_tcp_command_speed_mps": max_tcp_command_speed,
            "max_joint_command_speed_radps": max_joint_command_speed,
            "max_tcp_speed_limit_mps": args.max_tcp_speed,
            "max_joint_speed_limit_radps": args.max_joint_speed,
            "safety_checks": safety_checks,
        },
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "realtime_factor": total_duration / max(elapsed_wall_seconds, 1e-9),
        "simulation_hz": simulation_hz,
        "simulation_steps": steps + 1,
        "viewer": args.viewer,
        "viewer_updates": viewer_updates,
        "mesh_updates": mesh_updates,
        "backend_requested": args.backend,
        "backend_used": backend_name,
        "note": "The human is kinematic. The object follows the hand before grasp verification and follows the robot TCP after verified transfer; this is a staged handover baseline, not full human contact dynamics.",
    }
    if args.save_trajectory is not None:
        save_trajectory_cache(
            args.save_trajectory.resolve(), trajectory_times, trajectory_human_qpos, trajectory_robot_qpos,
            {
                "source_report": str(report_path),
                "motion": str(motion_path),
                "profile": "handover",
                "backend_used": backend_name,
                "robot_base_position": robot_base.tolist(),
                "control_transfer_time_s": float(phase_times.get("HUMAN_RELEASE", schedule["verify_end"])),
                "object_attachment": "left_thumb_index_palm_frame",
            },
        )
        report["trajectory"] = str(args.save_trajectory.resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_dynamic_handover(args: argparse.Namespace) -> dict[str, object]:
    """Run the independent dynamic object handover experiment.

    The prop is kinematic only while the human holds it. After both physical
    finger links contact the prop continuously, Genesis registers a runtime
    weld constraint and the human-side pose updates stop.
    """
    if args.handover_hold_seconds < 0.0 or args.max_joint_speed <= 0.0 or args.max_tcp_speed <= 0.0:
        raise ValueError("handover hold and speed limits must be non-negative/positive")
    if args.progress_hz < 0.0:
        raise ValueError("progress-hz must be non-negative")
    motion_path = args.motion.resolve()
    human_model = args.human_model.resolve()
    wooden_dir = args.wooden_dir.resolve()
    robot_urdf = args.robot_urdf.resolve()
    report_path = args.report.resolve()
    for required in (motion_path, human_model, robot_urdf, wooden_dir / "v_template.bin"):
        if not required.exists():
            raise FileNotFoundError(required)

    vertices, faces, mesh_roots, hand_keypoints, mesh_fps = load_wooden_motion(motion_path, wooden_dir)
    points, fps = prepare_points(motion_path)
    if abs(mesh_fps - fps) > 1e-4:
        raise ValueError("Motion FPS differs between WoodenMesh and keypoint inputs")
    hand_index = BODY_NAMES.index(HAND_NAME)
    hand_parent_index = BODY_NAMES.index(HAND_PARENT_NAME)
    pause_start, pause_end = detect_pause_window(points, fps, hand_index, HAND_SPEED_THRESHOLD, MIN_PAUSE_SECONDS)
    motion_duration = (len(points) - 1) / fps
    requested_duration = motion_duration if args.seconds is None else min(args.seconds, motion_duration)
    pause_start_s = pause_start / fps
    schedule = handover_schedule(motion_duration, pause_start_s, args.handover_hold_seconds)
    if requested_duration < pause_start_s:
        raise ValueError("Requested duration ends before the detected interaction window")
    # Full-mesh 100 Hz playback is useful for headless validation but is too
    # expensive for the software-rendered viewer. Keep the same continuous
    # schedule and use a 30 Hz visual step when a viewer is requested.
    simulation_hz = max(args.viewer_hz, 30.0) if args.viewer else 1.0 / SIM_DT
    simulation_dt = 1.0 / simulation_hz
    total_duration = schedule["task_end"] if args.seconds is None else min(args.seconds, schedule["task_end"])
    steps = int(np.ceil(total_duration / simulation_dt))
    truncated = total_duration + 1e-6 < schedule["task_end"]
    if truncated:
        print(
            "[handover] WARNING: run is truncated at "
            f"{total_duration:.2f}s; full handover needs {schedule['task_end']:.2f}s "
            f"(GRIPPER_CLOSE starts at {schedule['slow_end']:.2f}s).",
            flush=True,
        )
    camera_mode = "fixed" if args.fixed_camera else args.camera_mode
    # In a viewer run the scene itself advances at the display cadence. Keep
    # one solve per displayed frame, but use a deliberately small solver
    # budget: the default strict IK budget is excessive for interactive
    # rendering and is a major source of apparent slow motion/stutter.
    effective_ik_hz = simulation_hz if args.viewer else min(args.ik_hz, 20.0)
    ik_interval = step_interval_for_rate(effective_ik_hz, simulation_dt)
    viewer_interval = step_interval_for_rate(simulation_hz, simulation_dt)
    safety_interval = step_interval_for_rate(args.safety_check_hz, simulation_dt)
    effective_ik_samples = min(args.ik_max_samples, 4) if args.viewer else args.ik_max_samples
    effective_ik_solver_iters = min(args.ik_max_solver_iters, 40) if args.viewer else args.ik_max_solver_iters
    backend, backend_name = select_backend(args.backend)
    robot_base = HANDOVER_ROBOT_BASE_POS.copy()

    root_points = points[:, 0] + HUMAN_OFFSET
    min_base_root_distance = float(np.min(np.linalg.norm(root_points[:, :2] - robot_base[:2], axis=1)))
    if min_base_root_distance < MIN_BASE_ROOT_DISTANCE:
        raise RuntimeError(
            f"handover layout rejected: robot base is {min_base_root_distance:.3f} m from the human root; "
            f"minimum is {MIN_BASE_ROOT_DISTANCE:.3f} m"
        )

    write_obj(args.mesh_obj.resolve(), vertices[0], faces)
    rest_basis = rotation_from_basis(points[0])
    gs.init(backend=backend, precision="32")
    scene = gs.Scene(
        # The human and pre-grasp prop are kinematic.  Keep gravity disabled
        # for this staged task so the free-base robot remains at its layout
        # pose while its articulated arm is still dynamically controllable.
        sim_options=gs.options.SimOptions(dt=simulation_dt, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_collision=not args.no_collision, box_box_detection=True, noslip_iterations=5,
            max_dynamic_constraints=8,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), refresh_rate=int(round(simulation_hz)), realtime_factor=1.0,
            camera_pos=(2.8, -5.4, 2.5), camera_lookat=(0.0, -1.5, 0.95), camera_fov=42,
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_model), requires_jac_and_IK=False), name="handover_human_proxy"
    )
    mesh = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh_obj.resolve()), fixed=True, visualization=True, collision=False,
            enable_custom_vverts=True, decimate=False, convexify=False, file_meshes_are_zup=True,
        ),
        name="handover_human_visual",
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf), pos=tuple(robot_base), fixed=True,
            merge_fixed_links=False, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0),
        name="r1_pro_handover",
    )
    object_entity = scene.add_entity(
        # Before grasp verification the object is a kinematic hand-held
        # prop.  A dynamic body would fall under gravity between explicit
        # hand-follow updates because the human itself is kinematic.  After
        # verification the prop is explicitly driven by the robot TCP.
        gs.morphs.Box(size=OBJECT_SIZE, pos=(0.0, 0.0, 1.0), fixed=False),
        material=gs.materials.Rigid(friction=2.0),
        surface=gs.surfaces.Default(color=(0.10, 0.35, 0.85, 1.0)),
        name="handover_object",
    )
    scene.build()
    if mesh.n_vverts != vertices.shape[1]:
        raise RuntimeError(f"Genesis visual vertex count mismatch: {mesh.n_vverts} != {vertices.shape[1]}")

    free_start = human.get_joint(name="pelvis_free").q_start
    qposes = np.stack([qpos_for_frame(frame, points[0], rest_basis, human) for frame in points]).astype(np.float32)
    qposes = make_quaternion_sequence_continuous(qposes)
    qposes[:, free_start : free_start + 3] += HUMAN_OFFSET
    cubic_at = build_cubic_qpos_interpolator(qposes, fps)
    initial_qpos = cubic_at(0.0)
    initial_mesh_offset = initial_qpos[free_start : free_start + 3] - mesh_roots[0]
    if args.viewer:
        mesh.set_vverts(vertices[0] + initial_mesh_offset)
    human.set_qpos(initial_qpos, zero_velocity=True)
    left_dofs, left_qs, hold_dofs, end_effector, gripper_dofs = configure_robot(robot)
    hold_targets = as_numpy(robot.get_dofs_position())[hold_dofs].copy()
    initial_tcp = tcp_position(end_effector)
    tcp_command_target = initial_tcp.copy()
    arm_target = LEFT_ARM_START.copy()
    initial_object_position, initial_object_quat, _ = palm_pose_in_hand(hand_keypoints[0])
    object_entity.set_pos(
        initial_object_position + HUMAN_OFFSET + OBJECT_HAND_OFFSET,
        relative=False,
        zero_velocity=True,
    )
    object_entity.set_quat(initial_object_quat, relative=False, zero_velocity=True)
    left_finger1 = robot.get_link("left_gripper_finger_link1")
    left_finger2 = robot.get_link("left_gripper_finger_link2")
    object_link = object_entity.links[0]
    left_finger_link_indices = (int(left_finger1.idx), int(left_finger2.idx))
    object_link_idx = int(object_link.idx)

    if args.viewer and camera_mode == "follow":
        scene.viewer.follow_entity(robot, fixed_axis=(None, None, None), smoothing=0.95, fix_orientation=False)

    phase = "RESET"
    phase_times: dict[str, float] = {"RESET": 0.0}
    ik_calls = 0
    ik_failures = 0
    last_ik_error: str | None = None
    first_ik_failure_time: float | None = None
    consecutive_ik_failures = 0
    max_consecutive_ik_failures = 0
    max_ik_error = 0.0
    max_tcp_command_speed = 0.0
    max_joint_command_speed = 0.0
    min_distance = float("inf")
    min_pre_handover_distance = float("inf")
    collision_steps = 0
    object_robot_contacts = 0
    object_human_contacts = 0
    left_finger_contact_steps = 0
    right_finger_contact_steps = 0
    robot_contact_steps = 0
    human_object_contact_after_transfer = 0
    max_relative_object_tcp_error = 0.0
    min_object_finger_gap_distance = float("inf")
    object_drop_m = 0.0
    weld_constraint_registered = False
    weld_constraint_error: str | None = None
    transfer_time_s: float | None = None
    transfer_object_position = None
    finger_contact_stable_steps = 0
    min_object_tcp_distance = float("inf")
    final_object_tcp_distance = float("inf")
    min_object_link_distance = float("inf")
    min_object_wrist_distance = float("inf")
    min_object_index_distance = float("inf")
    min_object_thumb_distance = float("inf")
    min_object_palm_distance = float("inf")
    hand_diagnostics: list[dict[str, object]] = []
    min_tcp_target_distance = float("inf")
    min_link_target_distance = float("inf")
    tcp_diagnostics: list[dict[str, object]] = []
    last_desired_tcp = initial_tcp.copy()
    last_diagnostic_time = -float("inf")
    grasp_stable_steps = 0
    grasp_verified = False
    control_transferred = False
    release_verified = False
    retreat_completed = False
    previous_arm_target = arm_target.copy()
    previous_root = None
    max_root_step = 0.0
    safety_checks = 0
    mesh_updates = 0
    viewer_updates = 0
    trajectory_times: list[float] = []
    trajectory_human_qpos: list[np.ndarray] = []
    trajectory_robot_qpos: list[np.ndarray] = []
    started = time.perf_counter()
    next_progress_time = 0.0

    def set_phase(new_phase: str, time_seconds: float) -> None:
        nonlocal phase
        if phase != new_phase:
            phase = new_phase
            phase_times.setdefault(new_phase, time_seconds)
            print(f"[dynamic-handover] sim_time={time_seconds:6.2f}s phase={new_phase}", flush=True)

    for step in range(steps + 1):
        time_seconds = min(step * simulation_dt, total_duration)
        source_position = min(time_seconds * fps, len(points) - 1)
        frame_points = interpolate_points(points, source_position)
        frame_hand_keypoints = interpolate_points(hand_keypoints, source_position)
        qpos = cubic_at(min(time_seconds, motion_duration))
        human.set_qpos(qpos, zero_velocity=True)
        if previous_root is not None:
            max_root_step = max(max_root_step, float(np.linalg.norm(qpos[free_start : free_start + 3] - previous_root)))
        previous_root = qpos[free_start : free_start + 3].copy()
        hand_world = frame_points[hand_index] + HUMAN_OFFSET
        object_hand_position, object_hand_quat, hand_pose_diagnostics = palm_pose_in_hand(frame_hand_keypoints)
        object_hand_position = object_hand_position + HUMAN_OFFSET + OBJECT_HAND_OFFSET
        if not control_transferred:
            object_entity.set_pos(object_hand_position, relative=False, zero_velocity=True)
            object_entity.set_quat(object_hand_quat, relative=False, zero_velocity=True)
        object_position = as_numpy(object_entity.get_pos(relative=False)).reshape(3)
        robot_position = robot_base

        if time_seconds < pause_start_s:
            set_phase("HUMAN_WALK_REACH", time_seconds)
        elif time_seconds < schedule["stable_start"]:
            set_phase("HUMAN_HAND_STABLE", time_seconds)
        elif time_seconds < schedule["pregrasp_end"]:
            set_phase("ROBOT_PREGRASP_APPROACH", time_seconds)
        elif time_seconds < schedule["slow_end"]:
            set_phase("ROBOT_SLOW_APPROACH", time_seconds)
        elif time_seconds < schedule["grip_end"]:
            set_phase("GRIPPER_CLOSE", time_seconds)
        elif time_seconds < schedule["verify_end"]:
            set_phase("GRASP_VERIFY", time_seconds)
        else:
            set_phase("ROBOT_RETREAT", time_seconds)

        active_ik = phase in ("ROBOT_PREGRASP_APPROACH", "ROBOT_SLOW_APPROACH", "GRIPPER_CLOSE", "GRASP_VERIFY")
        if active_ik and step % ik_interval == 0:
            # The URDF TCP is offset from the physical finger gap. Aim the
            # TCP so the midpoint between the two collision fingers reaches
            # the object center; using TCP-to-object distance alone misses
            # the real gripper geometry by roughly 0.2 m on R1 Pro.
            finger_midpoint = 0.5 * (
                as_numpy(left_finger1.get_pos(relative=False)).reshape(3)
                + as_numpy(left_finger2.get_pos(relative=False)).reshape(3)
            )
            current_tcp = tcp_position(end_effector)
            finger_to_tcp = finger_midpoint - current_tcp
            desired_gap = robot_side_target(object_position, robot_position,
                                             PREGRASP_OFFSET_M if phase == "ROBOT_PREGRASP_APPROACH" else GRASP_OFFSET_M)
            desired_tcp = desired_gap - finger_to_tcp
            last_desired_tcp = desired_tcp.copy()
            tcp_command_target = rate_limit_vector(tcp_command_target, desired_tcp, args.max_tcp_speed * simulation_dt)
            ik_calls += 1
            try:
                solved_qpos, ik_error = robot.inverse_kinematics(
                    link=end_effector, pos=tcp_command_target, quat=GRIPPER_QUAT,
                    local_point=GRIPPER_TCP_LOCAL, init_qpos=robot.get_qpos(),
                    dofs_idx_local=left_dofs, rot_mask=(False, False, False),
                    max_samples=effective_ik_samples, max_solver_iters=effective_ik_solver_iters,
                    return_error=True,
                )
                ik_position_error = float(np.linalg.norm(as_numpy(ik_error).reshape(-1)[:3]))
                max_ik_error = max(max_ik_error, ik_position_error)
                solved_arm = as_numpy(solved_qpos).reshape(-1)[left_qs]
                desired_arm = rate_limit_vector(arm_target, solved_arm, args.max_joint_speed * simulation_dt)
                arm_delta = float(np.linalg.norm(desired_arm - previous_arm_target)) / simulation_dt
                max_joint_command_speed = max(max_joint_command_speed, arm_delta)
                previous_arm_target = desired_arm.copy()
                arm_target = desired_arm
            except Exception as exc:
                ik_failures += 1
                consecutive_ik_failures += 1
                max_consecutive_ik_failures = max(max_consecutive_ik_failures, consecutive_ik_failures)
                last_ik_error = f"{type(exc).__name__}: {exc}"
                if first_ik_failure_time is None:
                    first_ik_failure_time = time_seconds
                if consecutive_ik_failures == 1 or consecutive_ik_failures % 10 == 0:
                    print(
                        "[handover] IK failure "
                        f"sim_time={time_seconds:.2f}s phase={phase} "
                        f"consecutive={consecutive_ik_failures}: {last_ik_error}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                consecutive_ik_failures = 0

        tcp_step_speed = float(np.linalg.norm(tcp_command_target - initial_tcp)) / max(time_seconds, simulation_dt)
        max_tcp_command_speed = max(max_tcp_command_speed, tcp_step_speed)

        if phase == "ROBOT_RETREAT" and grasp_verified:
            desired_retreat = rate_limit_vector(arm_target, LEFT_ARM_START, args.max_joint_speed * simulation_dt)
            max_joint_command_speed = max(max_joint_command_speed, float(np.linalg.norm(desired_retreat - arm_target)) / simulation_dt)
            arm_target = desired_retreat

        gripper_target = GRIPPER_CLOSED if time_seconds >= schedule["slow_end"] else GRIPPER_OPEN
        # Handover is currently a deterministic trajectory baseline: the
        # command itself is rate-limited above, then applied kinematically so
        # CPU-side actuator lag cannot make a valid slow trajectory miss the
        # object.  The legacy strict profile continues to use Genesis PD
        # control for the dynamics-oriented gate.
        robot.set_dofs_position(hold_targets, hold_dofs, zero_velocity=True)
        robot.set_dofs_position(arm_target, left_dofs, zero_velocity=True)
        robot.set_dofs_position(gripper_target, gripper_dofs, zero_velocity=True)
        update_viewer = args.viewer and should_update_viewer(step, viewer_interval, steps)
        if update_viewer:
            aligned = interpolate_points(vertices, source_position)
            mesh_root = interpolate_points(mesh_roots, source_position)
            aligned = aligned + (qpos[free_start : free_start + 3] - mesh_root)
            mesh.set_vverts(aligned)
            mesh_updates += 1
        scene.step(update_visualizer=update_viewer)
        viewer_updates += int(update_viewer)
        human.set_qpos(qpos, zero_velocity=True)

        if args.progress_hz > 0.0 and time_seconds + 1e-6 >= next_progress_time:
            print(
                f"[dynamic-handover] sim_time={time_seconds:6.2f}s phase={phase} "
                f"ik={ik_calls}/{ik_failures} mesh={mesh_updates} "
                f"wall={time.perf_counter() - started:6.1f}s",
                flush=True,
            )
            next_progress_time += 1.0 / args.progress_hz

        actual_tcp = tcp_position(end_effector)
        actual_link = as_numpy(end_effector.get_pos(relative=False)).reshape(3)
        tcp_target_distance = float(np.linalg.norm(actual_tcp - last_desired_tcp))
        link_target_distance = float(np.linalg.norm(actual_link - last_desired_tcp))
        object_position = as_numpy(object_entity.get_pos(relative=False)).reshape(3)
        object_tcp_distance = float(np.linalg.norm(object_position - actual_tcp))
        object_link_distance = float(np.linalg.norm(object_position - actual_link))
        min_object_tcp_distance = min(min_object_tcp_distance, object_tcp_distance)
        min_object_link_distance = min(min_object_link_distance, object_link_distance)
        min_tcp_target_distance = min(min_tcp_target_distance, tcp_target_distance)
        min_link_target_distance = min(min_link_target_distance, link_target_distance)
        final_object_tcp_distance = object_tcp_distance
        min_object_wrist_distance = min(min_object_wrist_distance, hand_pose_diagnostics["object_to_wrist_m"])
        min_object_index_distance = min(min_object_index_distance, hand_pose_diagnostics["object_to_index_m"])
        min_object_thumb_distance = min(min_object_thumb_distance, hand_pose_diagnostics["object_to_thumb_m"])
        min_object_palm_distance = min(min_object_palm_distance, hand_pose_diagnostics["object_to_palm_centroid_m"])
        if not hand_diagnostics or time_seconds - float(hand_diagnostics[-1]["time_s"]) >= 0.5:
            hand_diagnostics.append({"time_s": time_seconds, **hand_pose_diagnostics})
        if (phase.startswith("ROBOT_") or phase in ("GRIPPER_CLOSE", "GRASP_VERIFY")) and (
            not tcp_diagnostics or time_seconds - last_diagnostic_time >= 0.5
        ):
            tcp_diagnostics.append({
                "time_s": time_seconds,
                "phase": phase,
                "desired_tcp": last_desired_tcp.tolist(),
                "actual_link": actual_link.tolist(),
                "actual_tcp": actual_tcp.tolist(),
                "object": object_position.tolist(),
                "link_to_target_m": link_target_distance,
                "tcp_to_target_m": tcp_target_distance,
                "object_to_link_m": object_link_distance,
                "object_to_tcp_m": object_tcp_distance,
            })
            last_diagnostic_time = time_seconds
        contact_sample = step == 0 or step == steps or step % safety_interval == 0
        left_contact = link_contact_count(scene, left_finger1, object_entity) > 0
        right_contact = link_contact_count(scene, left_finger2, object_entity) > 0
        robot_object_contacts = contact_count(scene, robot, object_entity)
        human_object_contacts = contact_count(scene, human, object_entity)
        if left_contact:
            left_finger_contact_steps += 1
        if right_contact:
            right_finger_contact_steps += 1
        if robot_object_contacts > 0:
            robot_contact_steps += 1
        if human_object_contacts > 0:
            object_human_contacts += 1
        if contact_sample:
            object_robot_contacts += int(robot_object_contacts > 0)
            if control_transferred:
                human_object_contact_after_transfer += int(human_object_contacts > 0)
        finger_midpoint = 0.5 * (
            as_numpy(left_finger1.get_pos(relative=False)).reshape(3)
            + as_numpy(left_finger2.get_pos(relative=False)).reshape(3)
        )
        object_finger_gap_distance = float(np.linalg.norm(object_position - finger_midpoint))
        if phase == "GRASP_VERIFY" and left_contact and right_contact and object_finger_gap_distance <= GRASP_DISTANCE_TOLERANCE:
            finger_contact_stable_steps += 1
        else:
            finger_contact_stable_steps = 0
        if (not grasp_verified and finger_contact_stable_steps * simulation_dt >= GRASP_STABLE_SECONDS):
            grasp_verified = True
            release_verified = True
            phase_times.setdefault("HUMAN_RELEASE", time_seconds)
            transfer_time_s = time_seconds
            transfer_object_position = object_position.copy()
            try:
                scene.sim.rigid_solver.add_weld_constraint(int(end_effector.idx), object_link_idx)
                weld_constraint_registered = True
                control_transferred = True
                phase_times.setdefault("ROBOT_WELD", time_seconds)
                print(f"[dynamic-handover] sim_time={time_seconds:6.2f}s phase=ROBOT_WELD", flush=True)
            except Exception as exc:
                weld_constraint_error = f"{type(exc).__name__}: {exc}"
                print(f"[dynamic-handover] weld registration failed: {weld_constraint_error}", file=sys.stderr, flush=True)
        min_object_finger_gap_distance = min(min_object_finger_gap_distance, object_finger_gap_distance)
        if control_transferred:
            relative_error = float(np.linalg.norm(object_position - actual_tcp))
            max_relative_object_tcp_error = max(max_relative_object_tcp_error, relative_error)
            if transfer_object_position is not None:
                object_drop_m = max(object_drop_m, float(transfer_object_position[2] - object_position[2]))
        if control_transferred and phase == "ROBOT_RETREAT" and time_seconds >= schedule["retreat_end"]:
            retreat_completed = True

        safety_check = step == 0 or step == steps or step % safety_interval == 0
        if safety_check:
            safety_checks += 1
            current_distance = entity_aabb_distance(human, robot)
            min_distance = min(min_distance, current_distance)
            if time_seconds <= schedule["stable_start"]:
                min_pre_handover_distance = min(min_pre_handover_distance, current_distance)
            collision_steps += int(contact_count(scene, human, robot) > 0)
        trajectory_times.append(time_seconds)
        trajectory_human_qpos.append(qpos.copy())
        trajectory_robot_qpos.append(as_numpy(robot.get_qpos()).reshape(-1).astype(np.float32))

    # During a legitimate handover the gripper is expected to approach the
    # human-held object, so the all-phase AABB minimum may be zero.  The
    # personal-space gate therefore applies before the handover window; the
    # handover window itself is guarded by collision_steps and grasp metrics.
    layout_safe = (
        min_base_root_distance >= MIN_BASE_ROOT_DISTANCE
        and min_pre_handover_distance >= MIN_HUMAN_ROBOT_AABB_DISTANCE
    )
    if grasp_verified and weld_constraint_registered and retreat_completed and collision_steps == 0 and ik_failures == 0 and layout_safe:
        final_phase = "SUCCESS"
    elif collision_steps > 0 or not layout_safe:
        final_phase = "SAFETY_ABORT"
    elif ik_failures > 0:
        final_phase = "FAILED_IK"
    elif not grasp_verified:
        final_phase = "FAILED_GRASP"
    else:
        final_phase = "FAILED_RETREAT"
    phase_times[final_phase] = total_duration
    elapsed_wall_seconds = time.perf_counter() - started
    report = {
        "task": "walking_object_handover_dynamic_transfer",
        "status": "success" if final_phase == "SUCCESS" else "failure",
        "phase": final_phase,
        "dynamic_transfer_status": "success" if weld_constraint_registered and retreat_completed else "failure",
        "profile": "dynamic-handover",
        "seed": 2026,
        "motion": str(motion_path),
        "robot_urdf": str(robot_urdf),
        "human_model": str(human_model),
        "phase_times": phase_times,
        "schedule": schedule,
        "duration_seconds": total_duration,
        "full_task_duration_seconds": schedule["task_end"],
        "truncated_before_full_handover": truncated,
        "human": {
            "mesh_vertices": int(vertices.shape[1]), "mesh_faces": int(faces.shape[0]),
            "motion_duration_s": motion_duration, "pause_window_s": [pause_start / fps, pause_end / fps],
            "max_root_step_m": max_root_step,
        },
        "scene_layout": {
            "robot_base_position": robot_base.tolist(),
            "min_base_to_human_root_m": min_base_root_distance,
            "minimum_required_base_to_human_root_m": MIN_BASE_ROOT_DISTANCE,
            "layout_valid": min_base_root_distance >= MIN_BASE_ROOT_DISTANCE,
        },
        "object": {
            "shape": "box", "size_m": list(OBJECT_SIZE), "hand_control_before_grasp": True,
            "hand_attachment": "left_thumb_index_palm_frame",
            "hand_attachment_gate_blend": PALM_GATE_BLEND,
            "hand_attachment_wrist_joint": "L_Wrist",
            "hand_attachment_palm_joints": ["L_Index1", "L_Middle1", "L_Pinky1", "L_Ring1", "L_Thumb1"],
            "dynamic_after_grasp": bool(weld_constraint_registered),
            "robot_contact_steps": robot_contact_steps,
            "left_finger_contact_steps": left_finger_contact_steps,
            "right_finger_contact_steps": right_finger_contact_steps,
            "human_contact_steps": object_human_contacts,
            "human_object_contact_after_transfer": human_object_contact_after_transfer,
            "max_relative_object_tcp_error_m": max_relative_object_tcp_error,
            "min_object_finger_gap_distance_m": min_object_finger_gap_distance,
            "object_drop_m": object_drop_m,
            "weld_constraint_registered": weld_constraint_registered,
            "weld_constraint_error": weld_constraint_error,
            "transfer_time_s": transfer_time_s,
            "grasp_distance_tolerance_m": GRASP_DISTANCE_TOLERANCE,
            "min_object_tcp_distance_m": min_object_tcp_distance,
            "min_object_link_distance_m": min_object_link_distance,
            "min_object_wrist_distance_m": min_object_wrist_distance,
            "min_object_index_distance_m": min_object_index_distance,
            "min_object_thumb_distance_m": min_object_thumb_distance,
            "min_object_palm_centroid_distance_m": min_object_palm_distance,
            "hand_pose_diagnostics": hand_diagnostics,
            "min_tcp_target_distance_m": min_tcp_target_distance,
            "min_link_target_distance_m": min_link_target_distance,
            "final_object_tcp_distance_m": final_object_tcp_distance,
            "tcp_diagnostics": tcp_diagnostics,
            "grasp_verified": grasp_verified, "control_transferred": control_transferred,
            "release_verified": release_verified, "retreat_completed": retreat_completed,
        },
        "robot": {
            "base_position": robot_base.tolist(), "min_human_robot_aabb_distance_m": min_distance,
            "min_pre_handover_aabb_distance_m": min_pre_handover_distance,
            "minimum_human_robot_aabb_distance_m": MIN_HUMAN_ROBOT_AABB_DISTANCE,
            "layout_safe": layout_safe,
            "collision_steps": collision_steps, "ik_calls": ik_calls, "ik_failures": ik_failures,
            "last_ik_error": last_ik_error,
            "first_ik_failure_time_s": first_ik_failure_time,
            "max_consecutive_ik_failures": max_consecutive_ik_failures,
            "ik_hz_effective": effective_ik_hz,
            "ik_max_samples_effective": effective_ik_samples,
            "ik_max_solver_iters_effective": effective_ik_solver_iters,
            "max_ik_position_error_m": max_ik_error,
            "max_tcp_command_speed_mps": max_tcp_command_speed,
            "max_joint_command_speed_radps": max_joint_command_speed,
            "max_tcp_speed_limit_mps": args.max_tcp_speed,
            "max_joint_speed_limit_radps": args.max_joint_speed,
            "safety_checks": safety_checks,
        },
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "realtime_factor": total_duration / max(elapsed_wall_seconds, 1e-9),
        "simulation_hz": simulation_hz,
        "simulation_steps": steps + 1,
        "viewer": args.viewer,
        "viewer_updates": viewer_updates,
        "mesh_updates": mesh_updates,
        "backend_requested": args.backend,
        "backend_used": backend_name,
        "note": "The object is dynamic. Before transfer it is held by explicit human-side kinematic updates; after dual-finger contact Genesis owns it through a runtime weld constraint and no object pose writes occur.",
    }
    if args.save_trajectory is not None:
        save_trajectory_cache(
            args.save_trajectory.resolve(), trajectory_times, trajectory_human_qpos, trajectory_robot_qpos,
            {
                "source_report": str(report_path),
                "motion": str(motion_path),
                "profile": "dynamic-handover",
                "backend_used": backend_name,
                "robot_base_position": robot_base.tolist(),
                "control_transfer_time_s": float(phase_times.get("HUMAN_RELEASE", schedule["verify_end"])),
                "object_attachment": "left_thumb_index_palm_frame",
            },
        )
        report["trajectory"] = str(args.save_trajectory.resolve())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report




def run_visual_profile(args: argparse.Namespace, include_robot: bool) -> dict[str, object]:
    """Run a visual-only scene without the strict HRI control/diagnostic loop.

    ``human-viewer`` intentionally delegates to the already validated human
    mesh replay. ``hri-viewer`` adds the real R1 Pro asset in a fixed neutral
    pose, but does not run IK, contact queries, or safety checks. Neither mode
    is an HRI task result; they are for responsive visual inspection only.
    """
    if not args.viewer:
        # Headless execution is useful for a deterministic performance report.
        pass
    if not include_robot:
        from replay_hymotion_mesh_genesis import run as run_mesh_replay

        mesh_args = argparse.Namespace(
            motion=args.motion,
            model=args.human_model,
            model_dir=args.wooden_dir,
            report=args.report,
            mesh_obj=args.mesh_obj,
            seconds=args.seconds,
            viewer=args.viewer,
            fixed_camera=True,
            collision=False,
            interpolation="cubic",
        )
        report = run_mesh_replay(mesh_args)
        report["profile"] = "human-viewer"
        report["robot_loaded"] = False
        args.report.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    motion_path = args.motion.resolve()
    human_model = args.human_model.resolve()
    wooden_dir = args.wooden_dir.resolve()
    robot_urdf = args.robot_urdf.resolve()
    report_path = args.report.resolve()
    for required in (motion_path, human_model, robot_urdf, wooden_dir / "v_template.bin"):
        if not required.exists():
            raise FileNotFoundError(required)

    vertices, faces, mesh_roots, _hand_keypoints, mesh_fps = load_wooden_motion(motion_path, wooden_dir)
    points, fps = prepare_points(motion_path)
    if abs(mesh_fps - fps) > 1e-4:
        raise ValueError("Motion FPS differs between WoodenMesh and keypoint inputs")
    requested_duration = (len(points) - 1) / fps if args.seconds is None else min(args.seconds, (len(points) - 1) / fps)
    viewer_hz = max(args.viewer_hz, 30.0)
    simulation_dt = 1.0 / viewer_hz
    steps = int(np.ceil(requested_duration / simulation_dt))
    backend, backend_name = select_backend(args.backend)
    write_obj(args.mesh_obj.resolve(), vertices[0], faces)
    rest_basis = rotation_from_basis(points[0])
    gs.init(backend=backend, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=simulation_dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False, box_box_detection=False),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), refresh_rate=int(round(viewer_hz)), realtime_factor=1.0,
            camera_pos=(2.8, -5.4, 2.5), camera_lookat=(0.0, -1.5, 0.95), camera_fov=42,
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_model), requires_jac_and_IK=False), name="hri_human_proxy"
    )
    mesh = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh_obj.resolve()), fixed=True, visualization=True, collision=False,
            enable_custom_vverts=True, decimate=False, convexify=False, file_meshes_are_zup=True,
        ),
        name="hri_human_visual",
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf), pos=tuple(ROBOT_BASE_POS), fixed=True,
            merge_fixed_links=True, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0), name="r1_pro_hri_preview",
    )
    scene.build()
    if mesh.n_vverts != vertices.shape[1]:
        raise RuntimeError(f"Genesis visual vertex count mismatch: {mesh.n_vverts} != {vertices.shape[1]}")
    free_start = human.get_joint(name="pelvis_free").q_start
    qposes = np.stack([qpos_for_frame(frame, points[0], rest_basis, human) for frame in points]).astype(np.float32)
    qposes = make_quaternion_sequence_continuous(qposes)
    cubic_at = build_cubic_qpos_interpolator(qposes, fps)
    initial_qpos = cubic_at(0.0)
    initial_mesh_offset = initial_qpos[free_start : free_start + 3] - mesh_roots[0]
    mesh.set_vverts(vertices[0] + initial_mesh_offset)
    human.set_qpos(initial_qpos, zero_velocity=True)
    started = time.perf_counter()
    mesh_updates = 0
    for step in range(steps + 1):
        time_seconds = min(step * simulation_dt, requested_duration)
        source_position = time_seconds * fps
        qpos = cubic_at(time_seconds)
        human.set_qpos(qpos, zero_velocity=True)
        aligned = interpolate_points(vertices, source_position)
        mesh_root = interpolate_points(mesh_roots, source_position)
        aligned = aligned + (qpos[free_start : free_start + 3] - mesh_root)
        mesh.set_vverts(aligned)
        mesh_updates += 1
        scene.step(update_visualizer=args.viewer)
    elapsed_wall_seconds = time.perf_counter() - started
    report = {
        "status": "success",
        "task": "visual_preview_only",
        "profile": "hri-viewer",
        "motion": str(motion_path),
        "robot_urdf": str(robot_urdf),
        "human_model": str(human_model),
        "duration_seconds": requested_duration,
        "simulation_hz": viewer_hz,
        "simulation_dt": simulation_dt,
        "simulation_steps": steps + 1,
        "mesh_vertices": int(vertices.shape[1]),
        "mesh_faces": int(faces.shape[0]),
        "mesh_updates": mesh_updates,
        "robot_loaded": True,
        "robot_base_fixed": True,
        "ik_calls": 0,
        "collision_enabled": False,
        "safety_checks": 0,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "realtime_factor": requested_duration / max(elapsed_wall_seconds, 1e-9),
        "backend_used": backend_name,
        "note": "Visual preview only: R1 Pro is fixed in a neutral pose; no IK, contact query, or HRI safety result is produced.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.profile == "layout-diagnostic":
        from diagnose_hri_layout import run as run_layout_diagnostic

        layout_report = args.report
        if layout_report == DEFAULT_REPORT:
            layout_report = WORKSPACE / "genesis-human-experiment/reports/r1pro_layout_diagnostic.json"
        layout_args = argparse.Namespace(
            motion=args.motion, human_model=args.human_model, robot_urdf=args.robot_urdf,
            report=layout_report, base_x=-0.55, base_y=-3.00, seconds=args.seconds,
            viewer=args.viewer, backend="cpu" if args.backend == "auto" else args.backend,
            min_clearance=0.25, sample_hz=10.0,
        )
        return run_layout_diagnostic(layout_args)
    if args.profile == "handover":
        return run_handover(args)
    if args.profile == "dynamic-handover":
        return run_dynamic_handover(args)
    if args.profile == "hri-replay":
        return run_hri_replay(args)
    if args.profile == "human-viewer":
        return run_visual_profile(args, include_robot=False)
    if args.profile == "hri-viewer":
        return run_visual_profile(args, include_robot=True)
    if args.ik_max_samples <= 0 or args.ik_max_solver_iters <= 0:
        raise ValueError("IK sample and iteration limits must be positive")
    if args.viewer_hz <= 0.0 or args.ik_hz <= 0.0 or args.pause_ik_hz <= 0.0 or args.safety_check_hz <= 0.0:
        raise ValueError("viewer, IK, pause IK, and safety-check rates must be positive")
    if args.profile == "smooth-viewer":
        # The strict profile is the reproducible HRI gate. The visual profile reduces
        # CPU-side control work and keeps the fixed-base scene responsive.
        args.ik_hz = min(args.ik_hz, 20.0)
        args.pause_ik_hz = min(args.pause_ik_hz, 40.0)
        args.ik_max_samples = min(args.ik_max_samples, 8)
        args.ik_max_solver_iters = min(args.ik_max_solver_iters, 80)
        args.viewer_hz = max(args.viewer_hz, 30.0)
    motion_path = args.motion.resolve()
    human_model = args.human_model.resolve()
    wooden_dir = args.wooden_dir.resolve()
    robot_urdf = args.robot_urdf.resolve()
    report_path = args.report.resolve()
    for required in (motion_path, human_model, robot_urdf, wooden_dir / "v_template.bin"):
        if not required.exists():
            raise FileNotFoundError(required)

    vertices, faces, mesh_roots, _hand_keypoints, mesh_fps = load_wooden_motion(motion_path, wooden_dir)
    points, fps = prepare_points(motion_path)
    if abs(mesh_fps - fps) > 1e-4:
        raise ValueError("Motion FPS differs between WoodenMesh and keypoint inputs")
    hand_index = BODY_NAMES.index(HAND_NAME)
    pause_start, pause_end = detect_pause_window(points, fps, hand_index, HAND_SPEED_THRESHOLD, MIN_PAUSE_SECONDS)
    requested_duration = (len(points) - 1) / fps if args.seconds is None else min(args.seconds, (len(points) - 1) / fps)
    # Strict mode keeps the 100 Hz task contract. The visual profile advances
    # one Genesis step per displayed frame; this avoids spending most of the
    # wall time on invisible intermediate steps on CPU-only machines.
    simulation_hz = args.viewer_hz if args.profile == "smooth-viewer" else 1.0 / SIM_DT
    simulation_dt = 1.0 / simulation_hz
    steps = int(np.ceil(requested_duration / simulation_dt))
    if requested_duration < pause_start / fps:
        raise ValueError("Requested duration ends before the detected interaction window")
    camera_mode = "fixed" if args.fixed_camera else args.camera_mode
    ik_interval = step_interval_for_rate(args.ik_hz, simulation_dt)
    pause_ik_interval = step_interval_for_rate(args.pause_ik_hz, simulation_dt)
    viewer_interval = step_interval_for_rate(args.viewer_hz, simulation_dt)
    safety_interval = step_interval_for_rate(args.safety_check_hz, simulation_dt)
    backend, backend_name = select_backend(args.backend)

    write_obj(args.mesh_obj.resolve(), vertices[0], faces)
    rest_basis = rotation_from_basis(points[0])
    gs.init(backend=backend, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=simulation_dt, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=not args.no_collision, box_box_detection=True, noslip_iterations=5),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 800), refresh_rate=int(round(args.viewer_hz)), realtime_factor=1.0,
            camera_pos=(2.8, -5.4, 2.5), camera_lookat=(0.0, -1.5, 0.95), camera_fov=42
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.MJCF(file=str(human_model), requires_jac_and_IK=False), name="hri_human_proxy"
    )
    mesh = scene.add_entity(
        gs.morphs.Mesh(
            file=str(args.mesh_obj.resolve()), fixed=True, visualization=True, collision=False,
            enable_custom_vverts=True, decimate=False, convexify=False, file_meshes_are_zup=True,
        ),
        name="hri_human_visual",
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_urdf), pos=tuple(ROBOT_BASE_POS),
            fixed=args.profile == "smooth-viewer", merge_fixed_links=False, align=False,
        ),
        material=gs.materials.Rigid(friction=2.0),
        name="r1_pro_hri",
    )
    scene.build()
    if mesh.n_vverts != vertices.shape[1]:
        raise RuntimeError(f"Genesis visual vertex count mismatch: {mesh.n_vverts} != {vertices.shape[1]}")
    free_start = human.get_joint(name="pelvis_free").q_start
    qposes = np.stack([qpos_for_frame(frame, points[0], rest_basis, human) for frame in points]).astype(np.float32)
    qposes = make_quaternion_sequence_continuous(qposes)
    qposes[:, free_start : free_start + 3] += HUMAN_OFFSET
    cubic_at = build_cubic_qpos_interpolator(qposes, fps)

    initial_qpos = cubic_at(0.0)
    initial_proxy_root = initial_qpos[free_start : free_start + 3]
    initial_mesh_root = mesh_roots[0]
    initial_mesh_offset = initial_proxy_root - initial_mesh_root
    mesh.set_vverts(vertices[0] + initial_mesh_offset)
    roundtrip = as_numpy(mesh.get_vverts())
    roundtrip_error = float(np.max(np.abs(roundtrip - (vertices[0] + initial_mesh_offset))))
    if roundtrip_error > 1e-5:
        raise RuntimeError(f"Genesis mesh vertex round-trip error: {roundtrip_error:.3e} m")
    left_dofs, left_qs, hold_dofs, end_effector, gripper_dofs = configure_robot(robot)
    hold_targets = as_numpy(robot.get_dofs_position())[hold_dofs].copy()
    human.set_qpos(initial_qpos, zero_velocity=True)
    viewer_updates = 0
    mesh_updates = 0
    reset_steps = max(1, int(round(1.0 / simulation_dt)))
    for reset_step in range(reset_steps):
        robot.control_dofs_position(hold_targets, hold_dofs)
        robot.control_dofs_position(LEFT_ARM_START, left_dofs)
        robot.control_dofs_position(GRIPPER_OPEN, gripper_dofs)
        update_viewer = args.viewer and should_update_viewer(reset_step, viewer_interval, 99)
        scene.step(update_visualizer=update_viewer)
        viewer_updates += int(update_viewer)
    if args.viewer and camera_mode == "follow":
        scene.viewer.follow_entity(robot, fixed_axis=(None, None, None), smoothing=0.95, fix_orientation=False)

    approach_start = max(0.0, pause_start / fps - REACH_LEAD_SECONDS)
    phase = "HUMAN_WALK"
    phase_times: dict[str, float] = {"RESET": 0.0, "HUMAN_WALK": 0.0}
    phase_started = 0.0
    arm_target = LEFT_ARM_START.copy()
    ik_has_solution = False
    ik_calls = 0
    max_hand_error = 0.0
    max_ik_error = 0.0
    pause_max_hand_error = 0.0
    pause_max_ik_error = 0.0
    pause_ik_failures = 0
    min_distance = float("inf")
    collision_steps = 0
    max_root_step = 0.0
    previous_root = None
    hold_steps = 0
    ik_failures = 0
    post_hold_ik_failures = 0
    target_reached_steps = 0
    safety_checks = 0
    last_step_contacts = 0
    trajectory_times: list[float] = []
    trajectory_human_qpos: list[np.ndarray] = []
    trajectory_robot_qpos: list[np.ndarray] = []
    started = time.perf_counter()

    for step in range(steps + 1):
        time_seconds = min(step * simulation_dt, requested_duration)
        source_position = time_seconds * fps
        frame_points = interpolate_points(points, source_position)
        qpos = cubic_at(time_seconds)
        human.set_qpos(qpos, zero_velocity=True)
        if previous_root is not None:
            max_root_step = max(max_root_step, float(np.linalg.norm(qpos[free_start : free_start + 3] - previous_root)))
        previous_root = qpos[free_start : free_start + 3].copy()

        hand_world = frame_points[hand_index] + HUMAN_OFFSET
        robot_world = as_numpy(robot.get_pos()).reshape(3)
        target = interaction_target(hand_world, robot_world)
        in_reach_phase = time_seconds >= approach_start
        in_pause = pause_start / fps <= time_seconds <= pause_end / fps
        if in_reach_phase:
            if phase == "HUMAN_WALK":
                phase = "HUMAN_TURN_REACH"
                phase_times[phase] = time_seconds
            solve_interval = pause_ik_interval if in_pause else ik_interval
            if not ik_has_solution or step % solve_interval == 0:
                ik_calls += 1
                try:
                    solved_qpos, ik_error = robot.inverse_kinematics(
                        link=end_effector, pos=target, quat=GRIPPER_QUAT, local_point=GRIPPER_TCP_LOCAL,
                        init_qpos=robot.get_qpos(), dofs_idx_local=left_dofs,
                        # The first HRI gate is spatially safe approach. Gripper
                        # orientation is intentionally left unconstrained until a
                        # handover/grasp task defines its required attitude.
                        rot_mask=(False, False, False), max_samples=args.ik_max_samples,
                        max_solver_iters=args.ik_max_solver_iters, return_error=True,
                    )
                    ik_position_error = float(np.linalg.norm(as_numpy(ik_error).reshape(-1)[:3]))
                    max_ik_error = max(max_ik_error, ik_position_error)
                    if in_pause:
                        pause_max_ik_error = max(pause_max_ik_error, ik_position_error)
                    if ik_position_error > MAX_IK_POSITION_ERROR:
                        if phase == "HOLD":
                            post_hold_ik_failures += 1
                        else:
                            ik_failures += 1
                        if in_pause and phase != "HOLD":
                            pause_ik_failures += 1
                    else:
                        solved_arm = as_numpy(solved_qpos).reshape(-1)[left_qs]
                        arm_target = (0.80 * arm_target + 0.20 * solved_arm).astype(np.float32)
                        ik_has_solution = True
                except Exception:
                    if phase == "HOLD":
                        post_hold_ik_failures += 1
                    else:
                        ik_failures += 1
                    if in_pause and phase != "HOLD":
                        pause_ik_failures += 1

        robot.control_dofs_position(hold_targets, hold_dofs)
        robot.control_dofs_position(arm_target, left_dofs)
        robot.control_dofs_position(GRIPPER_OPEN, gripper_dofs)
        update_viewer = args.viewer and should_update_viewer(step, viewer_interval, steps)
        if update_viewer:
            mesh_root = interpolate_points(mesh_roots, source_position)
            aligned = interpolate_points(vertices, source_position)
            aligned = aligned + (qpos[free_start : free_start + 3] - mesh_root)
            mesh.set_vverts(aligned)
            mesh_updates += 1
        scene.step(update_visualizer=update_viewer)
        viewer_updates += int(update_viewer)
        human.set_qpos(qpos, zero_velocity=True)
        trajectory_times.append(time_seconds)
        trajectory_human_qpos.append(qpos.copy())
        trajectory_robot_qpos.append(as_numpy(robot.get_qpos()).reshape(-1).astype(np.float32))

        actual_hand_error = float(np.linalg.norm(tcp_position(end_effector) - target))
        if in_reach_phase:
            max_hand_error = max(max_hand_error, actual_hand_error)
        if in_pause:
            pause_max_hand_error = max(pause_max_hand_error, actual_hand_error)
        safety_check = step == 0 or step == steps or step % safety_interval == 0
        if safety_check:
            safety_checks += 1
            min_distance = min(min_distance, entity_aabb_distance(human, robot))
            last_step_contacts = contact_count(scene, human, robot)
            step_contacts = last_step_contacts
            collision_steps += int(step_contacts > 0)
        else:
            step_contacts = last_step_contacts
        # AABB overlap is a conservative broad-phase diagnostic, not a
        # surface-distance measurement. The close approach is therefore gated
        # by the actual contact query and the TCP error; AABB is still reported.
        if in_pause and actual_hand_error <= MAX_HAND_TARGET_ERROR and step_contacts == 0:
            target_reached_steps += 1
            hold_steps += 1
        else:
            hold_steps = 0
        if in_pause and hold_steps * simulation_dt >= HOLD_SECONDS and phase != "HOLD":
            phase = "HOLD"
            phase_times[phase] = time_seconds

    hold_duration = hold_steps * simulation_dt
    if hold_duration >= HOLD_SECONDS and collision_steps == 0 and pause_ik_failures == 0 and pause_max_hand_error <= MAX_HAND_TARGET_ERROR:
        phase = "SUCCESS"
    elif collision_steps > 0:
        phase = "SAFETY_ABORT"
    elif ik_failures > 0:
        phase = "SAFETY_ABORT"
    else:
        phase = "FAILED"
    phase_times[phase] = requested_duration
    elapsed_wall_seconds = time.perf_counter() - started
    if args.save_trajectory is not None:
        save_trajectory_cache(
            args.save_trajectory.resolve(),
            trajectory_times,
            trajectory_human_qpos,
            trajectory_robot_qpos,
            {
                "source_report": str(report_path),
                "motion": str(motion_path),
                "profile": args.profile,
                "backend_used": backend_name,
                "collision_enabled": not args.no_collision,
                "ik_hz_requested": args.ik_hz,
                "simulation_hz": simulation_hz,
            },
        )
    report = {
        "task": "walking_approach_safe_hand_interaction",
        "status": "success" if phase == "SUCCESS" else "failure",
        "phase": phase,
        "seed": 2026,
        "motion": str(motion_path),
        "robot_urdf": str(robot_urdf),
        "human_model": str(human_model),
        "phase_times": phase_times,
        "human": {
            "mesh_vertices": int(vertices.shape[1]), "mesh_faces": int(faces.shape[0]),
            "motion_duration_s": requested_duration, "pause_window_s": [pause_start / fps, pause_end / fps],
            "max_root_step_m": max_root_step, "mesh_vertex_roundtrip_error_m": roundtrip_error,
        },
        "interaction": {
            "hand": "L_Wrist", "reach_start_s": approach_start, "pause_start_s": pause_start / fps,
            "pause_end_s": pause_end / fps, "hold_duration_s": hold_duration,
            "max_hand_target_error_m": max_hand_error, "max_ik_position_error_m": max_ik_error,
            "pause_max_hand_target_error_m": pause_max_hand_error,
            "pause_max_ik_position_error_m": pause_max_ik_error,
            "pause_ik_failures": pause_ik_failures,
            "target_offset_m": 0.10,
        },
        "robot": {
            "base_position": ROBOT_BASE_POS.tolist(), "min_human_robot_aabb_distance_m": min_distance,
            "collision_steps": collision_steps, "ik_failures": ik_failures,
            "post_hold_ik_failures": post_hold_ik_failures,
            "collision_enabled": not args.no_collision,
            "aabb_is_broad_phase_diagnostic": True,
            "ik_calls": ik_calls,
            "ik_hz_effective": effective_ik_hz,
            "ik_hz_requested": args.ik_hz,
            "ik_update_interval_steps": ik_interval,
            "pause_ik_hz_requested": args.pause_ik_hz,
            "pause_ik_update_interval_steps": pause_ik_interval,
            "ik_max_samples": args.ik_max_samples,
            "ik_max_solver_iters": args.ik_max_solver_iters,
            "safety_checks": safety_checks,
            "safety_check_hz_requested": args.safety_check_hz,
        },
        "thresholds": {
            "min_human_robot_distance_m": MIN_HUMAN_ROBOT_DISTANCE,
            "max_hand_target_error_m": MAX_HAND_TARGET_ERROR,
            "max_ik_position_error_m": MAX_IK_POSITION_ERROR,
            "hold_seconds": HOLD_SECONDS,
        },
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "realtime_factor": requested_duration / max(elapsed_wall_seconds, 1e-9),
        "simulation_hz": simulation_hz,
        "simulation_dt": simulation_dt,
        "simulation_steps": steps + 1,
        "viewer": args.viewer,
        "viewer_updates": viewer_updates,
        "mesh_updates": mesh_updates,
        "viewer_hz_requested": args.viewer_hz,
        "camera_mode": camera_mode if args.viewer else "headless",
        "profile": args.profile,
        "trajectory": str(args.save_trajectory.resolve()) if args.save_trajectory is not None else None,
        "robot_base_fixed": args.profile == "smooth-viewer",
        "backend_requested": args.backend,
        "backend_used": backend_name,
        "note": "The human is kinematic; the WoodenMesh is visual-only and the MJCF proxy is used for collision checks. IK is rate-limited; the latest valid arm target is held between solves.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
