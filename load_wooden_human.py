"""Load the official HY-Motion wooden human as a static Genesis visual asset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import genesis as gs
import numpy as np
import trimesh


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = WORKSPACE / "genesis-human-experiment/assets/wooden_model/boy_Rigging_smplx_tex.glb"
DEFAULT_REPORT = WORKSPACE / "genesis-human-experiment/reports/wooden_human_static.json"
HY_TO_GENESIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def get_genesis_bounds(model_path: Path, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    loaded = trimesh.load(str(model_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometry = loaded.dump(concatenate=True)
    else:
        geometry = loaded
    vertices = np.asarray(geometry.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Could not read mesh vertices from {model_path}")
    points = (vertices @ HY_TO_GENESIS.T) * scale
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    ground_lift = max(0.0, 0.02 - float(minimum[2]))
    return minimum, maximum, ground_lift


def run(model_path: Path, report_path: Path, scale: float, show_viewer: bool) -> None:
    model_path = model_path.resolve()
    report_path = report_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if scale <= 0:
        raise ValueError("scale must be positive")
    bounds_min, bounds_max, ground_lift = get_genesis_bounds(model_path, scale)

    gs.init(backend=gs.cpu, precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 640),
            camera_pos=(2.8, -2.8, 1.8),
            camera_lookat=(0.0, 0.0, 0.85),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), name="ground")
    human = scene.add_entity(
        gs.morphs.Mesh(
            file=str(model_path),
            scale=scale,
            file_meshes_are_zup=False,
            pos=(0.0, 0.0, ground_lift),
            fixed=True,
            collision=False,
            visualization=True,
            decimate=False,
        ),
        name="hymotion_wooden_human",
    )
    started = time.perf_counter()
    scene.build()
    if show_viewer:
        while scene.viewer.is_alive():
            scene.step()
    else:
        for _ in range(3):
            scene.step(update_visualizer=False)

    report = {
        "status": "success",
        "model": str(model_path),
        "scale": scale,
        "coordinate_conversion": "GLB Y-up -> Genesis Z-up",
        "mesh_bounds_before_ground_lift_m": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
        },
        "ground_lift_m": ground_lift,
        "human_mode": "static_visual_mesh",
        "collision": False,
        "viewer": show_viewer,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "note": "The GLB contains the source skeleton, but this stage does not drive skinning or joints.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.report, args.scale, args.viewer)
