"""Generate a 22-joint Genesis/MJCF human collision skeleton from a HY-Motion clip."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
DEFAULT_OUTPUT = WORKSPACE / "genesis-human-experiment/assets/human_22ball.xml"
BODY_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int32,
)
HY_TO_GENESIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=float, default=0.045)
    return parser.parse_args()


def rotation_from_basis(points: np.ndarray) -> np.ndarray:
    """Construct a right-handed pelvis frame: x=left-right, z=up, y=forward."""
    x = points[1] - points[2]
    z = points[3] - points[0]
    x /= max(np.linalg.norm(x), 1e-8)
    z -= x * np.dot(x, z)
    z /= max(np.linalg.norm(z), 1e-8)
    y = np.cross(z, x)
    y /= max(np.linalg.norm(y), 1e-8)
    return np.stack([x, y, z], axis=1)


def load_rest_points(motion_path: Path) -> np.ndarray:
    data = np.load(motion_path, allow_pickle=False)
    points_hy = data["keypoints_hy"].astype(np.float32)
    points = points_hy @ HY_TO_GENESIS.T
    ground_lift = max(0.0, 0.02 - float(points[..., 2].min()))
    points[..., 2] += ground_lift
    return points[0, : len(BODY_NAMES)]


def fmt(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.8f}" for value in values)


def generate(motion_path: Path, output_path: Path, radius: float) -> None:
    rest = load_rest_points(motion_path)
    rest_basis = rotation_from_basis(rest)
    local_offsets = np.zeros_like(rest)
    for child in range(1, len(BODY_NAMES)):
        parent = int(PARENTS[child])
        local_offsets[child] = rest_basis.T @ (rest[child] - rest[parent])

    root = ET.Element("mujoco", {"model": "hymotion_human_22ball"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local", "inertiafromgeom": "true"})
    ET.SubElement(root, "option", {"timestep": "0.01", "gravity": "0 0 0", "integrator": "Euler"})
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"damping": "0.2", "armature": "0.01", "limited": "false"})
    ET.SubElement(
        default,
        "geom",
        {
            "contype": "1", "conaffinity": "1", "condim": "3", "density": "500",
            "friction": "0.8 0.1 0.1", "rgba": "0.72 0.58 0.38 1",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    root_body = ET.SubElement(worldbody, "body", {"name": BODY_NAMES[0], "pos": "0 0 0"})
    ET.SubElement(root_body, "freejoint", {"name": "pelvis_free"})
    ET.SubElement(root_body, "geom", {"name": "pelvis_joint", "type": "sphere", "size": f"{radius * 1.25:.6f}"})

    body_elements = {0: root_body}
    for child in range(1, len(BODY_NAMES)):
        parent = int(PARENTS[child])
        parent_body = body_elements[parent]
        offset = local_offsets[child]
        length = float(np.linalg.norm(offset))
        geom_radius = min(radius, max(0.018, 0.32 * length))
        # The parent-child bone must live in the parent frame. If it is put
        # inside the child body, a ball-joint rotation rotates the far end of
        # the capsule away from the parent anchor, creating a visual gap even
        # though the MJCF joint anchors remain connected.
        ET.SubElement(
            parent_body,
            "geom",
            {
                "name": f"{BODY_NAMES[child]}_bone",
                "type": "capsule",
                "fromto": f"0 0 0 {fmt(offset)}",
                "size": f"{geom_radius:.6f}",
            },
        )
        body = ET.SubElement(
            parent_body,
            "body",
            {"name": BODY_NAMES[child], "pos": fmt(offset)},
        )
        ET.SubElement(body, "joint", {"name": f"{BODY_NAMES[child]}_ball", "type": "ball", "pos": "0 0 0"})
        ET.SubElement(body, "geom", {"name": f"{BODY_NAMES[child]}_joint", "type": "sphere", "size": f"{radius:.6f}"})
        body_elements[child] = body

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"output={output_path.resolve()}")
    print(f"joints={len(BODY_NAMES)}")
    print(f"root_rest_basis={rest_basis.tolist()}")


if __name__ == "__main__":
    args = parse_args()
    generate(args.motion.resolve(), args.output.resolve(), args.radius)
