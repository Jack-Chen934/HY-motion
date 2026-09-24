"""Small, read-only adapter for the Galaxea R1 Pro URDF used by HRI scenes.

The adapter keeps robot-specific names and gripper geometry out of the task
runner.  It reads the checked-in URDF and does not modify the robot project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class FingerCollision:
    link_name: str
    joint_name: str
    joint_origin: np.ndarray
    joint_axis: np.ndarray
    collision_origin: np.ndarray
    size: np.ndarray
    lower: float
    upper: float

    def center(self, joint_position: float) -> np.ndarray:
        return self.joint_origin + self.collision_origin + self.joint_axis * joint_position


@dataclass(frozen=True)
class R1ProModel:
    urdf_path: Path
    links: tuple[str, ...]
    joints: tuple[str, ...]
    left_fingers: tuple[FingerCollision, FingerCollision]
    tcp_local: np.ndarray

    def jaw_geometry(self, joint_positions: tuple[float, float]) -> dict[str, object]:
        """Return finger centers, opening axis, and inner-face gap in gripper frame."""
        if len(joint_positions) != 2:
            raise ValueError("left gripper requires two joint positions")
        first, second = self.left_fingers
        centers = np.stack((first.center(joint_positions[0]), second.center(joint_positions[1])))
        axis = first.joint_axis / np.linalg.norm(first.joint_axis)
        center_gap = abs(float(np.dot(centers[1] - centers[0], axis)))
        half_widths = (abs(float(np.dot(first.size / 2.0, np.abs(axis)))),
                      abs(float(np.dot(second.size / 2.0, np.abs(axis)))))
        inner_gap = max(0.0, center_gap - sum(half_widths))
        return {
            "finger_centers_m": centers.tolist(),
            "opening_axis_gripper": axis.tolist(),
            "center_gap_m": center_gap,
            "inner_face_gap_m": inner_gap,
        }


def _vector(element: ET.Element | None, attribute: str, default: tuple[float, float, float]) -> np.ndarray:
    if element is None or element.get(attribute) is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(element.get(attribute, ""), sep=" ", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected three values for {attribute}, got {values}")
    return values


def _parse_finger(robot: ET.Element, link_name: str) -> FingerCollision:
    link = robot.find(f"link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"URDF is missing required link {link_name}")
    joint = robot.find(f"joint[@name='{link_name.replace('link', 'joint', 1)}']")
    # The asset uses finger_linkN / finger_jointN names; derive the canonical
    # joint name explicitly when the above textual substitution is not valid.
    if joint is None:
        suffix = link_name.rsplit("link", 1)[-1]
        joint = robot.find(f"joint[@name='left_gripper_finger_joint{suffix}']")
    if joint is None:
        raise ValueError(f"URDF is missing a joint for {link_name}")
    collision = link.find("collision")
    box = collision.find("geometry/box") if collision is not None else None
    if box is None:
        raise ValueError(f"{link_name} must have a box collision proxy")
    size = np.fromstring(box.get("size", ""), sep=" ", dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError(f"invalid collision box size for {link_name}: {size}")
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"{joint.get('name')} is missing joint limits")
    return FingerCollision(
        link_name=link_name,
        joint_name=joint.get("name", ""),
        joint_origin=_vector(joint.find("origin"), "xyz", (0.0, 0.0, 0.0)),
        joint_axis=_vector(joint.find("axis"), "xyz", (0.0, 0.0, 0.0)),
        collision_origin=_vector(collision.find("origin"), "xyz", (0.0, 0.0, 0.0)),
        size=size,
        lower=float(limit.get("lower", "0")),
        upper=float(limit.get("upper", "0")),
    )


def load_r1pro(urdf_path: Path | str) -> R1ProModel:
    """Parse and validate the parts of the R1 Pro URDF needed by HRI."""
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    links = tuple(link.get("name", "") for link in root.findall("link"))
    joints = tuple(joint.get("name", "") for joint in root.findall("joint"))
    for required in ("left_gripper_link", "left_gripper_finger_link1", "left_gripper_finger_link2"):
        if required not in links:
            raise ValueError(f"URDF is missing required link {required}")
    fingers = (
        _parse_finger(root, "left_gripper_finger_link1"),
        _parse_finger(root, "left_gripper_finger_link2"),
    )
    return R1ProModel(
        urdf_path=path,
        links=links,
        joints=joints,
        left_fingers=fingers,
        tcp_local=np.asarray((0.0, 0.0, -0.03689), dtype=np.float64),
    )


def object_fits_between_fingers(model: R1ProModel, object_size: tuple[float, float, float],
                                joint_positions: tuple[float, float], clearance: float = 0.0) -> bool:
    """Conservative axis-aligned fit check for a box centered in the gripper."""
    size = np.asarray(object_size, dtype=np.float64)
    if size.shape != (3,) or np.any(size <= 0.0) or clearance < 0.0:
        raise ValueError("object size must be positive and clearance non-negative")
    jaw = model.jaw_geometry(joint_positions)
    axis = np.abs(np.asarray(jaw["opening_axis_gripper"], dtype=np.float64))
    object_span = float(np.dot(size, axis))
    return object_span + 2.0 * clearance <= float(jaw["inner_face_gap_m"])
