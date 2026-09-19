"""Export a HY-Motion SMPL-H clip into a Genesis-friendly keypoint cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymotion.pipeline.body_model import WoodenMesh


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = WORKSPACE / "outputs/stability/final_full_cooperation_seed2026_000.npz"
DEFAULT_OUTPUT = WORKSPACE / "genesis-human-experiment/motions/prepared/final_full_cooperation_seed2026.npz"
WOODEN_MODEL_DIR = WORKSPACE / "HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def export_motion(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    data = np.load(input_path, allow_pickle=False)
    required = {"poses", "Rh", "trans"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    poses = torch.from_numpy(data["poses"]).float()
    trans = torch.from_numpy(data["trans"]).float()
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"Expected poses shape (T, 156), got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"Expected trans shape {(poses.shape[0], 3)}, got {trans.shape}")

    model = WoodenMesh(model_path=str(WOODEN_MODEL_DIR))
    with torch.inference_mode():
        result = model({"poses": poses, "trans": trans})

    # WoodenMesh returns keypoints without the global root translation.
    keypoints_hy = result["keypoints3d"].cpu().numpy() + trans.numpy()[:, None, :]
    if not np.isfinite(keypoints_hy).all():
        raise FloatingPointError("keypoints contain NaN or Inf")

    parents = np.fromfile(WOODEN_MODEL_DIR / "kintree.bin", dtype=np.int32)
    with (WOODEN_MODEL_DIR / "joint_names.json").open("r", encoding="utf-8") as handle:
        joint_names = np.asarray(json.load(handle), dtype="U32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        keypoints_hy=keypoints_hy.astype(np.float32),
        poses=data["poses"].astype(np.float32),
        Rh=data["Rh"].astype(np.float32),
        trans=data["trans"].astype(np.float32),
        parents=parents,
        joint_names=joint_names,
        fps=np.array([30.0], dtype=np.float32),
    )

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"frames={keypoints_hy.shape[0]} joints={keypoints_hy.shape[1]} fps=30")
    print(f"keypoints_min={keypoints_hy.min(axis=(0, 1)).tolist()}")
    print(f"keypoints_max={keypoints_hy.max(axis=(0, 1)).tolist()}")


if __name__ == "__main__":
    args = parse_args()
    export_motion(args.input, args.output)
