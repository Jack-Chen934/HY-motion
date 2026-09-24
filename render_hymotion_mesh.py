"""Render a HY-Motion clip as a complete skinned human mesh.

This is the visual reference path for the Genesis experiment.  It uses the
official HY-Motion WoodenMesh linear-blend-skinning model and the original
SMPL-H ``poses``/``trans`` fields, so it does not pass through the 22-joint
MJCF retargeter.  The resulting MP4/GIF is therefore the closest local
comparison to the official HY-Motion mesh visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk
from vtk.util import numpy_support


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = WORKSPACE / "outputs/stability/final_full_cooperation_seed2026_000.npz"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/final"
DEFAULT_MODEL_DIR = WORKSPACE / "HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden"
HY_TO_GENESIS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="final_full_cooperation_seed2026_mesh")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame.")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--no-mp4", action="store_true")
    return parser.parse_args()


def load_wooden_mesh(model_dir: Path):
    repo = model_dir.parents[4]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hymotion.pipeline.body_model import WoodenMesh

    return WoodenMesh(model_path=str(model_dir))


def render_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    root: np.ndarray,
    height_limits: tuple[float, float],
    frame_index: int,
    total_frames: int,
    dpi: int,
) -> np.ndarray:
    # Genesis coordinates: X, Y-forward, Z-up.  Matplotlib's 3D camera is
    # configured with Y as the forward/depth axis and Z as vertical.
    face_vertices = vertices[faces]
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    mesh = Poly3DCollection(
        face_vertices,
        facecolor=(0.70, 0.48, 0.30, 1.0),
        edgecolor="none",
        linewidth=0.0,
        antialiased=False,
    )
    ax.add_collection3d(mesh)
    # Follow the root along the walking direction.  A single world-fixed box
    # would include the whole 2 m trajectory and make the human appear tiny.
    xmin, xmax = float(root[0] - 0.85), float(root[0] + 0.85)
    ymin, ymax = float(root[1] - 0.95), float(root[1] + 0.95)
    zmin, zmax = height_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_xlabel("X", color="#d1d5db")
    ax.set_ylabel("Y", color="#d1d5db")
    ax.set_zlabel("Z / up", color="#d1d5db")
    ax.set_title(
        f"HY-Motion skinned mesh — frame {frame_index + 1}/{total_frames}",
        color="white",
    )
    ax.view_init(elev=10, azim=-72)
    ax.grid(False)
    ax.tick_params(colors="#d1d5db", labelsize=7)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_sequence(
    vertices: np.ndarray,
    faces: np.ndarray,
    roots: np.ndarray,
    height_limits: tuple[float, float],
    frame_indices: list[int],
    dpi: int,
) -> list[np.ndarray]:
    """Render frames while reusing one Matplotlib scene and mesh object."""
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    xmin, xmax = -0.85, 0.85
    ymin, ymax = -0.95, 0.95
    zmin, zmax = height_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_xlabel("X", color="#d1d5db")
    ax.set_ylabel("Y", color="#d1d5db")
    ax.set_zlabel("Z / up", color="#d1d5db")
    ax.view_init(elev=10, azim=-72)
    ax.grid(False)
    ax.tick_params(colors="#d1d5db", labelsize=7)
    mesh = Poly3DCollection(
        vertices[frame_indices[0]][faces],
        facecolor=(0.70, 0.48, 0.30, 1.0),
        edgecolor="none",
        linewidth=0.0,
        antialiased=False,
    )
    ax.add_collection3d(mesh)
    canvas = FigureCanvasAgg(fig)
    frames = []
    for index in frame_indices:
        root = roots[index]
        ax.set_xlim(float(root[0] - 0.85), float(root[0] + 0.85))
        ax.set_ylim(float(root[1] - 0.95), float(root[1] + 0.95))
        ax.set_title(
            f"HY-Motion skinned mesh — frame {index + 1}/{len(vertices)}",
            color="white",
        )
        mesh.set_verts(vertices[index][faces])
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


def render_sequence_vtk(
    vertices: np.ndarray,
    faces: np.ndarray,
    roots: np.ndarray,
    height_limits: tuple[float, float],
    frame_indices: list[int],
    width: int = 800,
    height: int = 640,
) -> list[np.ndarray]:
    """Render the mesh with VTK off-screen OpenGL for practical throughput."""
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(vertices[frame_indices[0]], deep=True))
    cell_array = vtk.vtkCellArray()
    face_array = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces.astype(np.int64)]
    ).reshape(-1)
    cell_array.SetCells(len(faces), numpy_support.numpy_to_vtkIdTypeArray(face_array, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cell_array)
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.70, 0.48, 0.30)
    actor.GetProperty().SetSpecular(0.1)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.067, 0.094, 0.153)
    renderer.AddActor(actor)
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(38.0)
    camera.SetClippingRange(0.01, 100.0)
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()

    frames = []
    zmin, zmax = height_limits
    for index in frame_indices:
        points.SetData(numpy_support.numpy_to_vtk(vertices[index], deep=True))
        points.Modified()
        polydata.Modified()
        root = roots[index]
        camera.SetPosition(float(root[0] + 2.6), float(root[1] - 3.8), float(root[2] + 1.6))
        camera.SetFocalPoint(float(root[0]), float(root[1]), float((zmin + zmax) * 0.5))
        renderer.ResetCameraClippingRange()
        render_window.Render()
        window_to_image.Modified()
        window_to_image.Update()
        vtk_image = window_to_image.GetOutput()
        frame = numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        frame = frame.reshape(height, width, 3)[::-1].copy()
        frames.append(frame)
    render_window.Finalize()
    return frames


def run(args: argparse.Namespace) -> dict[str, object]:
    motion_path = args.motion.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if not (model_dir / "v_template.bin").is_file():
        raise FileNotFoundError(model_dir / "v_template.bin")
    if args.stride < 1 or args.fps <= 0 or args.dpi < 40:
        raise ValueError("stride >= 1, fps > 0 and dpi >= 40 are required")

    with np.load(motion_path, allow_pickle=False) as data:
        poses = torch.from_numpy(data["poses"]).float()
        trans_hy = data["trans"].astype(np.float32)
        source_fps = float(data["fps"][0]) if "fps" in data else 30.0
    mesh_model = load_wooden_mesh(model_dir)
    with torch.inference_mode():
        result = mesh_model({"poses": poses, "trans": torch.from_numpy(trans_hy)})
    vertices_hy = result["vertices"].cpu().numpy().astype(np.float32)
    vertices = vertices_hy @ HY_TO_GENESIS.T
    faces = np.asarray(mesh_model.faces, dtype=np.int32)

    # Keep a fixed vertical scale for the whole clip while the horizontal
    # camera follows the pelvis.  This shows the complete body at a useful
    # scale throughout the walk.
    minimum = vertices.min(axis=(0, 1))
    maximum = vertices.max(axis=(0, 1))
    z_margin = 0.08
    height_limits = (
        min(-0.02, float(minimum[2] - z_margin)),
        max(1.90, float(maximum[2] + z_margin)),
    )
    # The WoodenMesh keypoints include the root translation and use the same
    # HY coordinate system as the vertices before this conversion.
    with torch.inference_mode():
        mesh_joints_hy = result["keypoints3d"].cpu().numpy().astype(np.float32)
    roots = (mesh_joints_hy[:, 0] + trans_hy) @ HY_TO_GENESIS.T

    frame_indices = list(range(0, len(vertices), args.stride))
    if frame_indices[-1] != len(vertices) - 1:
        frame_indices.append(len(vertices) - 1)
    frames = render_sequence_vtk(vertices, faces, roots, height_limits, frame_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    png_path = output_dir / f"{prefix}_first_frame.png"
    gif_path = output_dir / f"{prefix}.gif"
    imageio.imwrite(png_path, frames[0])
    imageio.mimsave(gif_path, frames, duration=1.0 / args.fps * args.stride, loop=0)
    mp4_path = output_dir / f"{prefix}.mp4"
    mp4_error = None
    if not args.no_mp4:
        try:
            with imageio.get_writer(mp4_path, fps=args.fps / args.stride, codec="libx264", quality=7) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception as exc:  # pragma: no cover - codec availability is host-specific
            mp4_error = f"{type(exc).__name__}: {exc}"

    report = {
        "status": "success" if mp4_error is None else "partial",
        "motion": str(motion_path),
        "model_dir": str(model_dir),
        "vertices": int(vertices.shape[1]),
        "faces": int(faces.shape[0]),
        "source_frames": int(len(vertices)),
        "source_fps": source_fps,
        "rendered_frames": len(frames),
        "render_fps": args.fps / args.stride,
        "coordinate_conversion": "HY-Motion Y-up -> Genesis Z-up",
        "output_png": str(png_path),
        "output_gif": str(gif_path),
        "output_mp4": str(mp4_path) if mp4_error is None and not args.no_mp4 else None,
        "mp4_error": mp4_error,
        "note": "Complete skinned visual mesh; Genesis MJCF collision proxy is not included in this offline renderer.",
    }
    report_path = output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
