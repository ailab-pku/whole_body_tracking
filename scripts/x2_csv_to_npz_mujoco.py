"""Convert a named-column X2 Ultra motion CSV to BeyondMimic NPZ and render it with MuJoCo.

The standard BeyondMimic arrays contain all 31 X2 joints, including the two
head joints, in MuJoCo model order. Isaac Lab resolves joint_names into its
own default order when loading. Full model-order arrays are retained as metadata.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="X2 scene.xml or x2_ultra.xml")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=140.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--crf", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[dict[str, str], list[str], np.ndarray]:
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    header_line = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.startswith("#"):
                key, separator, value = line[1:].strip().partition(":")
                if separator:
                    metadata[key.strip()] = value.strip()
                continue
            header = next(csv.reader([line]))
            header_line = line_number
            break
    if header is None:
        raise ValueError(f"No CSV header found in {path}")
    values = np.loadtxt(path, delimiter=",", comments="#", skiprows=header_line, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != len(header):
        raise ValueError(f"Header has {len(header)} columns but data has {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("CSV contains NaN or infinite values")
    return metadata, header, values


def ensure_quaternion_continuity(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions.copy()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_interval_velocity(start: np.ndarray, end: np.ndarray, duration: np.ndarray) -> np.ndarray:
    conjugate = start.copy()
    conjugate[..., 1:] *= -1.0
    relative = quaternion_multiply(end, conjugate)
    relative[relative[:, 0] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], -1.0, 1.0))
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1.0e-12)
    return relative[:, 1:] * scale[:, None] / duration[:, None]


def quaternion_velocity(quaternions: np.ndarray, times: np.ndarray) -> np.ndarray:
    quaternions = ensure_quaternion_continuity(quaternions)
    velocity = np.empty((len(times), 3), dtype=np.float64)
    velocity[0] = quaternion_interval_velocity(quaternions[0:1], quaternions[1:2], times[1:2] - times[0:1])[0]
    velocity[-1] = quaternion_interval_velocity(
        quaternions[-2:-1], quaternions[-1:], times[-1:] - times[-2:-1]
    )[0]
    velocity[1:-1] = quaternion_interval_velocity(
        quaternions[:-2], quaternions[2:], times[2:] - times[:-2]
    )
    return velocity


def start_ffmpeg(path: Path, width: int, height: int, fps: float, crf: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.9g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> None:
    args = parse_args()
    metadata, header, values = read_csv(args.input_csv)
    column = {name: index for index, name in enumerate(header)}
    required_root_columns = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
    missing_root = [name for name in required_root_columns if name not in column]
    if missing_root:
        raise ValueError(f"Missing root columns: {missing_root}")

    csv_joint_names = [name.removeprefix("dof_") for name in header if name.startswith("dof_")]
    if len(csv_joint_names) != 31 or len(set(csv_joint_names)) != 31:
        raise ValueError(f"Expected 31 unique dof_* columns, got {len(csv_joint_names)}")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
    ]
    if set(csv_joint_names) != set(model_joint_names):
        raise ValueError(
            f"CSV/model joint mismatch; CSV-only={sorted(set(csv_joint_names) - set(model_joint_names))}, "
            f"model-only={sorted(set(model_joint_names) - set(csv_joint_names))}"
        )

    times = values[:, column["time"]]
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Time must be strictly increasing and contain at least two samples")
    fps = float(metadata.get("sample_rate", 1.0 / np.median(np.diff(times))))
    root_pos = values[:, [column["root_x"], column["root_y"], column["root_z"]]]
    # CSV uses xyzw; MuJoCo and BeyondMimic body arrays use wxyz.
    root_quat = values[:, [column["root_qw"], column["root_qx"], column["root_qy"], column["root_qz"]]]
    root_quat = ensure_quaternion_continuity(root_quat)
    joint_pos_full = np.column_stack([values[:, column[f"dof_{name}"]] for name in model_joint_names])
    joint_vel_full = np.gradient(joint_pos_full, times, axis=0)

    body_ids = np.arange(1, model.nbody, dtype=np.int32)  # omit MuJoCo's world body
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(index)) for index in body_ids]
    if len(body_names) != 32 or body_names[0] != "pelvis":
        raise ValueError(f"Expected pelvis + 31 articulated bodies, got {body_names}")

    num_frames = len(times)
    body_pos_w = np.empty((num_frames, len(body_ids), 3), dtype=np.float64)
    body_quat_w = np.empty((num_frames, len(body_ids), 4), dtype=np.float64)

    renderer = None
    ffmpeg = None
    camera = None
    if args.output_video is not None:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        ffmpeg = start_ffmpeg(args.output_video, args.width, args.height, fps, args.crf)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation

    qpos_addresses = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in model_joint_names
    }
    try:
        for frame_index in range(num_frames):
            data.qpos[:3] = root_pos[frame_index]
            data.qpos[3:7] = root_quat[frame_index]
            for joint_index, name in enumerate(model_joint_names):
                data.qpos[qpos_addresses[name]] = joint_pos_full[frame_index, joint_index]
            mujoco.mj_forward(model, data)
            body_pos_w[frame_index] = data.xpos[body_ids]
            body_quat_w[frame_index] = data.xquat[body_ids]

            if renderer is not None and ffmpeg is not None and camera is not None:
                camera.lookat[:] = root_pos[frame_index] + np.array([0.0, 0.0, 0.35])
                renderer.update_scene(data, camera=camera)
                ffmpeg.stdin.write(renderer.render().tobytes())
            if frame_index % 500 == 0 or frame_index == num_frames - 1:
                print(f"Processed {frame_index + 1}/{num_frames} frames", flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        if ffmpeg is not None:
            if ffmpeg.stdin is not None:
                ffmpeg.stdin.close()
            return_code = ffmpeg.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")

    body_lin_vel_w = np.gradient(body_pos_w, times, axis=0)
    body_ang_vel_w = np.empty_like(body_lin_vel_w)
    for body_index in range(len(body_ids)):
        body_ang_vel_w[:, body_index] = quaternion_velocity(body_quat_w[:, body_index], times)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        fps=np.asarray([fps], dtype=np.float32),
        joint_pos=joint_pos_full.astype(np.float32),
        joint_vel=joint_vel_full.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        joint_pos_full=joint_pos_full.astype(np.float32),
        joint_vel_full=joint_vel_full.astype(np.float32),
        joint_names=np.asarray(model_joint_names),
        full_joint_names=np.asarray(model_joint_names),
        body_names=np.asarray(body_names),
        time=times.astype(np.float64),
    )
    print(f"Saved NPZ: {args.output_npz}")
    if args.output_video is not None:
        print(f"Saved video: {args.output_video}")


if __name__ == "__main__":
    main()
