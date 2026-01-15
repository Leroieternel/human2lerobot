#!/usr/bin/env python3
"""
Convert a MimicGen / robomimic-style HDF5 dataset into a LeRobot-style dataset
(parquet episodes + mp4 videos + meta/*.json).

Input (MimicGen HDF5):
  data/demo_x/obs/agentview_image            (T,H,W,3) uint8
  data/demo_x/obs/robot0_eye_in_hand_image   (T,H,W,3) uint8   [optional]
  data/demo_x/obs/robot0_eef_pos             (T,3) float
  data/demo_x/obs/robot0_eef_quat            (T,4) float  (robosuite typically xyzw)
  data/demo_x/obs/robot0_gripper_qpos        (T,2) float
  data/demo_x/actions                        (T,7) float  [dx,dy,dz, dax,day,daz, gripper]

Output (LeRobot-like):
  OUTPUT_ROOT/
    data/chunk-000/episode_000000.parquet ...
    videos/chunk-000/<camera_key>/episode_000000.mp4 ...
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    meta/episodes_stats.jsonl

We map observations into:
  observation.state: (8,) = [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
  action:            (7,) = mimicgen actions (delta pose + gripper)

Notes:
- quat -> axis-angle conversion assumes quat order xyzw.
- fps is user-defined (default 20). MimicGen HDF5 typically doesn't store fps.
- Videos are written from the raw uint8 frames inside HDF5.

Dependencies:
  pip install h5py numpy imageio pyarrow tqdm
Optional (for accurate codec/pix_fmt):
  ffprobe available in PATH (from ffmpeg)

Example:
  python mimicgen_to_lerobot.py \
      --input_hdf5 /path/to/mimicgen.hdf5 \
      --output_root /raid/xiangyi/mimicgen_lerobot \
      --fps 20 \
      --write_wrist
"""

import os
import json
import argparse
import subprocess
from typing import Dict, Optional, List, Tuple

import h5py
import numpy as np
import imageio
from tqdm import tqdm
import pyarrow as pa
from pyarrow import parquet as pq
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from mimicgen.config import MIMICGEN_FEATURES

# OUTPUT_ROOT = "/raid/xiangyi/mimicgen_lerobot/scratch"   # change me
# meta_dir = os.path.join(OUTPUT_ROOT, "meta")
# os.makedirs(meta_dir, exist_ok=True)


# -----------------------------
# helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_demos(f: h5py.File, filter_key: Optional[str] = None) -> List[str]:
    if filter_key is not None:
        mask_path = f"mask/{filter_key}"
        if mask_path not in f:
            raise ValueError(f"filter_key '{filter_key}' requested, but '{mask_path}' not found.")
        demos = sorted([x.decode("utf-8") for x in np.array(f[mask_path])])
    else:
        demos = sorted(list(f["data"].keys()))

    def demo_idx(s: str) -> int:
        try:
            return int(s.split("_")[1])
        except Exception:
            return 0

    return sorted(demos, key=demo_idx)


def to_pa_list_array(x: np.ndarray) -> pa.Array:
    """
    x: (N,D) -> pa.list_(float32, D)
    x: (N,)  -> pa.float32
    x: (N,1) -> pa.list_(float32,1)  (we keep list for consistency with your rewards shape)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return pa.array(x.astype(np.float32))
    else:
        D = x.shape[1]
        return pa.array(x.astype(np.float32).tolist(), type=pa.list_(pa.float32(), D))


def write_video(frames: np.ndarray, video_path: str, fps: float):
    # frames: (T,H,W,3) uint8
    imageio.mimsave(video_path, [frames[t] for t in range(frames.shape[0])], fps=fps, macro_block_size=1)


def ffprobe_video(video_path: str) -> Optional[Dict]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        meta = json.loads(out)
        v = meta["streams"][0]
        fps_str = v.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        return {
            "codec": v.get("codec_name", "h264"),
            "pix_fmt": v.get("pix_fmt", "yuv420p"),
            "height": int(v.get("height", 0)),
            "width": int(v.get("width", 0)),
            "fps": fps,
            "channels": int(v.get("channels", 3)),
        }
    except Exception:
        return None


# -----------------------------
# main conversion
# -----------------------------
def convert_mimicgen_to_lerobot(
    input_hdf5: str,
    output_root: str,
    fps: float,
    filter_key: Optional[str] = None,
    include_wrist: bool = True,
):
    # output dirs
    data_dir = os.path.join(output_root, "data", "chunk-000")
    meta_dir = os.path.join(output_root, "meta")
    videos_root = os.path.join(output_root, "videos", "chunk-000")
    ensure_dir(data_dir)
    ensure_dir(meta_dir)
    ensure_dir(videos_root)

    # tasks: mimicgen 通常没有 llm_description；我们用 env_name 当 instruction
    tasks_jsonl: List[Dict] = []
    episodes_jsonl: List[Dict] = []
    task_to_index: Dict[str, int] = {}
    next_task_index = 0

    total_frames = 0

    # cameras (for info.json)
    camera_features = []
    # always write agentview video
    camera_features.append(("observation.images.agentview_image", MIMICGEN_FEATURES["observation.images.agentview_image"]))
    # optionally write wrist
    if include_wrist:
        camera_features.append(("observation.images.robot0_eye_in_hand_image", MIMICGEN_FEATURES["observation.images.robot0_eye_in_hand_image"]))

    with h5py.File(input_hdf5, "r") as f:
        demos = list_demos(f, filter_key)

        # best-effort instruction from env_args
        instr = "mimicgen_task"
        if "data" in f and "env_args" in f["data"].attrs:
            try:
                env_meta = json.loads(f["data"].attrs["env_args"])
                instr = env_meta.get("env_name", instr)
            except Exception:
                pass

        if instr not in task_to_index:
            task_to_index[instr] = next_task_index
            tasks_jsonl.append({"task_index": next_task_index, "task": instr})
            next_task_index += 1

        task_index = task_to_index[instr]

        # for info.json: record actual HW and codec/pixfmt later by probing first written mp4
        video_hw: Dict[str, Tuple[int, int]] = {}  # camera_dir -> (H,W)

        for epi, demo in enumerate(tqdm(demos, desc="[Convert] demos")):
            base = f"data/{demo}"
            obs = f[f"{base}/obs"]

            # --- extract frames (agentview required) ---
            agent_frames = obs["agentview_image"][:]  # (T,84,84,3) uint8
            T = agent_frames.shape[0]

            # wrist optional
            wrist_frames = None
            if include_wrist and "robot0_eye_in_hand_image" in obs:
                wrist_frames = obs["robot0_eye_in_hand_image"][:]
                if wrist_frames.shape[0] != T:
                    raise ValueError(f"{demo}: wrist length {wrist_frames.shape[0]} != agent length {T}")

            # --- write videos ---
            # agentview -> videos/chunk-000/camera_front/episode_xxxxxx.mp4
            cam_front_dir = os.path.join(videos_root, MIMICGEN_FEATURES["observation.images.agentview_image"]["camera_dir"])
            ensure_dir(cam_front_dir)
            cam_front_path = os.path.join(cam_front_dir, f"episode_{epi:06d}.mp4")
            write_video(agent_frames, cam_front_path, fps=fps)

            if epi == 0:
                video_hw[MIMICGEN_FEATURES["observation.images.agentview_image"]["camera_dir"]] = (
                    int(agent_frames.shape[1]), int(agent_frames.shape[2])
                )

            if include_wrist and wrist_frames is not None:
                wrist_dir = os.path.join(videos_root, MIMICGEN_FEATURES["observation.images.robot0_eye_in_hand_image"]["camera_dir"])
                ensure_dir(wrist_dir)
                wrist_path = os.path.join(wrist_dir, f"episode_{epi:06d}.mp4")
                write_video(wrist_frames, wrist_path, fps=fps)
                if epi == 0:
                    video_hw[MIMICGEN_FEATURES["observation.images.robot0_eye_in_hand_image"]["camera_dir"]] = (
                        int(wrist_frames.shape[1]), int(wrist_frames.shape[2])
                    )

            # --- extract numeric arrays exactly as config says ---
            # Each becomes a parquet column with same key name.
            def read_key(rel_key: str) -> np.ndarray:
                # rel_key like "obs/robot0_joint_pos" or "actions"
                if rel_key.startswith("obs/"):
                    k = rel_key.split("/", 1)[1]
                    if k not in obs:
                        raise KeyError(f"Missing obs key '{k}' in {demo}")
                    return obs[k][:]
                else:
                    full = f"{base}/{rel_key}"
                    if full not in f:
                        raise KeyError(f"Missing key '{full}' in HDF5")
                    return f[full][:]

            columns = {}

            # states (all float32)
            for feat_key, spec in MIMICGEN_FEATURES.items():
                if feat_key.startswith("observation.states.") or feat_key in ["observation.rewards", "action"]:
                    src = spec["source_key"]
                    arr = read_key(src)

                    # align length check (per-step arrays)
                    if arr.shape[0] != T:
                        raise ValueError(f"{demo}: '{feat_key}' length {arr.shape[0]} != frames length {T}")

                    # cast float32
                    arr = arr.astype(np.float32)

                    # rewards: (T,) -> (T,1) to match your shape (1,)
                    if feat_key == "observation.rewards":
                        if arr.ndim == 1:
                            arr = arr.reshape(T, 1)
                        elif arr.ndim == 2 and arr.shape[1] == 1:
                            pass
                        else:
                            # if rewards came oddly shaped, force (T,1)
                            arr = arr.reshape(T, 1)

                    # action: keep as (T,7)
                    if feat_key == "action":
                        # ensure (T,7)
                        if arr.ndim != 2 or arr.shape[1] != 7:
                            raise ValueError(f"{demo}: actions shape expected (T,7) but got {arr.shape}")

                    columns[feat_key] = to_pa_list_array(arr) if arr.ndim == 2 else to_pa_list_array(arr)

            # always include indexing columns
            columns["timestamp"] = pa.array((np.arange(T) / float(fps)).astype(np.float32))
            columns["frame_index"] = pa.array(np.arange(T, dtype=np.int32))
            columns["episode_index"] = pa.array([epi] * T, type=pa.int32())
            columns["task_index"] = pa.array([task_index] * T, type=pa.int32())

            table = pa.table(columns)
            pq.write_table(table, os.path.join(data_dir, f"episode_{epi:06d}.parquet"))

            episodes_jsonl.append({"episode_index": epi, "tasks": [instr], "length": int(T)})
            total_frames += int(T)

    # --- write episodes.jsonl / tasks.jsonl ---
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for e in episodes_jsonl:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for t in tasks_jsonl:
            f.write(json.dumps(t) + "\n")

    # --- build info.json ---
    # probe first written videos for codec/pix_fmt if ffprobe exists
    video_infos = {}
    for cam_feat_key, spec in camera_features:
        cam_dir = spec["camera_dir"]
        vid0 = os.path.join(videos_root, cam_dir, "episode_000000.mp4")
        info = ffprobe_video(vid0) or {}
        H, W = video_hw.get(cam_dir, (spec["shape"][0], spec["shape"][1]))
        video_infos[cam_dir] = {
            "height": H,
            "width": W,
            "codec": info.get("codec", "h264"),
            "pix_fmt": info.get("pix_fmt", "yuv420p"),
            "fps": float(fps),
            "channels": 3,
            "has_audio": False,
        }

    features_out = {}

    # video features in info.json use the folder camera_dir
    for cam_feat_key, spec in camera_features:
        cam_dir = spec["camera_dir"]
        v = video_infos[cam_dir]
        features_out[cam_feat_key] = {
            "dtype": "video",
            "shape": [v["height"], v["width"], 3],
            "info": {
                "video.height": v["height"],
                "video.width": v["width"],
                "video.codec": v["codec"],
                "video.pix_fmt": v["pix_fmt"],
                "video.is_depth_map": False,
                "video.fps": v["fps"],
                "video.channels": v["channels"],
                "has_audio": False,
            },
        }

    # numeric features in info.json exactly from spec
    for feat_key, spec in MIMICGEN_FEATURES.items():
        if feat_key.startswith("observation.states.") or feat_key in ["observation.rewards", "action"]:
            features_out[feat_key] = {
                "dtype": spec["dtype"],
                "shape": list(spec["shape"]) if isinstance(spec["shape"], (list, tuple)) else spec["shape"],
                "names": spec["names"],
            }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "mimicgen_robosuite_lerobot",
        "total_episodes": len(episodes_jsonl),
        "total_frames": total_frames,
        "total_tasks": len(tasks_jsonl),
        "total_videos": len(episodes_jsonl) * len(camera_features),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": float(fps),
        "splits": {"train": f"0:{len(episodes_jsonl)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features_out,
        # map camera_key -> folder name, so your loader knows which key corresponds to which folder
        "camera_keys": {k: MIMICGEN_FEATURES[k]["camera_dir"] for k, _ in camera_features},
    }

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    print(f"[OK] info.json saved: {os.path.join(meta_dir, 'info.json')}")

    # --- episodes_stats.jsonl (minimal numeric + rgb stats) ---
    # Keep it simple and fast: compute stats from parquet columns, and rgb stats by sampling frames.
    stats_path = os.path.join(meta_dir, "episodes_stats.jsonl")
    parquet_dir = os.path.join(output_root, "data", "chunk-000")

    def compute_stats(x: np.ndarray) -> Dict:
        x = np.asarray(x)
        return {
            "min": np.min(x, axis=0).reshape(-1).tolist(),
            "max": np.max(x, axis=0).reshape(-1).tolist(),
            "mean": np.mean(x, axis=0).reshape(-1).tolist(),
            "std": np.std(x, axis=0).reshape(-1).tolist(),
            "count": [int(x.shape[0])],
        }

    parquet_files = sorted([p for p in os.listdir(parquet_dir) if p.endswith(".parquet")])
    with open(stats_path, "w") as out_f:
        for epi_idx, fname in enumerate(tqdm(parquet_files, desc="[Stats] episodes_stats.jsonl")):
            table = pq.read_table(os.path.join(parquet_dir, fname))
            df = table.to_pandas()
            N = len(df)

            stats = {}

            # numeric columns
            for feat_key in features_out:
                if feat_key.startswith("observation.states.") or feat_key in ["observation.rewards", "action"]:
                    col = np.vstack(df[feat_key].values).astype(np.float32)
                    stats[feat_key] = compute_stats(col)

            # basic meta
            stats["episode_index"] = compute_stats(np.array([epi_idx]*N, dtype=np.int32).reshape(N, 1))
            stats["index"] = compute_stats(np.arange(N, dtype=np.int32).reshape(N, 1))
            stats["task_index"] = compute_stats(np.array(df["task_index"].values, dtype=np.int32).reshape(N, 1))

            # rgb stats (sample every 10 frames)
            SAMPLE = 10
            for cam_feat_key, spec in camera_features:
                cam_dir = spec["camera_dir"]
                vid_path = os.path.join(videos_root, cam_dir, f"episode_{epi_idx:06d}.mp4")
                reader = imageio.get_reader(vid_path)
                frames = []
                for i, fr in enumerate(reader):
                    if i % SAMPLE == 0:
                        frames.append(fr.astype(np.float32) / 255.0)
                reader.close()

                if len(frames) > 0:
                    frames = np.stack(frames, axis=0)
                    rgb_min = frames.min(axis=(0, 1, 2))
                    rgb_max = frames.max(axis=(0, 1, 2))
                    rgb_mean = frames.mean(axis=(0, 1, 2))
                    rgb_std = frames.std(axis=(0, 1, 2))
                    stats[cam_feat_key] = {
                        "min": rgb_min.reshape(3, 1, 1).tolist(),
                        "max": rgb_max.reshape(3, 1, 1).tolist(),
                        "mean": rgb_mean.reshape(3, 1, 1).tolist(),
                        "std": rgb_std.reshape(3, 1, 1).tolist(),
                        "count": [frames.shape[0]],
                    }

            out_f.write(json.dumps({"episode_index": epi_idx, "stats": stats}) + "\n")

    print(f"[OK] episodes_stats.jsonl saved: {stats_path}")
    print("[DONE] Converted to LeRobot structure.")
    print(f"Output: {output_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_hdf5", type=str, required=True, help="MimicGen/robomimic HDF5 file")
    ap.add_argument("--output_root", type=str, required=True, help="Output root folder")
    ap.add_argument("--fps", type=float, default=20.0, help="FPS for written mp4 videos")
    ap.add_argument("--filter_key", type=str, default=None, help="Optional: robomimic mask key (e.g. train/valid)")
    ap.add_argument("--no_wrist", action="store_true", help="Do NOT export robot0_eye_in_hand video")
    args = ap.parse_args()

    convert_mimicgen_to_lerobot(
        input_hdf5=args.input_hdf5,
        output_root=args.output_root,
        fps=args.fps,
        filter_key=args.filter_key,
        include_wrist=(not args.no_wrist),
    )


if __name__ == "__main__":
    main()
