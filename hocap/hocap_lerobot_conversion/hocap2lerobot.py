#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import imageio
import pyarrow as pa
from pyarrow import parquet as pq
from tqdm import tqdm
import cv2

from hocap_config import HOCAP_CONFIG
from lerobot_utils import compute_and_write_episode_stats
from extract_delta_eef import (
    load_extrinsics_yaml,
    load_episode_world_joints_from_camera_folder,
    compute_delta_eef_from_world_joints,
)

CHUNK_SIZE = 1000


# -----------------------------
# utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_video(frames: np.ndarray, path: str, fps: float):
    """Write MP4 with H.264 (libx264). frames: (T,H,W,3) uint8 RGB"""
    writer = imageio.get_writer(
        path,
        format="ffmpeg",
        fps=float(fps),
        codec="libx264",     # H.264
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    finally:
        writer.close()


def to_pa_list_array(x: np.ndarray) -> pa.Array:
    x = np.asarray(x)
    if x.ndim == 1:
        return pa.array(x.astype(np.float32))
    D = x.shape[1]
    return pa.array(x.astype(np.float32).tolist(), type=pa.list_(pa.float32(), D))


def list_episode_dirs(subject_dir: str, subset_size: int = None) -> List[str]:
    eps = [os.path.join(subject_dir, d) for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
    eps = sorted(eps)
    if subset_size is not None:
        eps = eps[: int(subset_size)]
    return eps


def load_frames_from_camera_folder(cam_dir: str, resize_hw: Tuple[int, int] = None) -> np.ndarray:
    imgs = sorted(glob.glob(os.path.join(cam_dir, "color_*.jpg")))
    if not imgs:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)

    frames = []
    for p in imgs:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            H, W = resize_hw
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(im)

    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0).astype(np.uint8)


def get_selected_image_specs(num_views: int = None) -> List[Tuple[str, Dict[str, Any]]]:
    items = list(HOCAP_CONFIG["images"].items())  # keep insertion order
    if num_views is None:
        return items
    return items[: max(0, int(num_views))]


def parse_subjects(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def build_info_json(dataset_name: str, fps: float, total_episodes: int, total_frames: int,
                    num_views: int, mode: str) -> Dict[str, Any]:
    """
    mode in {"left_hand","right_hand","both_hands"}
    - left/right: action shape (6,), only one state key
    - both: action shape (12,), two state keys
    """
    features: Dict[str, Any] = {}

    for img_key, spec in get_selected_image_specs(num_views):
        if img_key == "hololens":
            H, W = 480, 640
        else:
            H, W, _ = spec["shape"]

        features[f"observation.images.{img_key}"] = {
            "dtype": "video",
            "shape": [int(H), int(W), 3],
            "info": {
                "video.height": int(H),
                "video.width": int(W),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    # states (subset by mode)
    if mode in ("left_hand", "both_hands"):
        spec = HOCAP_CONFIG["states"]["left_delta_end_effector"]
        features["observation.states.left_delta_end_effector"] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "names": spec.get("names", None),
        }
    if mode in ("right_hand", "both_hands"):
        spec = HOCAP_CONFIG["states"]["right_delta_end_effector"]
        features["observation.states.right_delta_end_effector"] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "names": spec.get("names", None),
        }

    act_dim = 12 if mode == "both_hands" else 6
    features["action"] = {"dtype": "float32", "shape": [act_dim], "names": None}

    total_chunks = int((total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE) if total_episodes > 0 else 1

    return {
        "codebase_version": "v2.1",
        "dataset_name": dataset_name,
        "fps": float(fps),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_episodes),
        "total_chunks": int(total_chunks),
        "chunks_size": int(CHUNK_SIZE),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def _hand_valid_from_delta(delta_6: np.ndarray) -> bool:
    """
    delta_6: (T-1,6) possibly contains NaN if missing.
    valid if any finite element exists.
    """
    if delta_6 is None:
        return False
    return bool(np.isfinite(delta_6).any())


# -----------------------------
# dataset contexts (3 outputs)
# -----------------------------
def _init_dataset_ctx(root: str):
    ensure_dir(root)
    ctx = {
        "root": root,
        "data_root": os.path.join(root, "data"),
        "videos_root": os.path.join(root, "videos"),
        "meta_root": os.path.join(root, "meta"),
        "epi": 0,
        "total_frames": 0,
        "episodes_meta": [],
        "tasks": [],
        "stats_fh": None,
    }
    ensure_dir(ctx["data_root"])
    ensure_dir(ctx["videos_root"])
    ensure_dir(ctx["meta_root"])

    ctx["paths"] = {
        "tasks": os.path.join(ctx["meta_root"], "tasks.jsonl"),
        "episodes": os.path.join(ctx["meta_root"], "episodes.jsonl"),
        "info": os.path.join(ctx["meta_root"], "info.json"),
        "episode_stats": os.path.join(ctx["meta_root"], "episode_stats.jsonl"),
    }
    ctx["stats_fh"] = open(ctx["paths"]["episode_stats"], "w", encoding="utf-8")
    return ctx


def _close_dataset_ctx(ctx):
    if ctx["stats_fh"] is not None:
        ctx["stats_fh"].close()
        ctx["stats_fh"] = None


def _flush_dataset_ctx(ctx, fps: float, num_views: int, mode: str):
    # write meta files
    with open(ctx["paths"]["tasks"], "w", encoding="utf-8") as f:
        for t in ctx["tasks"]:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(ctx["paths"]["episodes"], "w", encoding="utf-8") as f:
        for e in ctx["episodes_meta"]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    info = build_info_json(
        dataset_name=os.path.basename(os.path.abspath(ctx["root"].rstrip("/"))),
        fps=fps,
        total_episodes=ctx["epi"],
        total_frames=ctx["total_frames"],
        num_views=num_views,
        mode=mode,
    )
    with open(ctx["paths"]["info"], "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# -----------------------------
# convert
# -----------------------------
def convert(
    input_root: str,
    output_root: str,
    subjects: List[int],
    fps: float,
    extrinsics_yaml: str,
    subset_size: int = None,
    num_views: int = None,
):
    selected_views = get_selected_image_specs(num_views)

    # init 3 output datasets
    left_ctx = _init_dataset_ctx(os.path.join(output_root, "left_hand"))
    right_ctx = _init_dataset_ctx(os.path.join(output_root, "right_hand"))
    both_ctx = _init_dataset_ctx(os.path.join(output_root, "both_hands"))

    extr_dict, rs_master = load_extrinsics_yaml(extrinsics_yaml)
    rs_master = str(rs_master) if rs_master is not None else HOCAP_CONFIG["images"]["camera_5"]["source_key"]

    try:
        for sx in subjects:
            subj_dir = os.path.join(input_root, f"subject_{sx}")
            if not os.path.isdir(subj_dir):
                continue

            ep_dirs = list_episode_dirs(subj_dir, subset_size=subset_size)

            for ep_dir in tqdm(ep_dirs, desc=f"subject_{sx}", leave=False):
                episode_id = os.path.basename(ep_dir)
                instruction = f"subject_{sx}_{episode_id}"

                # ---- 1) load frames for selected views ----
                frames_by_key: Dict[str, np.ndarray] = {}
                T_candidates: List[int] = []

                for img_key, spec_img in selected_views:
                    cam_name = spec_img["source_key"]
                    cam_dir = os.path.join(ep_dir, cam_name)
                    frames = load_frames_from_camera_folder(
                        cam_dir,
                        resize_hw=(480, 640) if img_key == "hololens" else None,
                    )
                    frames_by_key[img_key] = frames
                    if frames.shape[0] > 0:
                        T_candidates.append(int(frames.shape[0]))

                if not T_candidates:
                    continue
                T_img = min(T_candidates)

                # ---- 2) compute delta_eef from labels ----
                label_cam = rs_master
                label_dir = os.path.join(ep_dir, label_cam)
                if not os.path.isdir(label_dir):
                    label_cam = HOCAP_CONFIG["images"]["camera_5"]["source_key"]
                    label_dir = os.path.join(ep_dir, label_cam)

                try:
                    world_all_steps, _ = load_episode_world_joints_from_camera_folder(
                        camera_folder=label_dir,
                        extrinsics_yaml_path=extrinsics_yaml,
                    )
                except Exception:
                    # cannot compute hands -> skip this episode
                    continue

                T_lbl = int(world_all_steps.shape[0])
                T = min(T_img, T_lbl)
                world_all_steps = world_all_steps[:T]

                delta_dict = compute_delta_eef_from_world_joints(world_all_steps)  # left/right: (T-1,6)
                left_delta = delta_dict.get("left_eef", None)
                right_delta = delta_dict.get("right_eef", None)

                left_valid = _hand_valid_from_delta(left_delta)
                right_valid = _hand_valid_from_delta(right_delta)

                # decide which dataset bucket
                if left_valid and right_valid:
                    mode = "both_hands"
                    ctx = both_ctx
                elif left_valid and (not right_valid):
                    mode = "left_hand"
                    ctx = left_ctx
                elif right_valid and (not left_valid):
                    mode = "right_hand"
                    ctx = right_ctx
                else:
                    # neither hand valid -> skip
                    continue

                # ---- 3) make per-mode states/action (length T) ----
                # convert delta (T-1) -> state (T) with leading zero
                if left_valid:
                    left_state = np.zeros((T, 6), dtype=np.float32)
                    if T > 1:
                        left_state[1:] = left_delta[: T - 1]
                else:
                    left_state = None

                if right_valid:
                    right_state = np.zeros((T, 6), dtype=np.float32)
                    if T > 1:
                        right_state[1:] = right_delta[: T - 1]
                else:
                    right_state = None

                if mode == "both_hands":
                    action = np.concatenate([left_state, right_state], axis=1).astype(np.float32)  # (T,12)
                elif mode == "left_hand":
                    action = left_state.astype(np.float32)   # (T,6)
                else:
                    action = right_state.astype(np.float32)  # (T,6)

                # ---- 4) slice frames to T ----
                for k in list(frames_by_key.keys()):
                    if frames_by_key[k].shape[0] >= T:
                        frames_by_key[k] = frames_by_key[k][:T]
                    else:
                        frames_by_key[k] = np.zeros((0, 0, 0, 3), dtype=np.uint8)

                # ---- 5) output paths by ctx ----
                epi = ctx["epi"]
                task_idx = epi  # one task per episode (per dataset)
                ctx["tasks"].append({"task_index": int(task_idx), "task": instruction})

                episode_chunk = epi // CHUNK_SIZE
                chunk_name = f"chunk-{episode_chunk:03d}"
                data_dir = os.path.join(ctx["data_root"], chunk_name)
                vids_dir = os.path.join(ctx["videos_root"], chunk_name)
                ensure_dir(data_dir)
                ensure_dir(vids_dir)

                # ---- 6) write videos + collect video_paths ----
                video_paths: Dict[str, str] = {}
                for img_key, frames in frames_by_key.items():
                    if frames.shape[0] != T:
                        continue
                    out_dir = os.path.join(vids_dir, f"observation.images.{img_key}")
                    ensure_dir(out_dir)
                    out_path = os.path.join(out_dir, f"episode_{epi:06d}.mp4")
                    write_video(frames, out_path, fps=fps)
                    video_paths[f"observation.images.{img_key}"] = out_path

                # ---- 7) write parquet (keys depend on mode) ----
                cols = {
                    "action": to_pa_list_array(action),
                    "timestamp": pa.array((np.arange(T) / float(fps)).astype(np.float32)),
                    "frame_index": pa.array(np.arange(T, dtype=np.int32)),
                    "episode_index": pa.array([epi] * T, type=pa.int32()),
                    "task_index": pa.array([int(task_idx)] * T, type=pa.int32()),
                }

                numeric_states = {}
                if mode in ("left_hand", "both_hands"):
                    cols["observation.states.left_delta_end_effector"] = to_pa_list_array(left_state)
                    numeric_states["observation.states.left_delta_end_effector"] = left_state
                if mode in ("right_hand", "both_hands"):
                    cols["observation.states.right_delta_end_effector"] = to_pa_list_array(right_state)
                    numeric_states["observation.states.right_delta_end_effector"] = right_state

                pq.write_table(pa.table(cols), os.path.join(data_dir, f"episode_{epi:06d}.parquet"))

                ctx["episodes_meta"].append(
                    {
                        "episode_index": int(epi),
                        "episode_chunk": int(episode_chunk),
                        "tasks": [instruction],
                        "length": int(T),
                    }
                )

                # ---- 8) episode stats (only selected keys) ----
                compute_and_write_episode_stats(
                    ctx["stats_fh"],
                    episode_index=epi,
                    cfg=HOCAP_CONFIG,
                    video_paths=video_paths,
                    numeric_states=numeric_states,
                    action=action,
                )

                ctx["epi"] += 1
                ctx["total_frames"] += int(T)

    finally:
        _close_dataset_ctx(left_ctx)
        _close_dataset_ctx(right_ctx)
        _close_dataset_ctx(both_ctx)

    # write info/meta for each dataset (after counters are final)
    _flush_dataset_ctx(left_ctx, fps=fps, num_views=num_views, mode="left_hand")
    _flush_dataset_ctx(right_ctx, fps=fps, num_views=num_views, mode="right_hand")
    _flush_dataset_ctx(both_ctx, fps=fps, num_views=num_views, mode="both_hands")

    print(
        "[DONE]\n"
        f"  left_hand : {left_ctx['epi']} episodes\n"
        f"  right_hand: {right_ctx['epi']} episodes\n"
        f"  both_hands: {both_ctx['epi']} episodes\n"
        f"  -> {output_root}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, default="/mnt/central_storage/data_pool_world/HO-Cap/datasets")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--subjects", type=str, default="1-9")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--extrinsics_yaml", type=str, default="/mnt/central_storage/data_pool_world/HO-Cap/datasets/calibration/extrinsics/extrinsics_20231014.yaml")
    ap.add_argument("--subset_size", type=int, default=None, help="Take first N episodes per subject for quick validation.")
    ap.add_argument("--num_views", type=int, default=None, help="Use first N views from HOCAP_CONFIG['images'].")
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    convert(
        input_root=args.input_root,
        output_root=args.output_root,
        subjects=subjects,
        fps=float(args.fps),
        extrinsics_yaml=args.extrinsics_yaml,
        subset_size=args.subset_size,
        num_views=args.num_views,
    )


if __name__ == "__main__":
    main()
