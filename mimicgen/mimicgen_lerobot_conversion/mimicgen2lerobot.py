#!/usr/bin/env python3
"""
mimicgen2lerobot.py (group-by-config + stats by lerobot_utils)

Batch convert MimicGen/robomimic-style HDF5 datasets under input_root into
LeRobot-style datasets (parquet episodes + mp4 videos + meta/info.json),
grouped by CONFIG schema:
  - general
  - coffee
  - hammer_kitchen
  - three_assembly

Robot type is only used to standardize gripper -> 1D via gripper_utils.standardize_gripper().

DEFAULT:
- Always writes meta/episodes_stats.jsonl using LeRobot official stats (via lerobot_utils.py)

Chunking:
- data/chunk-000: episode_000000 ~ episode_000999
- data/chunk-001: episode_001000 ~ episode_001999
- ...
(and same for videos/)
"""

import os
import json
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np
import imageio
import pyarrow as pa
from pyarrow import parquet as pq
from tqdm import tqdm

from gripper_utils import standardize_gripper
from lerobot_utils import compute_and_write_episode_stats

# configs
from mimicgen_config_general import MIMICGEN_MIN_CONFIG
from mimicgen_config_coffee import MIMICGEN_COFFEE_CONFIG
from mimicgen_config_three_assembly import MIMICGEN_THREE_ASSEMBLY_CONFIG
from mimicgen_config_hammer_kitchen import MIMICGEN_HAMMER_KITCHEN_CONFIG

CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("coffee", MIMICGEN_COFFEE_CONFIG),
    ("three_assembly", MIMICGEN_THREE_ASSEMBLY_CONFIG),
    ("hammer_kitchen", MIMICGEN_HAMMER_KITCHEN_CONFIG),
    ("general", MIMICGEN_MIN_CONFIG),  # fallback
]


# -----------------------------
# basic utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def iter_hdf5_files(root_dir: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith((".hdf5", ".h5")):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def read_env_args(f: h5py.File) -> Optional[Dict[str, Any]]:
    if "data" not in f or "env_args" not in f["data"].attrs:
        return None
    try:
        return json.loads(f["data"].attrs["env_args"])
    except Exception:
        return None


def get_robot_type(h5_path: str, env_args: Optional[Dict[str, Any]]) -> str:
    robots = None
    if env_args:
        robots = (env_args.get("env_kwargs", {}) or {}).get("robots", None)
    if isinstance(robots, list) and robots:
        r = str(robots[0]).lower()
        if "ur5" in r:
            return "UR5e"
        if "panda" in r:
            return "Panda"
        if "sawyer" in r:
            return "Sawyer"
        if "iiwa" in r or "kuka" in r:
            return "IIWA"

    bn = os.path.basename(h5_path).lower()
    if "ur5e" in bn or "_ur5" in bn:
        return "UR5e"
    if "_panda" in bn:
        return "Panda"
    if "_sawyer" in bn:
        return "Sawyer"
    if "_iiwa" in bn:
        return "IIWA"
    return "unknown"


def canonical_robot_type_for_gripper(robot_type: str) -> str:
    rt = robot_type.lower()
    if rt == "panda":
        return "Panda"
    if rt in ["ur5", "ur5e"]:
        return "UR5e"
    if rt == "iiwa":
        return "IIWA"
    if rt == "sawyer":
        return "Sawyer"
    return "Panda"  # best-effort fallback


def get_gripper_type(env_args: Optional[Dict[str, Any]]) -> Optional[str]:
    if not env_args:
        return None
    gtypes = (env_args.get("env_kwargs", {}) or {}).get("gripper_types", None)
    if not (isinstance(gtypes, list) and gtypes):
        return None
    gt = str(gtypes[0]).lower()
    if "robotiq" in gt and ("85" in gt or "twofing" in gt):
        return "Robotiq85"
    if "robotiq" in gt and "140" in gt:
        return "Robotiq140"
    if "panda" in gt:
        return "PandaGripper"
    if "rethink" in gt:
        return "RethinkGripper"
    return None


def list_demos(f: h5py.File, filter_key: Optional[str] = None) -> List[str]:
    if filter_key:
        mask_path = f"mask/{filter_key}"
        if mask_path not in f:
            raise KeyError(f"mask key not found: {mask_path}")
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
    x = np.asarray(x)
    if x.ndim == 1:
        return pa.array(x.astype(np.float32))
    D = x.shape[1]
    return pa.array(x.astype(np.float32).tolist(), type=pa.list_(pa.float32(), D))


def write_video(frames: np.ndarray, path: str, fps: float):
    imageio.mimsave(path, [frames[i] for i in range(frames.shape[0])], fps=fps, macro_block_size=1)


def ffprobe_video(path: str) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "v:0", path]
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        meta = json.loads(out)["streams"][0]
        return {
            "codec": meta.get("codec_name", "h264"),
            "pix_fmt": meta.get("pix_fmt", "yuv420p"),
            "height": int(meta.get("height", 0)),
            "width": int(meta.get("width", 0)),
        }
    except Exception:
        return {"codec": "h264", "pix_fmt": "yuv420p"}


def init_lerobot_dirs(root: str) -> Dict[str, str]:
    # CHANGED: create roots, not only chunk-000
    data_root = os.path.join(root, "data")
    videos_root = os.path.join(root, "videos")
    meta_dir = os.path.join(root, "meta")
    ensure_dir(data_root)
    ensure_dir(videos_root)
    ensure_dir(meta_dir)
    return {"data_root": data_root, "videos_root": videos_root, "meta": meta_dir}


def get_chunk_dirs(ds: Dict[str, Any], chunk_id: int) -> Dict[str, str]:
    """
    Ensure per-chunk dirs exist and return them.
    data/chunk-xxx and videos/chunk-xxx
    """
    # minimal caching
    cache = ds.setdefault("_chunk_cache", {})
    if chunk_id in cache:
        return cache[chunk_id]

    chunk_name = f"chunk-{chunk_id:03d}"
    data_dir = os.path.join(ds["dirs"]["data_root"], chunk_name)
    videos_dir = os.path.join(ds["dirs"]["videos_root"], chunk_name)
    ensure_dir(data_dir)
    ensure_dir(videos_dir)
    cache[chunk_id] = {"data": data_dir, "videos": videos_dir}
    return cache[chunk_id]


# -----------------------------
# config selection & mapping
# -----------------------------
def cfg_required_obs_keys(cfg: Dict[str, Any]) -> List[str]:
    req = []
    for _, spec in cfg.get("images", {}).items():
        if "source_key" in spec:
            req.append(spec["source_key"])
    for k, spec in cfg.get("states", {}).items():
        req.append(spec.get("source_key", k))
    return sorted(set(req))


def choose_config(obs_keys: List[str]) -> Tuple[str, Dict[str, Any]]:
    obs_set = set(obs_keys)
    best_name, best_cfg, best_score = "general", MIMICGEN_MIN_CONFIG, -1
    for name, cfg in CONFIGS:
        required = cfg_required_obs_keys(cfg)
        if not all(k in obs_set for k in required):
            continue
        score = sum((spec.get("source_key", k) in obs_set) for k, spec in cfg.get("states", {}).items())
        score += sum((spec.get("source_key", "") in obs_set) for spec in cfg.get("images", {}).values())
        if score > best_score:
            best_name, best_cfg, best_score = name, cfg, score
    return best_name, best_cfg


def img_sources(cfg: Dict[str, Any]) -> Dict[str, str]:
    imgs = cfg.get("images", {})
    return {
        "front": imgs.get("camera_front", {}).get("source_key", "agentview_image"),
        "wrist": imgs.get("camera_wrist", {}).get("source_key", "robot0_eye_in_hand_image"),
    }


def state_specs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, spec in cfg.get("states", {}).items():
        out_key = f"observation.states.{k}"
        out[out_key] = {
            "source_key": spec.get("source_key", k),
            "dtype": spec.get("dtype", "float32"),
            "postprocess": spec.get("postprocess", None),
        }
    return out


def action_source(cfg: Dict[str, Any]) -> str:
    return cfg.get("actions", {}).get("action", {}).get("source_key", "actions")


# -----------------------------
# info.json finalize
# -----------------------------
def build_info_features(
    cfg: Dict[str, Any],
    fps: float,
    action_dim: int,
    has_wrist: bool,
    cam_hw: Dict[str, Tuple[int, int]],
    codec_info: Dict[str, Dict[str, Any]],
    inferred_state_shapes: Dict[str, int],
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    Hf, Wf = cam_hw["camera_front"]
    cf = codec_info.get("camera_front", {"codec": "h264", "pix_fmt": "yuv420p"})
    features["observation.images.camera_front"] = {
        "dtype": "video",
        "shape": [Hf, Wf, 3],
        "info": {
            "video.height": Hf,
            "video.width": Wf,
            "video.codec": cf["codec"],
            "video.pix_fmt": cf["pix_fmt"],
            "video.is_depth_map": False,
            "video.fps": float(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }

    if has_wrist:
        Hw, Ww = cam_hw["camera_wrist"]
        cw = codec_info.get("camera_wrist", {"codec": "h264", "pix_fmt": "yuv420p"})
        features["observation.images.camera_wrist"] = {
            "dtype": "video",
            "shape": [Hw, Ww, 3],
            "info": {
                "video.height": Hw,
                "video.width": Ww,
                "video.codec": cw["codec"],
                "video.pix_fmt": cw["pix_fmt"],
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    for k, spec in cfg.get("states", {}).items():
        out_key = f"observation.states.{k}"
        dtype = spec.get("dtype", "float32")
        shape = spec.get("shape", None)

        if spec.get("postprocess", None) == "standardize_gripper":
            dtype = "float32"
            shape_list = [1]
        else:
            if shape is None:
                dim = inferred_state_shapes.get(out_key, None)
                shape_list = [int(dim)] if dim is not None else None
            else:
                shape_list = list(shape) if isinstance(shape, (tuple, list)) else shape

        features[out_key] = {"dtype": dtype, "shape": shape_list, "names": spec.get("names", None)}

    features["action"] = {"dtype": "float32", "shape": [int(action_dim)], "names": None}
    features["observation.rewards"] = {"dtype": "float32", "shape": [1], "names": {"motors": ["reward"]}}
    return features


def finalize_dataset(
    root: str,
    dataset_name: str,
    fps: float,
    total_eps: int,
    total_frames: int,
    tasks: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    action_dim: int,
    has_wrist_any: bool,
    cam_hw: Dict[str, Tuple[int, int]],
    codec_info: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    inferred_state_shapes: Dict[str, int],
):
    meta_dir = os.path.join(root, "meta")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")

    info = {
        "codebase_version": "v2.1",
        "dataset_name": dataset_name,
        "fps": float(fps),
        "total_episodes": int(total_eps),
        "total_frames": int(total_frames),
        "total_tasks": int(len(tasks)),
        "total_chunks": 1,  # keep unchanged (your original behavior)
        "chunks_size": 1000,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_info_features(cfg, fps, action_dim, has_wrist_any, cam_hw, codec_info, inferred_state_shapes),
    }

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, type=str)
    ap.add_argument("--output_root", required=True, type=str)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--filter_key", type=str, default=None)
    ap.add_argument("--no_wrist", action="store_true")
    ap.add_argument("--subset_size", type=int, default=0)
    args = ap.parse_args()

    h5_files = iter_hdf5_files(args.input_root)
    print(f"[INFO] Found {len(h5_files)} HDF5 files under {args.input_root}")
    if not h5_files:
        return

    datasets: Dict[str, Dict[str, Any]] = {}

    def get_ds(ds_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if ds_name in datasets:
            return datasets[ds_name]
        out_root = os.path.join(args.output_root, ds_name)
        dirs = init_lerobot_dirs(out_root)

        stats_path = os.path.join(out_root, "meta", "episodes_stats.jsonl")
        stats_f = open(stats_path, "w", encoding="utf-8")  # DEFAULT ON

        ds = {
            "root": out_root,
            "dirs": dirs,
            "fps": float(args.fps),
            "epi": 0,
            "frames": 0,
            "tasks": [],
            "task2idx": {},
            "episodes": [],
            "action_dim": None,
            "has_wrist_any": False,
            "cam_hw": {},
            "codec_info": {},
            "cfg": cfg,
            "inferred_state_shapes": {},
            "stats_f": stats_f,
        }
        datasets[ds_name] = ds
        return ds

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            env_args = read_env_args(f)
            demos = list_demos(f, args.filter_key)
            if args.subset_size and args.subset_size > 0:
                demos = demos[: args.subset_size]
                print(f"[INFO] {os.path.basename(h5_path)}: subset_size={args.subset_size}, demos={len(demos)}")
            if not demos:
                continue

            robot_type = get_robot_type(h5_path, env_args)

            obs0 = f[f"data/{demos[0]}/obs"]
            ds_name, cfg = choose_config(list(obs0.keys()))

            if ds_name == "general":
                rt = robot_type.lower()
                if rt == "panda":
                    ds_name = "general_panda"
                elif rt == "iiwa":
                    ds_name = "general_iiwa"
                elif rt == "ur5e":
                    ds_name = "general_ur5e"
                elif rt == "sawyer":
                    ds_name = "general_sawyer"
                else:
                    ds_name = "general_unknown"

            ds = get_ds(ds_name, cfg)

            task_name = (env_args or {}).get("env_name", "mimicgen_task")
            if task_name not in ds["task2idx"]:
                ds["task2idx"][task_name] = len(ds["tasks"])
                ds["tasks"].append({"task_index": ds["task2idx"][task_name], "task": task_name})
            task_idx = ds["task2idx"][task_name]

            robot = canonical_robot_type_for_gripper(get_robot_type(h5_path, env_args))
            gripper_type = get_gripper_type(env_args)

            imgs = img_sources(cfg)
            states = state_specs(cfg)
            act_src = action_source(cfg)

            for demo in tqdm(demos, desc=f"[{ds_name}] {os.path.basename(h5_path)}", leave=False):
                base = f"data/{demo}"
                obs = f[f"{base}/obs"]

                # images
                front_frames = obs[imgs["front"]][:]
                T = int(front_frames.shape[0])

                wrist_frames = None
                if (not args.no_wrist) and (imgs["wrist"] in obs):
                    wrist_frames = obs[imgs["wrist"]][:]
                    if int(wrist_frames.shape[0]) != T:
                        raise ValueError(f"{h5_path} {demo}: wrist T mismatch")

                cols: Dict[str, pa.Array] = {}
                numeric_states: Dict[str, np.ndarray] = {}

                # states
                for out_key, spec in states.items():
                    src = spec["source_key"]
                    if src not in obs:
                        raise KeyError(f"Missing obs key '{src}' for config '{ds_name}' in {h5_path} {demo}")

                    arr = obs[src][:]
                    if arr.shape[0] != T:
                        raise ValueError(f"{h5_path} {demo}: '{src}' length {arr.shape[0]} != {T}")

                    if arr.ndim == 1:
                        arr = arr.reshape(T, 1)

                    if spec.get("postprocess") == "standardize_gripper":
                        arr = standardize_gripper(
                            arr.astype(np.float32),
                            robot_type=robot,
                            gripper_type=gripper_type,
                        ).astype(np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(T, 1)
                        if arr.shape != (T, 1):
                            raise ValueError(f"{h5_path} {demo}: gripper expected (T,1), got {arr.shape}")
                    else:
                        if spec.get("dtype") == "bool":
                            arr = arr.astype(np.uint8)
                        else:
                            arr = arr.astype(np.float32)

                    if out_key not in ds["inferred_state_shapes"] and arr.ndim == 2:
                        ds["inferred_state_shapes"][out_key] = int(arr.shape[1])

                    cols[out_key] = to_pa_list_array(arr)
                    numeric_states[out_key] = arr

                # action + rewards
                actions = f[f"{base}/{act_src}"][:].astype(np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(T, 1)
                if actions.shape[0] != T:
                    raise ValueError(f"{h5_path} {demo}: actions length {actions.shape[0]} != {T}")

                rewards = f[f"{base}/rewards"][:] if f"{base}/rewards" in f else np.zeros((T,), dtype=np.float32)
                rewards = np.asarray(rewards, dtype=np.float32).reshape(T, 1)

                if ds["action_dim"] is None:
                    ds["action_dim"] = int(actions.shape[1])

                epi = int(ds["epi"])
                ds["epi"] += 1
                ds["frames"] += int(T)

                # CHANGED: compute chunk id and dirs
                episode_chunk = epi // 1000
                chunk_dirs = get_chunk_dirs(ds, episode_chunk)

                # write videos (per chunk)
                front_dir = os.path.join(chunk_dirs["videos"], "observation.images.camera_front")
                ensure_dir(front_dir)
                front_path = os.path.join(front_dir, f"episode_{epi:06d}.mp4")
                write_video(front_frames, front_path, ds["fps"])
                if "camera_front" not in ds["cam_hw"]:
                    ds["cam_hw"]["camera_front"] = (int(front_frames.shape[1]), int(front_frames.shape[2]))
                    ds["codec_info"]["camera_front"] = ffprobe_video(front_path)

                wrist_path = None
                if wrist_frames is not None:
                    ds["has_wrist_any"] = True
                    wrist_dir = os.path.join(chunk_dirs["videos"], "observation.images.camera_wrist")
                    ensure_dir(wrist_dir)
                    wrist_path = os.path.join(wrist_dir, f"episode_{epi:06d}.mp4")
                    write_video(wrist_frames, wrist_path, ds["fps"])
                    if "camera_wrist" not in ds["cam_hw"]:
                        ds["cam_hw"]["camera_wrist"] = (int(wrist_frames.shape[1]), int(wrist_frames.shape[2]))
                        ds["codec_info"]["camera_wrist"] = ffprobe_video(wrist_path)

                # write parquet (per chunk)
                cols.update(
                    {
                        "action": to_pa_list_array(actions),
                        "observation.rewards": to_pa_list_array(rewards),
                        "timestamp": pa.array((np.arange(T) / float(ds["fps"])).astype(np.float32)),
                        "frame_index": pa.array(np.arange(T, dtype=np.int32)),
                        "episode_index": pa.array([epi] * T, type=pa.int32()),
                        "task_index": pa.array([int(task_idx)] * T, type=pa.int32()),
                    }
                )
                pq.write_table(pa.table(cols), os.path.join(chunk_dirs["data"], f"episode_{epi:06d}.parquet"))

                # DEFAULT stats (unchanged)
                compute_and_write_episode_stats(
                    ds["stats_f"],
                    episode_index=epi,
                    cfg=cfg,
                    front_video_path=front_path,
                    wrist_video_path=wrist_path,
                    numeric_states=numeric_states,
                    action=actions,
                    rewards=rewards,
                )

                # meta episode (add episode_chunk, minimal + safe)
                ds["episodes"].append(
                    {
                        "episode_index": epi,
                        "episode_chunk": int(episode_chunk),
                        "tasks": [task_name],
                        "length": int(T),
                    }
                )

    # finalize
    for ds_name, ds in datasets.items():
        if ds.get("stats_f") is not None:
            ds["stats_f"].close()

        if int(ds["epi"]) == 0:
            continue

        if "camera_front" not in ds["cam_hw"]:
            ds["cam_hw"]["camera_front"] = (84, 84)

        finalize_dataset(
            root=ds["root"],
            dataset_name=ds_name,
            fps=ds["fps"],
            total_eps=int(ds["epi"]),
            total_frames=int(ds["frames"]),
            tasks=ds["tasks"],
            episodes=ds["episodes"],
            action_dim=int(ds["action_dim"] or 0),
            has_wrist_any=bool(ds["has_wrist_any"]),
            cam_hw=ds["cam_hw"],
            codec_info=ds["codec_info"],
            cfg=ds["cfg"],
            inferred_state_shapes=ds["inferred_state_shapes"],
        )
        print(f"[OK] {ds_name}: episodes={ds['epi']}, frames={ds['frames']} -> {ds['root']}")

    print("[DONE]")


if __name__ == "__main__":
    main()
