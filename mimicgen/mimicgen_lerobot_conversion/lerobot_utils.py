# import numpy as np
# import torchvision
# from lerobot.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
# from lerobot.datasets.utils import load_image_as_numpy

# torchvision.set_video_backend("pyav")


# def generate_features_from_config(AgiBotWorld_CONFIG):
#     features = {}
#     for key, value in AgiBotWorld_CONFIG["images"].items():
#         features[f"observation.images.{key}"] = value
#     for key, value in AgiBotWorld_CONFIG["states"].items():
#         features[f"observation.states.{key}"] = value
#     for key, value in AgiBotWorld_CONFIG["actions"].items():
#         features[f"actions.{key}"] = value
#     return features


# def sample_images(input):
#     if type(input) is list:
#         image_paths = input

#         sampled_indices = sample_indices(len(image_paths))
#         images = None
#         for i, idx in enumerate(sampled_indices):
#             path = image_paths[idx]

#             img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
#             img = auto_downsample_height_width(img)

#             if images is None:
#                 images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

#             images[i] = img
#     elif type(input) is np.ndarray:
#         frames_array = input[:, None, :, :]  # Shape: [T, 1, H, W]
#         sampled_indices = sample_indices(len(frames_array))
#         images = None
#         for i, idx in enumerate(sampled_indices):
#             img = frames_array[idx]
#             img = auto_downsample_height_width(img)

#             if images is None:
#                 images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

#             images[i] = img

#     return images


# def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
#     ep_stats = {}
#     for key, data in episode_data.items():
#         if features[key]["dtype"] == "string":
#             continue  # HACK: we should receive np.arrays of strings
#         elif features[key]["dtype"] in ["image", "video"]:
#             ep_ft_array = sample_images(data)
#             axes_to_reduce = (0, 2, 3)  # keep channel dim
#             keepdims = True
#         else:
#             ep_ft_array = data  # data is already a np.ndarray
#             axes_to_reduce = 0  # compute stats over the first axis
#             keepdims = data.ndim == 1  # keep as np.array

#         ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

#         if features[key]["dtype"] in ["image", "video"]:
#             value_norm = 1.0 if "depth" in key else 255.0
#             ep_stats[key] = {
#                 k: v if k == "count" else np.squeeze(v / value_norm, axis=0) for k, v in ep_stats[key].items()
#             }

#     return ep_stats


# lerobot_utils.py
import json
from typing import Dict, Any, Optional

import numpy as np
import torch
import torchvision

from lerobot.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.datasets.utils import load_image_as_numpy

torchvision.set_video_backend("pyav")


def generate_features_from_config(cfg: Dict[str, Any], has_wrist: bool) -> Dict[str, Dict[str, Any]]:
    """
    Build a minimal features dict for stats computation.
    Keys MUST match episode_data keys.
    Only 'dtype' is required.

    Expected episode_data keys:
      - observation.images.camera_front (video)
      - observation.images.camera_wrist (video) [optional]
      - observation.states.*            (numeric)
      - action                          (numeric)
      - observation.rewards             (numeric)
    """
    feats: Dict[str, Dict[str, Any]] = {
        "observation.images.camera_front": {"dtype": "video"},
        "action": {"dtype": "float32"},
        "observation.rewards": {"dtype": "float32"},
    }
    if has_wrist:
        feats["observation.images.camera_wrist"] = {"dtype": "video"}

    for k, spec in cfg.get("states", {}).items():
        out_key = f"observation.states.{k}"
        dtype = spec.get("dtype", "float32")
        if spec.get("postprocess") == "standardize_gripper":
            dtype = "float32"
        if dtype == "bool":
            dtype = "float32"
        feats[out_key] = {"dtype": dtype}

    return feats


def sample_images(inp):
    """
    LeRobot official sampling + downsample behavior.

    Supports:
      1) str: mp4 path -> VideoReader -> [T,C,H,W] uint8
      2) list[str]: image paths -> load_image_as_numpy -> [S,C,H,W]
      3) np.ndarray:
           - (T,H,W) grayscale
           - (T,H,W,3) RGB
           - (T,C,H,W) channel-first
    Returns:
      np.ndarray uint8 [S,C,H,W]
    """
    # mp4 path
    if isinstance(inp, str):
        reader = torchvision.io.VideoReader(inp, stream="video")
        frames = [fr["data"] for fr in reader]  # list of [C,H,W] uint8
        if len(frames) == 0:
            return np.empty((0, 3, 1, 1), dtype=np.uint8)

        frames_array = torch.stack(frames).numpy()  # [T,C,H,W]
        sampled = sample_indices(len(frames_array))

        images = None
        for i, idx in enumerate(sampled):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled), *img.shape), dtype=np.uint8)
            images[i] = img
        return images

    # image paths list
    if type(inp) is list:
        image_paths = inp
        sampled = sample_indices(len(image_paths))

        images = None
        for i, idx in enumerate(sampled):
            path = image_paths[idx]
            img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)  # [C,H,W]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled), *img.shape), dtype=np.uint8)
            images[i] = img
        return images

    # ndarray
    if isinstance(inp, np.ndarray):
        arr = inp
        # (T,H,W,3) -> (T,3,H,W)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (0, 3, 1, 2))
        # (T,H,W) -> (T,1,H,W)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported ndarray shape for images: {inp.shape}")

        sampled = sample_indices(len(arr))
        images = None
        for i, idx in enumerate(sampled):
            img = arr[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled), *img.shape), dtype=np.uint8)
            images[i] = img.astype(np.uint8)
        return images

    raise TypeError(f"Unsupported input type for sample_images: {type(inp)}")


def compute_episode_stats(episode_data: Dict[str, Any], features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute per-episode stats for each feature:
      - numeric: stats over time axis 0
      - image/video: sample+downsample then stats over (S,H,W), keep channel dim

    Returns a dict:
      { key: {min,max,mean,std,count}, ... }
    """
    ep_stats: Dict[str, Any] = {}

    for key, data in episode_data.items():
        if key not in features:
            continue

        dtype = features[key]["dtype"]
        if dtype == "string":
            continue

        if dtype in ["image", "video"]:
            ep_ft_array = sample_images(data)        # [S,C,H,W]
            axes_to_reduce = (0, 2, 3)               # reduce S,H,W keep C
            keepdims = True
        else:
            ep_ft_array = data                       # np.ndarray
            axes_to_reduce = 0                       # reduce time
            keepdims = (getattr(data, "ndim", 0) == 1)

        stats = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if dtype in ["image", "video"]:
            value_norm = 1.0 if "depth" in key else 255.0
            stats = {k: v if k == "count" else np.squeeze(v / value_norm, axis=0) for k, v in stats.items()}

        ep_stats[key] = stats

    return ep_stats


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def write_episode_stats_jsonl(fh, episode_index: int, ep_stats: Dict[str, Any]):
    """
    Append one json line:
      {"episode_index": i, "stats": {...}}
    """
    fh.write(json.dumps({"episode_index": int(episode_index), "stats": _jsonable(ep_stats)}) + "\n")


def compute_and_write_episode_stats(
    fh,
    *,
    episode_index: int,
    cfg: Dict[str, Any],
    front_video_path: str,
    wrist_video_path: Optional[str],
    numeric_states: Dict[str, np.ndarray],
    action: np.ndarray,
    rewards: np.ndarray,
):
    """
    One-call convenience API for your converter.

    numeric_states keys should be exactly:
      observation.states.xxx
    """
    has_wrist = wrist_video_path is not None
    feats = generate_features_from_config(cfg, has_wrist=has_wrist)

    ep_data: Dict[str, Any] = {}
    ep_data["observation.images.camera_front"] = front_video_path
    if has_wrist:
        ep_data["observation.images.camera_wrist"] = wrist_video_path

    ep_data.update(numeric_states)
    ep_data["action"] = action
    ep_data["observation.rewards"] = rewards

    ep_stats = compute_episode_stats(ep_data, feats)
    write_episode_stats_jsonl(fh, episode_index=episode_index, ep_stats=ep_stats)
