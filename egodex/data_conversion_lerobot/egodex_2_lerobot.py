import os
import json
import h5py
import numpy as np
import imageio
import subprocess
from tqdm import tqdm
from pyarrow import parquet as pq
import pyarrow as pa
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from simple_dataset import SimpleDataset
from extract_delta_eef import extract_delta_eef_from_single_episode, compute_delta

OUTPUT_ROOT = "/raid/xiangyi/egodex_lerobot/scratch"   # change me
meta_dir = os.path.join(OUTPUT_ROOT, "meta")
os.makedirs(meta_dir, exist_ok=True)

# get video info
def ffprobe_video(video_path):
    """Return codec, pix_fmt, width, height, fps, channels for 1 video."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    meta = json.loads(out)

    v = meta["streams"][0]

    codec = v.get("codec_name")
    pix_fmt = v.get("pix_fmt")
    width = v.get("width")
    height = v.get("height")
    channels = v.get("channels", 3)  # usually 3
    # fps is in "r_frame_rate" like "30/1"
    fps_str = v.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    return dict(
        codec=codec,
        pix_fmt=pix_fmt,
        width=width,
        height=height,
        fps=fps,
        channels=channels
    )


def get_video_info_from_folder(folder_path):
    """
    Scan all .mp4 files under the folder and verify they share same video properties.
    Returns:
        video_info dict with keys:
        ["video.height", "video.width", "video.fps",
         "video.codec", "video.pix_fmt", "video.channels"]
    """
    mp4_files = sorted([os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if f.endswith(".mp4")])

    if len(mp4_files) == 0:
        raise ValueError(f"No mp4 files found in {folder_path}")

    first_info = None

    for idx, mp4 in enumerate(mp4_files):
        info = ffprobe_video(mp4)

        if idx == 0:
            first_info = info
            continue

        # compare all keys for consistency
        for key in first_info:
            if info[key] != first_info[key]:
                raise ValueError(
                    f"Inconsistent video metadata in {mp4}:\n"
                    f"  {key} mismatch: {info[key]} != {first_info[key]}"
                )

    # rename keys to LeRobot format
    result = {
        "height": first_info["height"],
        "width": first_info["width"],
        "fps": first_info["fps"],
        "codec": first_info["codec"],
        "pix_fmt": first_info["pix_fmt"],
        "channels": first_info["channels"],
        "has_audio": False,   # Egodex videos have no audio
    }

    return result


# write parquet dataset
def write_parquet_episode(episode_index, rgb_frames, eef_abs, delta_eef, joints_xyz, delta_joints_xyz, task_index, fps=30):
    """
    rgb_frames: list of HxWx3 uint8
    eef_abs:   (N, 12)  state: absolute eef
    delta_eef: (N, 12)  action: delta eef
    joints_xyz:(N, 12, 3)  action: 12 joints xyz
    delta_joints_xyz:(N, 12, 3)  action: 12 delta joints xyz
    """
    N = len(rgb_frames)

    # Write video
    video_dir = f"{OUTPUT_ROOT}/videos/chunk-000/camera_front"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/episode_{episode_index:06d}.mp4"
    imageio.mimsave(video_path, rgb_frames, fps=fps, macro_block_size=1)
    
    # joints_xyz to 36d to save in parquet
    joints_flat = joints_xyz.reshape(N, -1)  # (N, 36)
    delta_joints_flat = delta_joints_xyz.reshape(N, -1)  # (N, 36)
    
    a = pa.array(joints_flat.tolist(), type=pa.list_(pa.float32(), 36))
    b = pa.array(delta_joints_flat.tolist(), type=pa.list_(pa.float32(), 36))
    print("observation.states.joint_position shape ", a.type)
    print("actions.joint_position shape: ", b.type)

    # Write parquet
    table = pa.table({
        "observation.states.end_effector": pa.array(eef_abs.tolist(), type=pa.list_(pa.float32(), 12)),    # double hands absolute eef
        "observation.states.joint_position": pa.array(joints_flat.tolist(), type=pa.list_(pa.float32(), 36)),      # state: 12 joints xyz
        "actions.delta_end_effector": pa.array(delta_eef.tolist(), type=pa.list_(pa.float32(),   12)),             # action 1: double hands delta eef
        "actions.joint_position": pa.array(delta_joints_flat.tolist(), type=pa.list_(pa.float32(), 36)),     # action 2: 12 joints delta xyz
        "timestamp": pa.array((np.arange(N)/fps).astype(np.float32)),         
        "frame_index": pa.array(np.arange(N)),
        "episode_index": pa.array([episode_index]*N),
        "task_index": pa.array([task_index]*N),
    })

    data_dir = f"{OUTPUT_ROOT}/data/chunk-000"
    os.makedirs(data_dir, exist_ok=True)
    pq.write_table(table, f"{data_dir}/episode_{episode_index:06d}.parquet")


# write info json
def generate_info_json(example_hdf5, total_episodes, total_frames, total_tasks, video_info):
    with h5py.File(example_hdf5, "r") as f:
        # joint_names = list(f["transforms"].keys())
        joint_names = [j for j in f["transforms"].keys() if j != "camera"]

    info = {
        "codebase_version": "v2.1",
        "robot_type": "egodex_human_lerobot",

        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": video_info["fps"],

        "splits": {"train": f"0:{total_episodes}"},

        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/camera_front/episode_{episode_index:06d}.mp4",

        "features": {
            "observation.images.camera_front": {
                "dtype": "video",
                "shape": [video_info["height"], video_info["width"], 3],
                "info": {
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.codec": video_info["codec"],
                    "video.pix_fmt": video_info["pix_fmt"],
                    "video.is_depth_map": False,
                    "video.fps": video_info["fps"],
                    "video.channels": 3,
                    "has_audio": False
                }
            },

            "observation.states.end_effector": {
                "dtype": "float32",
                "shape": [12],
                "names": {
                    "motors": [
                        "right_x", "right_y", "right_z",
                        "right_roll", "right_pitch", "right_yaw",
                        "left_x", "left_y", "left_z",
                        "left_roll", "left_pitch", "left_yaw",
                    ]
                }
            },
            
            "observation.states.joint_position": {
                "dtype": "float32",
                "shape": [12, 3],
                "names": {
                    "motors": [
                        "right_wrist",
                        "right_thumb_tip",
                        "right_index_tip",
                        "right_middle_tip",
                        "right_ring_tip",
                        "right_little_tip",
                        "left_wrist",
                        "left_thumb_tip",
                        "left_index_tip",
                        "left_middle_tip",
                        "left_ring_tip",
                        "left_little_tip",
                    ]
                }
            },

            "actions.delta_end_effector": {
                "dtype": "float32",
                "shape": [12],
                "names": {
                    "motors": [
                        "d_right_x", "d_right_y", "d_right_z",
                        "d_right_roll", "d_right_pitch", "d_right_yaw",
                        "d_left_x", "d_left_y", "d_left_z",
                        "d_left_roll", "d_left_pitch", "d_left_yaw",
                    ]
                }
            },

            "actions.joint_position": {
                "dtype": "float32",
                "shape": [12, 3],
                "names": {
                    "motors": [
                        "right_wrist",
                        "right_thumb_tip",
                        "right_index_tip",
                        "right_middle_tip",
                        "right_ring_tip",
                        "right_little_tip",
                        "left_wrist",
                        "left_thumb_tip",
                        "left_index_tip",
                        "left_middle_tip",
                        "left_ring_tip",
                        "left_little_tip",
                    ]
                }
            },
        }
    }

    save_path = os.path.join(meta_dir, "info.json")
    with open(save_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"[OK] info.json saved to {save_path}")


# compute stats
def compute_stats_array_full(x):
    """Return dict(min,max,mean,std,count) with the same layout as episodes_stats.jsonl"""
    return {
        "min": np.min(x, axis=0).reshape(-1).tolist(),
        "max": np.max(x, axis=0).reshape(-1).tolist(),
        "mean": np.mean(x, axis=0).reshape(-1).tolist(),
        "std": np.std(x, axis=0).reshape(-1).tolist(),
        "count": [int(x.shape[0])]
    }


def generate_episodes_stats_jsonl():
    print("\n[Stats] Generating episodes_stats.jsonl ...")

    parquet_dir = f"{OUTPUT_ROOT}/data/chunk-000"
    video_dir = f"{OUTPUT_ROOT}/videos/chunk-000/camera_front"
    save_path = os.path.join(meta_dir, "episodes_stats.jsonl")

    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

    assert len(parquet_files) == len(video_files), "Episode count mismatch!"

    with open(save_path, "w") as out_f:

        for epi_idx, fname in enumerate(tqdm(parquet_files)):

            # ============================
            #   Load parquet per episode
            # ============================
            table = pq.read_table(os.path.join(parquet_dir, fname))
            df = table.to_pandas()

            stats = {}

            # ------- numeric stats -------
            eef = np.vstack(df["observation.states.end_effector"])
            delta = np.vstack(df["actions.delta_end_effector"])
            joints_s = np.vstack(df["observation.states.joint_position"])
            joints_a = np.vstack(df["actions.joint_position"])

            stats["observation.states.end_effector"] = compute_stats_array_full(eef)
            stats["observation.states.joint_position"] = compute_stats_array_full(joints_s)
            stats["actions.delta_end_effector"] = compute_stats_array_full(delta)
            stats["actions.joint_position"] = compute_stats_array_full(joints_a)

            # ------- index / episode meta -------
            N = len(df)
            stats["episode_index"] = compute_stats_array_full(
                np.array([epi_idx] * N).reshape(N, 1)
            )
            stats["index"] = compute_stats_array_full(
                np.arange(N).reshape(N, 1)
            )
            stats["task_index"] = compute_stats_array_full(
                np.zeros((N, 1))
            )

            # ============================
            #   RGB statistics (per episode)
            # ============================
            vid_path = os.path.join(video_dir, video_files[epi_idx])
            reader = imageio.get_reader(vid_path)
            frames = []

            SAMPLE = 10
            for idx, frame in enumerate(reader):
                if idx % SAMPLE == 0:
                    frames.append(frame.astype(np.float32) / 255.0)
            reader.close()

            frames = np.stack(frames, axis=0)

            rgb_min = frames.min(axis=(0, 1, 2))
            rgb_max = frames.max(axis=(0, 1, 2))
            rgb_mean = frames.mean(axis=(0, 1, 2))
            rgb_std = frames.std(axis=(0, 1, 2))

            stats["observation.images.camera_front"] = {
                "min": rgb_min.reshape(3, 1, 1).tolist(),
                "max": rgb_max.reshape(3, 1, 1).tolist(),
                "mean": rgb_mean.reshape(3, 1, 1).tolist(),
                "std": rgb_std.reshape(3, 1, 1).tolist(),
                "count": [frames.shape[0]],
            }

            # ============================
            #   Write one episode line
            # ============================
            out_obj = {
                "episode_index": epi_idx,
                "stats": stats
            }
            out_f.write(json.dumps(out_obj) + "\n")

    print(f"[OK] episodes_stats.jsonl saved to {save_path}\n")


def main_convert(root_dir):

    print("Scanning video meta...")
    video_info = get_video_info_from_folder(root_dir)
    print("Video info:", video_info)

    all_h5 = sorted([f for f in os.listdir(root_dir) if f.endswith(".hdf5")])

    episodes_jsonl = []
    tasks_jsonl = []
    task_to_index = {}
    next_task_index = 0

    total_frames = 0

    for epi, h5_name in enumerate(tqdm(all_h5)):
        h5_path = os.path.join(root_dir, h5_name)
        mp4_path = h5_path.replace(".hdf5", ".mp4")

        # Load eef + delta
        eef_abs, delta_eef, joints_xyz, delta_joints_xyz = extract_delta_eef_from_single_episode(h5_path)


        # Instruction
        with h5py.File(h5_path, "r") as f:
            raw = f.attrs["llm_description"]
            instr = raw.decode("utf-8") if isinstance(raw, bytes) else raw

        if instr not in task_to_index:
            task_to_index[instr] = next_task_index
            tasks_jsonl.append({"task_index": next_task_index, "instruction": instr})
            next_task_index += 1

        task_index = task_to_index[instr]

        # Load RGB frames
        video = imageio.get_reader(mp4_path)
        rgb_frames = [frame for frame in video]
        video.close()

        N = len(rgb_frames)
        total_frames += N

        # Write parquet + video
        write_parquet_episode(epi, rgb_frames, eef_abs=eef_abs, delta_eef=delta_eef, joints_xyz=joints_xyz, delta_joints_xyz=delta_joints_xyz, task_index=task_index, fps=video_info["fps"])

        episodes_jsonl.append({
            "episode_index": epi,
            "tasks": [instr],
            "length": N
        })

    # --- Save meta ---
    with open(f"{meta_dir}/episodes.jsonl", "w") as f:
        for e in episodes_jsonl:
            f.write(json.dumps(e) + "\n")

    with open(f"{meta_dir}/tasks.jsonl", "w") as f:
        for t in tasks_jsonl:
            f.write(json.dumps(t) + "\n")

    # --- Save info.json ---
    example_hdf5 = os.path.join(root_dir, all_h5[0])

    generate_info_json(
        example_hdf5,
        total_episodes=len(all_h5),
        total_frames=total_frames,
        total_tasks=len(tasks_jsonl),
        video_info=video_info
    )
    
    generate_episodes_stats_jsonl()
    
    print("DONE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    main_convert(args.data_dir)
