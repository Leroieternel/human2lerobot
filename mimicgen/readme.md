# MimicGen to LeRobot Conversion Toolkit

This repository provides a utility to convert **MimicGen / robomimic-style HDF5 datasets**
into a **LeRobot-compatible dataset format**, including:

- Per-episode **Parquet** files (low-dimensional states, actions, rewards)
- Per-episode **MP4 videos** (RGB observations)
- Standard **LeRobot metadata** (`episodes.jsonl`, `tasks.jsonl`, etc.)

The conversion logic is implemented in `mimicgen_2_lerobot.py`, with feature definitions
specified in `config.py`.

---

## 1. Repository Structure

```
mimicgen
├── config.py # Definition of all features to read from HDF5
├── mimicgen_2_lerobot.py # Main conversion script
├── read_mimicgen_data.py # Utility script to inspect MimicGen HDF5 files
└── readme.md # This file
```


---

## 2. Input: MimicGen / Robomimic HDF5 Format

The converter assumes a standard MimicGen / robomimic HDF5 structure:

- **T** is the episode length (number of time steps).
- The episode length is inferred from `agentview_image`.
- All other arrays must have the same first dimension `T`.

---

## 3. HDF5 Keys and Their Meanings

All fields are defined in `config.py` under `MIMICGEN_FEATURES`.
Below is a detailed explanation of each HDF5 key.

### 3.1 Image Observations

#### `data/<demo>/obs/agentview_image`
- Shape: `(T, 84, 84, 3)`
- Type: `uint8`
- Description: Main RGB camera view (agent view).
- Always exported as an MP4 video.

#### `data/<demo>/obs/robot0_eye_in_hand_image` (optional)
- Shape: `(T, 84, 84, 3)`
- Type: `uint8`
- Description: Wrist / eye-in-hand RGB camera.
- Exported unless `--no_wrist` is specified.

---

### 3.2 Low-Dimensional State Observations

All of the following are stored under `data/<demo>/obs/` and exported as
columns in the episode Parquet file.

#### End-Effector States
- `robot0_eef_pos` `(T, 3)`  
  End-effector position `(x, y, z)` in world frame.

- `robot0_eef_pos_rel_pod` `(T, 3)`  
  End-effector position relative to the pod reference frame.

- `robot0_eef_pos_rel_pod_holder` `(T, 3)`  
  End-effector position relative to the pod holder.

- `robot0_eef_quat` `(T, 4)`  
  End-effector orientation quaternion `(x, y, z, w)`.

- `robot0_eef_quat_rel_pod` `(T, 4)`  
  End-effector orientation relative to the pod frame.

- `robot0_eef_quat_rel_pod_holder` `(T, 4)`  
  End-effector orientation relative to the pod holder.

#### Velocities
- `robot0_eef_vel_lin` `(T, 3)`  
  End-effector linear velocity.

- `robot0_eef_vel_ang` `(T, 3)`  
  End-effector angular velocity.

#### Gripper
- `robot0_gripper_qpos` `(T, 2)`  
  Gripper finger joint positions.

- `robot0_gripper_qvel` `(T, 2)`  
  Gripper finger joint velocities.

#### Robot Joints
- `robot0_joint_pos` `(T, 7)`  
  Robot arm joint positions.

- `robot0_joint_pos_cos` `(T, 7)`  
  Cosine of joint angles.

- `robot0_joint_pos_sin` `(T, 7)`  
  Sine of joint angles.

- `robot0_joint_vel` `(T, 7)`  
  Robot arm joint velocities.

---

### 3.3 Actions and Rewards

#### `data/<demo>/actions`
- Shape: `(T, 7)`
- Description: [dx, dy, dz, dax, day, daz, gripper]

End-effector delta translation, delta rotation (axis-angle),
and gripper command.

#### `data/<demo>/rewards`
- Shape: `(T,)` or `(T,1)`
- Description: Scalar reward at each timestep.
- Exported as `(T, 1)` in Parquet.

---

## 4. Output: LeRobot-Compatible Dataset

The converter produces the following directory structure:

```
OUTPUT_ROOT/
├── data/
│ └── chunk-000/
│ ├── episode_000000.parquet
│ ├── episode_000001.parquet
│ └── ...
├── videos/
│ └── chunk-000/
│ ├── observation.images.agentview_image/
│ │ ├── episode_000000.mp4
│ │ └── ...
│ └── observation.images.robot0_eye_in_hand_image/
│ └── ...
└── meta/
├── info.json
├── episodes.jsonl
├── tasks.jsonl
└── episodes_stats.jsonl
```


### Parquet Columns

Each episode Parquet file contains:

- All observation state fields defined in `config.py`
- `action`
- `reward`
- `timestamp` (seconds)
- `frame_index`
- `episode_index`
- `task_index`

Each row corresponds to one timestep.

---

## 5. Installation

Required Python packages:

```
pip install -U h5py numpy imageio pyarrow tqdm matplotlib
```

Optional (recommend): 
```
sudo apt-get install -y ffmpeg
```

## 6. Running the Converter

Basic

```
python mimicgen_2_lerobot.py \
  --input_hdf5 /path/to/mimicgen_dataset.hdf5 \
  --output_root /path/to/output_lerobot_dataset \
  --fps 20
```

## 7. Inspecting the HDF5 Dataset

```
python read_mimicgen_data.py
```
This prints demo lists, observation shapes, actions, rewards, and environment metadata.

## 8. Notes

- Episode length T is defined by agentview_image.
- All arrays must match this length.
- Quaternions are assumed to be xyzw (robosuite convention).
- Always verify actual HDF5 shapes before conversion.