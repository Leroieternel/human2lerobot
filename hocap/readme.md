# HO-Cap to LeRobot Conversion Toolkit

This repository provides a utility to convert **HO-Cap datasets**
into a **LeRobot-compatible dataset format (V2.1)**, including:

- Per-episode **Parquet** files (low-dimensional states, actions, rewards)
- Per-episode **MP4 videos** (RGB observations)
- Standard **LeRobot metadata** (`episodes.jsonl`, `tasks.jsonl`, etc.)

The conversion logic is implemented in `hocap_lerobot_conversion/hocap2lerobot.py`. It is divided into 3 subsets when converting, according to thir task and robot types.

---

## 1. Installation
Create Conda environment:
```
conda create -n hocap_lerobot python=3.10 -y
conda activate hocap_lerobot
```

Required Python packages:

```
sudo apt-get install -y ffmpeg
pip install -U h5py matplotlib numpy pyarrow imageio imageio-ffmpeg tqdm pyyaml opencv ffmpeg
```
Install Lerobot:
```
pip install lerobot
```
\

## 2. Repository Structure

```
hocap/
├── hocap_lerobot_conversion/
│   ├── extract_delta_eef.py            # extract delta end effector pose (in world frame) from .npz files
│   ├── hocap_config.py                 # config file to load the value by keys and generate info.json
│   ├── hocap2lerobot.py                # main script to convert hocap to lerobot
│   ├── lerobot_utils.py                # some utility functions from lerobot package
├── read_hocap_label/
│   ├── check_projection_2d.py          # check the projection from hand_joints_3d points to hand_joints_2d
│   ├── check_projection_world.py       # check if the key cam_K is the transform matrix from camera to world
│   ├── read_hocap_label.py             # read .npz label (check the keys and their values)
│   └── read_image_shape.py             # check image shape of each camera
└── readme.md
```

---


## 3. Running the Conversion Script

The converter assumes a standard MimicGen / robomimic HDF5 structure:

- **T** is the episode length (number of time steps).
- The episode length is inferred from `agentview_image`.
- All other arrays must have the same first dimension `T`.

---

To convert the whole Hocap dataset to Lerobot format, please run the following command:

```
python hocap_lerobot_conversion/hocap2lerobot.py \
  --input_root /path/to/hocap_dataset \
  --output_root /path/to/output_lerobot_dataset \
  --fps {fps}
```

For sanity check, we introduce an argument `subset_size`, `subjects`, `num_views`. When this argument is specified, the script samples `subset_size` episodes from each HDF5 file and uses them to generate the LeRobot dataset. An example usage:

```
python mimicgen_lerobot_conversion/mimicgen2lerobot.py \
  --input_root /path/to/mimicgen_dataset \
  --output_root /path/to/output_lerobot_dataset \
  --fps 20
  --subset_size 2
```

## 4. Read the HDF5 Dataset

To read the npz label of HO-Cap dataset:
```
python read_hocap_label/read_hocap_label.py
```
This prints the information of one `.npz` file, including the hand 3d joints coordinates in camera frame (key `hand_joints_3d`).



It will generate three `.json` files in the `read_hdf5_data` folder. `mimicgen_hdf5_keys_comparison.json` lists the most common keys appeared in all the hdf5 files, as well as the extra/missing keys of each hdf5 file compared to these common keys. `mimicgen_hdf5_summary.json` prints all the key names and their shapes of each hdf5 file. `mimicgen_robot_gripper_type.json` demonstrates the robot and gripper types of each hdf5 file.


## 5. HDF5 Keys and Their Meanings

All fields are defined in `mimicgen_lerobot_conversion/mimicgen_config_{subset_name}.py`.
Below is a detailed explanation of each HDF5 key.

### 5.1 Image Observations

#### `data/<demo>/obs/agentview_image`
- Shape: `(T, 84, 84, 3)`
- Type: `uint8`
- Description: Main RGB camera view (agent view).
- Always exported as an MP4 video.

#### `data/<demo>/obs/robot0_eye_in_hand_image` 
- Shape: `(T, 84, 84, 3)`
- Type: `uint8`
- Description: Wrist / eye-in-hand RGB camera.
- Exported unless `--no_wrist` is specified.

---

### 5.2 Low-Dimensional State Observations

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

### 5.3 Actions and Rewards

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

## 6. Output: LeRobot-Compatible Dataset

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


## 7. Notes

- Episode length T is defined by agentview_image.
- All arrays must match this length.
- Quaternions are assumed to be xyzw (robosuite convention).
- Always verify actual HDF5 shapes before conversion.