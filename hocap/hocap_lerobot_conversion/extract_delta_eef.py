import numpy as np
import yaml
import os
import glob
import re

SENTINEL = -0.5  # invalid joint threshold
EXTR_YAML = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/calibration/extrinsics/extrinsics_20231014.yaml"

LABEL_RE = re.compile(r"(?:label_)?(\d+)\.npz$", re.IGNORECASE)


def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def load_extrinsics_yaml(yaml_path):
    """
    Convert extrinsics from HO-Cap yaml to (4,4) camera-to-world transforms.
    Returns:
        extrinsics: dict[str, np.ndarray], each is (4,4) T_world_cam
        rs_master: str
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    extr_raw = data["extrinsics"]
    rs_master = data.get("rs_master", None)  # reference camera number of the world frame

    extrinsics_dict = {}
    for camera_name, extr_vec in extr_raw.items():
        extr_vec = np.asarray(extr_vec, dtype=np.float32).reshape(3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = extr_vec
        extrinsics_dict[str(camera_name)] = T

    return extrinsics_dict, rs_master


def compute_world_coord_from_camera(
    npz_path,
    extrinsics_yaml_path,
    valid_hand_sides=None,
):
    """
    Convert hand_joints_3d from camera frame to world frame for ONE npz.
    Returns:
        world_joints_all: (num_valid_hands, 21, 3)  (in world frame)
        meta: dict
    """
    camera_name = os.path.basename(os.path.dirname(npz_path))

    extrinsics, rs_master = load_extrinsics_yaml(extrinsics_yaml_path)
    if camera_name not in extrinsics:
        raise KeyError(f"Camera name {camera_name} not found in extrinsics yaml")
    T_world_cam = extrinsics[camera_name]

    d = np.load(npz_path, allow_pickle=True)
    joints_cam_all = d["hand_joints_3d"].astype(np.float32)  # (2,21,3)

    # select valid slots
    if valid_hand_sides is None:
        valid_slots = []
        for s in range(joints_cam_all.shape[0]):
            wrist = joints_cam_all[s, 0]
            if wrist[0] > SENTINEL:
                valid_slots.append(s)
        if len(valid_slots) == 0:
            valid_slots = []  # no hands in this frame
    else:
        valid_slots = [valid_hand_sides]

    world_joints_all = []
    valid_masks_all = []

    for s in valid_slots:
        joints_cam = joints_cam_all[s]  # (21,3)
        valid_mask = joints_cam[:, 0] > SENTINEL
        world_joints = np.full_like(joints_cam, -1.0, dtype=np.float32)

        if valid_mask.any():
            ones = np.ones((valid_mask.sum(), 1), dtype=np.float32)
            pts_h = np.hstack([joints_cam[valid_mask], ones])   # (N,4)
            pts_w = (T_world_cam @ pts_h.T).T[:, :3]            # (N,3)
            world_joints[valid_mask] = pts_w

        world_joints_all.append(world_joints)
        valid_masks_all.append(valid_mask)

    if len(world_joints_all) == 0:
        world_joints_all = np.zeros((0, 21, 3), dtype=np.float32)
    else:
        world_joints_all = np.stack(world_joints_all, axis=0)

    meta = {
        "camera_name": camera_name,
        "valid_hand_sides": valid_slots,
        "T_world_cam": T_world_cam,
        "rs_master": rs_master,
        "valid_masks": valid_masks_all,
    }
    return world_joints_all, meta


def _rotmat_to_rpy_xyz(R):
    x = float(-R[2, 0])
    x = max(-1.0, min(1.0, x))
    pitch = np.arcsin(x)

    if abs(np.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return float(roll), float(pitch), float(yaw)


def compute_delta_eef_from_world_joints(
    world_joints_all_steps,  # (T,2,21,3)
    wrist_idx=0,
    thumb_knuckle_idx=2,
    index_knuckle_idx=5,
):
    """
    Input:
      world_joints_all_steps: (T,2,21,3), slot0=right, slot1=left (HO-Cap约定)
    Output:
      dict: left_eef/right_eef, shape (T-1,6) where 6=[dx,dy,dz,droll,dpitch,dyaw]
    """
    J = np.asarray(world_joints_all_steps, dtype=np.float32)
    assert J.ndim == 4 and J.shape[1:] == (2, 21, 3), f"expect (T,2,21,3), got {J.shape}"
    T = J.shape[0]

    def _pose6_for_slot(slot):
        pose6 = np.full((T, 6), np.nan, dtype=np.float32)
        for t in range(T):
            jt = J[t, slot]
            if np.all(jt[wrist_idx] == -1.0) or np.all(jt[thumb_knuckle_idx] == -1.0) or np.all(jt[index_knuckle_idx] == -1.0):
                continue

            wrist = jt[wrist_idx]
            thumb = jt[thumb_knuckle_idx]
            index = jt[index_knuckle_idx]

            x_axis = _normalize(index - wrist)
            y_temp = _normalize(thumb - wrist)
            if x_axis is None or y_temp is None:
                continue

            z_axis = _normalize(np.cross(x_axis, y_temp))
            if z_axis is None:
                continue

            y_axis = _normalize(np.cross(z_axis, x_axis))
            if y_axis is None:
                continue

            R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
            roll, pitch, yaw = _rotmat_to_rpy_xyz(R)

            pose6[t, 0:3] = wrist
            pose6[t, 3:6] = np.array([roll, pitch, yaw], dtype=np.float32)
        return pose6

    def _delta_from_pose6(pose6):
        delta = np.full((T - 1, 6), np.nan, dtype=np.float32)
        delta[:, 0:3] = pose6[1:, 0:3] - pose6[:-1, 0:3]
        for t in range(1, T):
            if np.any(np.isnan(pose6[t, :])) or np.any(np.isnan(pose6[t - 1, :])):
                continue
            delta[t - 1, 3:6] = pose6[t, 3:6] - pose6[t - 1, 3:6]
        return delta

    left_pose6 = _pose6_for_slot(slot=1)
    right_pose6 = _pose6_for_slot(slot=0)

    return {
        "left_eef": _delta_from_pose6(left_pose6),
        "right_eef": _delta_from_pose6(right_pose6),
    }


def _step_id_from_path(p):
    name = os.path.basename(p)
    m = LABEL_RE.search(name)
    return int(m.group(1)) if m else None


def load_episode_world_joints_from_camera_folder(camera_folder, extrinsics_yaml_path):
    """
    Read ALL label_*.npz under one camera folder and build (T,2,21,3) world joints array.
    Missing hands in a frame will remain (-1,-1,-1).
    """
    npz_files = sorted(glob.glob(os.path.join(camera_folder, "*.npz")), key=_step_id_from_path)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz found under: {camera_folder}")

    T = len(npz_files)
    world_all = np.full((T, 2, 21, 3), -1.0, dtype=np.float32)

    for t, npz_path in enumerate(npz_files):
        # load raw camera joints to know which slots are valid and map them into fixed (2,21,3)
        d = np.load(npz_path, allow_pickle=True)
        joints_cam_all = d["hand_joints_3d"].astype(np.float32)  # (2,21,3)

        # convert BOTH slots to world (even if invalid, it will stay -1)
        # we reuse your compute_world_coord_from_camera by forcing hand_slot, but to keep minimal changes,
        # just do conversion inline once for slot 0/1.
        camera_name = os.path.basename(os.path.dirname(npz_path))
        extrinsics, _ = load_extrinsics_yaml(extrinsics_yaml_path)
        if camera_name not in extrinsics:
            raise KeyError(f"Camera name {camera_name} not found in extrinsics yaml")
        T_world_cam = extrinsics[camera_name]

        for slot in [0, 1]:
            joints_cam = joints_cam_all[slot]
            valid_mask = joints_cam[:, 0] > SENTINEL
            if not valid_mask.any():
                continue
            ones = np.ones((valid_mask.sum(), 1), dtype=np.float32)
            pts_h = np.hstack([joints_cam[valid_mask], ones])
            pts_w = (T_world_cam @ pts_h.T).T[:, :3]
            world_all[t, slot, valid_mask] = pts_w

    return world_all, npz_files


def main():
    camera_folder = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/037522251142"
    print("camera_folder:", camera_folder)

    world_joints_all_steps, npz_files = load_episode_world_joints_from_camera_folder(
        camera_folder=camera_folder,
        extrinsics_yaml_path=EXTR_YAML,
    )
    print("loaded steps:", len(npz_files))
    print("world_joints_all_steps shape:", world_joints_all_steps.shape)  # (T,2,21,3)

    delta = compute_delta_eef_from_world_joints(
        world_joints_all_steps,
        wrist_idx=0,
        thumb_knuckle_idx=2,
        index_knuckle_idx=5,
    )

    print("left_eef shape:", delta["left_eef"].shape)
    print("right_eef shape:", delta["right_eef"].shape)

    # 如果你想看第1个delta:
    print("left_eef[0]:", delta["left_eef"][0])
    print("right_eef[0]:", delta["right_eef"][0])


if __name__ == "__main__":
    main()
