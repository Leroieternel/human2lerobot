import sys
import os
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils.skeleton_tfs import RIGHT_FINGERS, RIGHT_INDEX, RIGHT_THUMB, RIGHT_RING, RIGHT_MIDDLE, RIGHT_LITTLE
from utils.skeleton_tfs import LEFT_FINGERS, LEFT_INDEX, LEFT_THUMB, LEFT_RING, LEFT_MIDDLE, LEFT_LITTLE
from utils.data_utils import convert_to_camera_frame
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R

QUERY_TFS = RIGHT_FINGERS + ['rightHand', 'rightForearm'] + LEFT_FINGERS + ['leftHand', 'leftForearm']

from simple_dataset import SimpleDataset

# Keypoints of left and right hand
RIGHT_WRIST = "rightHand"
RIGHT_THUMB_KNUCKLE = "rightThumbKnuckle"
RIGHT_INDEX_KNUCKLE = "rightIndexFingerKnuckle"
RIGHT_FINGERTIPS = [
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
]

LEFT_WRIST = "leftHand"
LEFT_THUMB_KNUCKLE = "leftThumbKnuckle"
LEFT_INDEX_KNUCKLE = "leftIndexFingerKnuckle"
LEFT_FINGERTIPS = [
    "leftThumbTip",
    "leftIndexFingerTip",
    "leftMiddleFingerTip",
    "leftRingFingerTip",
    "leftLittleFingerTip",
]

ALL_JOINTS_FOR_CONVERSION = (
    [RIGHT_WRIST, RIGHT_THUMB_KNUCKLE, RIGHT_INDEX_KNUCKLE]
    + RIGHT_FINGERTIPS
    + [LEFT_WRIST, LEFT_THUMB_KNUCKLE, LEFT_INDEX_KNUCKLE]
    + LEFT_FINGERTIPS
)


def compute_rpy_from_points(A, B, C):
    """
    A: wrist
    B: thumb knuckle
    C: index knuckle
    
    Use plane normal to compute RPY orientation
    """
    v1 = B - A
    v2 = C - A

    # normal vector
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # 构建一个“手部坐标系”：z=normal, x=v1方向
    z_axis = normal
    x_axis = v1 / (np.linalg.norm(v1) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3×3 rotation matrix
    rpy = R.from_matrix(R_mat).as_euler("xyz")          # roll, pitch, yaw

    return rpy


# Compute delta
def compute_delta(x):
    """
    x: array of shape (N, ...)
    returns delta_x of shape (N, ...)
    where delta_x[t] = x[t+1] - x[t]
    and delta_x[N-1] = 0
    """
    delta = x[1:] - x[:-1]
    # pad last frame with zeros
    pad = np.zeros((1,) + x.shape[1:], dtype=x.dtype)
    delta = np.concatenate([delta, pad], axis=0)
    return delta


def compute_rpy_from_matrix(R_mat):
    """
    R_mat: (3,3) rotation matrix
    return: (3,) euler xyz angles
    """
    return R.from_matrix(R_mat).as_euler("xyz")


# extract delta eef (left hand + right hand)
def extract_delta_eef_from_single_episode(h5_path):

    with h5py.File(h5_path, "r") as f:

        N = f["transforms/rightHand"].shape[0]
        
        eef_right_xyz_list = []
        eef_right_rpy_list = []
        eef_left_xyz_list = []
        eef_left_rpy_list = []
        joints_xyz_list = []
        name_to_idx = {name: idx for idx, name in enumerate(ALL_JOINTS_FOR_CONVERSION)}

        for i in range(N):
            # Camera extrinsic of this frame
            cam_ext = f["transforms/camera"][i]
            tfs_world = []    # world frame
            
            # append tf world coordinate
            for joint_name in ALL_JOINTS_FOR_CONVERSION:
                tfs_world.append(f["transforms/" + joint_name][i])
            tfs_world = np.stack(tfs_world, axis=0)  # (M, 4, 4)
            
            # convert to camera frame： T_cam = inv(cam_ext) @ T_world
            tfs_cam = convert_to_camera_frame(tfs_world, cam_ext)  # (M, 4, 4)

            # --- Right hand transforms ---
            right_wrist_position = tfs_cam[name_to_idx[RIGHT_WRIST], :3, 3]    # right wrist xyz
            right_thumb_knuckle = tfs_cam[name_to_idx[RIGHT_THUMB_KNUCKLE], :3, 3]
            right_index_knuckle = tfs_cam[name_to_idx[RIGHT_INDEX_KNUCKLE], :3, 3]
            
            rpy_r = compute_rpy_from_points(right_wrist_position, right_thumb_knuckle, right_index_knuckle)
            eef_right_xyz_list.append(right_wrist_position)
            eef_right_rpy_list.append(rpy_r)
            
            # --- Left hand transforms ---
            left_wrist_position = tfs_cam[name_to_idx[LEFT_WRIST], :3, 3]     # left wrist xyz
            left_thumb_knuckle = tfs_cam[name_to_idx[LEFT_THUMB_KNUCKLE], :3, 3]
            left_index_knuckle = tfs_cam[name_to_idx[LEFT_INDEX_KNUCKLE], :3, 3]

            rpy_l = compute_rpy_from_points(left_wrist_position, left_thumb_knuckle, left_index_knuckle)

            eef_left_xyz_list.append(left_wrist_position)
            eef_left_rpy_list.append(rpy_l)
            
            # --- left & right wrist xyz + fingertip xyz ---
            joint_xyz_frame = []
            
            
            # right wrist RPY
            right_wrist_R = tfs_cam[name_to_idx[RIGHT_WRIST], :3, :3]
            right_wrist_rpy = compute_rpy_from_matrix(right_wrist_R)

            # left wrist RPY
            left_wrist_R = tfs_cam[name_to_idx[LEFT_WRIST], :3, :3]
            left_wrist_rpy = compute_rpy_from_matrix(left_wrist_R)
            

            # right wrist: append directly
            joint_xyz_frame.append(right_wrist_rpy)
            # right 5 fingertips
            for joint in RIGHT_FINGERTIPS:
                joint_xyz_frame.append(tfs_cam[name_to_idx[joint], :3, 3])

            # left wrist: append directly
            joint_xyz_frame.append(left_wrist_rpy)
            # left 5 fingertips
            for joint in LEFT_FINGERTIPS:
                joint_xyz_frame.append(tfs_cam[name_to_idx[joint], :3, 3])

            joint_xyz_frame = np.stack(joint_xyz_frame, axis=0)  # (12, 3)
            joints_xyz_list.append(joint_xyz_frame)

        # Convert to numpy
        eef_right_xyz = np.stack(eef_right_xyz_list, axis=0)  # (N, 3)
        eef_right_rpy = np.stack(eef_right_rpy_list, axis=0)  # (N, 3)
        eef_left_xyz = np.stack(eef_left_xyz_list, axis=0)    # (N, 3)
        eef_left_rpy = np.stack(eef_left_rpy_list, axis=0)    # (N, 3)
        joints_xyz = np.stack(joints_xyz_list, axis=0)        # (N, 12, 3)

        # ---- absolute eef ---- #
        # [ right_xyz(3), right_rpy(3), left_xyz(3), left_rpy(3) ] → 12d
        eef_abs = np.concatenate(
            [eef_right_xyz, eef_right_rpy, eef_left_xyz, eef_left_rpy],
            axis=1
        )  # (N, 12)
        
        # delta eef
        d_right_xyz = compute_delta(eef_right_xyz)
        d_right_rpy = compute_delta(eef_right_rpy)
        d_left_xyz = compute_delta(eef_left_xyz)
        d_left_rpy = compute_delta(eef_left_rpy)
        
        delta_eef = np.concatenate(
            [d_right_xyz, d_right_rpy, d_left_xyz, d_left_rpy],
            axis=1
        )  # (N,12)
        
        # ==== delta joints xyz ====
        delta_joints_xyz = compute_delta(joints_xyz)   # (N,12,3)

    return eef_abs, delta_eef, joints_xyz, delta_joints_xyz


def main(args):
    eef_abs, delta_eef, joints_xyz, delta_joints_xyz = extract_delta_eef_from_single_episode(
        args.h5_path
    )

    print("eef_abs shape:", eef_abs.shape)
    print("delta_eef shape:", delta_eef.shape)
    print("joints_xyz shape:", joints_xyz.shape)
    print("delta_joints_xyz shape:", delta_joints_xyz.shape)
    print("Example delta_eef[0:5]:")
    print(delta_eef[:5])
    print("All eef_abs:")
    # print(eef_abs)
    print("All joints_xyz:")
    # print(joints_xyz)
    print('delta eef: ', delta_eef)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', help='path to hdf5')
    # parser.add_argument('--data_dir', help='path to data directory')
    # parser.add_argument('--num_episodes', help='number of episodes to visualize', default=1)
    # parser.add_argument('--output_mp4', help='where to save the output video', default='output.mp4')
    args = parser.parse_args()

    try:
        main(args)
    except ValueError as exp:
        print("Error:", exp)