MIMICGEN_FEATURES = {
    # --- images ---
    "observation.images.agentview_image": {
        "dtype": "video",               
        "shape": (84, 84, 3),
        "names": ["height", "width", "rgb"],
        "camera_dir": "observation.images.agentview_image",
        "source_key": "obs/agentview_image"
    },
    "observation.images.robot0_eye_in_hand_image": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "rgb"],
        "camera_dir": "observation.images.robot0_eye_in_hand_image",
        "source_key": "obs/robot0_eye_in_hand_image"
    },

    # --- low-dim states (LIBERO-like) ---
    # ee_state = [x,y,z, axis_angle1, axis_angle2, axis_angle3]
    "observation.states.robot0_eef_pos": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z"]},
        "source_key": "obs/robot0_eef_pos"
    },
    
    "observation.states.robot0_eef_pos_rel_pod": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z"]},
        "source_key": "obs/robot0_eef_pos_rel_pod"
    },
    
    "observation.states.robot0_eef_pos_rel_pod_holder": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z"]},
        "source_key": "obs/robot0_eef_pos_rel_pod_holder"
    },
    
    "observation.states.robot0_eef_quat": {
        "dtype": "float32",
        "shape": (4,),
        "names": {"motors": ["x", "y", "z", "w"]},
        "source_key": "obs/robot0_eef_quat"
    },
    
    "observation.states.robot0_eef_quat_rel_pod": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z", "w"]},
        "source_key": "obs/robot0_eef_quat_rel_pod"
    },
    
    "observation.states.robot0_eef_quat_rel_pod_holder": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z", "w"]},
        "source_key": "obs/robot0_eef_quat_rel_pod_holder"
    },
    
    "observation.states.robot0_eef_vel_ang": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z", "w"]},
        "source_key": "obs/robot0_eef_vel_ang"
    },
    
    "observation.states.robot0_eef_vel_lin": {
        "dtype": "float32",
        "shape": (3,),
        "names": {"motors": ["x", "y", "z", "w"]},
        "source_key": "obs/robot0_eef_vel_lin"
    },
    
    "observation.states.robot0_gripper_qpos": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper_left", "gripper_right"]},
        "source_key": "obs/robot0_gripper_qpos"
    },
    
    "observation.states.robot0_gripper_qvel": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper_left", "gripper_right"]},
        "source_key": "obs/robot0_gripper_qvel"
    },

    "observation.states.joint_pos": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        "source_key": "obs/robot0_joint_pos"
    },
    
    "observation.states.joint_pos_cos": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        "source_key": "obs/robot0_joint_pos_cos"
    },
    
    "observation.states.joint_pos_sin": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        "source_key": "obs/robot0_joint_pos_sin"
    },
    
    "observation.states.joint_vel": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        "source_key": "obs/robot0_joint_vel"
    },


    # state = ee_state (6) + gripper_state (2) = 8
    "observation.rewards": {
        "dtype": "float32",
        "shape": (1,),
        "names": {"motors": ["rewards"]},
        "source_key": "rewards"
    },

    # --- action ---
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
        "source_key": "actions",
    },
}
