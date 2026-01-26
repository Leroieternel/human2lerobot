# mimicgen_config_min.py
# A minimal, robomind2lerobot-like config style: images / states / actions.
# Explicit source_key is added to map from MimicGen HDF5 obs keys.

MIMICGEN_MIN_CONFIG = {
    "images": {
        "camera_front": {
            "dtype": "video",
            "shape": (84, 84, 3),   # overwritten by actual frames
            "names": ["height", "width", "rgb"],
            # from /data/demo_x/obs/agentview_image
            "source_key": "agentview_image",
        },
        "camera_wrist": {
            "dtype": "video",
            "shape": (84, 84, 3),   # overwritten by actual frames
            "names": ["height", "width", "rgb"],
            # from /data/demo_x/obs/robot0_eye_in_hand_image
            "source_key": "robot0_eye_in_hand_image",
        },
    },

    "states": {
        # /data/demo_x/obs/robot0_eef_pos
        "robot0_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["x", "y", "z"]},
            "source_key": "robot0_eef_pos",
        },
        
        # EEF velocity
        "robot0_eef_vel_lin": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["vx", "vy", "vz"]},
            "source_key": "robot0_eef_vel_lin",
        },
        "robot0_eef_vel_ang": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["wx", "wy", "wz"]},
            "source_key": "robot0_eef_vel_ang",
        },

        # /data/demo_x/obs/robot0_eef_quat
        "robot0_eef_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"motors": ["qx", "qy", "qz", "qw"]},
            "source_key": "robot0_eef_quat",
        },

        # standardized gripper (1D), from raw qpos
        # /data/demo_x/obs/robot0_gripper_qpos
        "robot0_gripper_qpos": {
            "dtype": "float32",
            "shape": None,
            "names": {"motors": ["finger_0", "finger_1"]},
            "source_key": "robot0_gripper_qpos",
        },
        "robot0_gripper_qvel": {
            "dtype": "float32",
            "shape": None,
            "names": {"motors": ["finger_0_vel", "finger_1_vel"]},
            "source_key": "robot0_gripper_qvel",
        },
        # standardized output (1D): written by your converter after standardize_gripper()
        "robot0_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"motors": ["gripper_open"]},
            "source_key": "robot0_gripper_qpos",
            "postprocess": "standardize_gripper",
        },

        # /data/demo_x/obs/robot0_joint_pos
        # joint dim differs by robot; inferred at runtime
        "robot0_joint_pos": {
            "dtype": "float32",
            "shape": None,
            "names": None,
            "source_key": "robot0_joint_pos",
        },
        "robot0_joint_pos_cos": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"joint_{i}_cos" for i in range(7)]},
            "source_key": "robot0_joint_pos_cos",
        },
        "robot0_joint_pos_sin": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"joint_{i}_sin" for i in range(7)]},
            "source_key": "robot0_joint_pos_sin",
        },
        "robot0_joint_vel": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"joint_{i}_vel" for i in range(7)]},
            "source_key": "robot0_joint_vel",
        },
    },

    "actions": {
        # /data/demo_x/actions
        # action dim inferred at runtime
        "action": {
            "dtype": "float32",
            "shape": None,
            "names": None,
            "source_key": "actions",
        },
    },
}
