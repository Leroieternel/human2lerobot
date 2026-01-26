# mimicgen_config_coffee.py
# Coffee task config (robomind2lerobot-like style): images / states / actions.
# Obs keys list is from coffee_d*.hdf5 (demo_0/obs/*).

MIMICGEN_COFFEE_CONFIG = {
    "images": {
        "camera_front": {
            "dtype": "video",
            "shape": (84, 84, 3),   # overwritten by actual frames at runtime
            "names": ["height", "width", "rgb"],
            # optional: source key in HDF5 obs (if your converter uses it)
            "source_key": "agentview_image",
        },
        "camera_wrist": {
            "dtype": "video",
            "shape": (84, 84, 3),   # overwritten by actual frames at runtime
            "names": ["height", "width", "rgb"],
            "source_key": "robot0_eye_in_hand_image",
        },
    },

    "states": {
        # # Object state
        # "object": {
        #     "dtype": "float32",
        #     "shape": (57,),
        #     "names": {"state": [f"object_{i}" for i in range(57)]},
        #     "source_key": "object",
        # },

        # EEF pose (absolute)
        "robot0_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["x", "y", "z"]},
            "source_key": "robot0_eef_pos",
        },
        "robot0_eef_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"motors": ["qx", "qy", "qz", "qw"]},
            "source_key": "robot0_eef_quat",
        },

        # EEF pose relative to pod / pod holder
        "robot0_eef_pos_rel_pod": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["dx", "dy", "dz"]},
            "source_key": "robot0_eef_pos_rel_pod",
        },
        "robot0_eef_pos_rel_pod_holder": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"motors": ["dx", "dy", "dz"]},
            "source_key": "robot0_eef_pos_rel_pod_holder",
        },
        "robot0_eef_quat_rel_pod": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"motors": ["dqx", "dqy", "dqz", "dqw"]},
            "source_key": "robot0_eef_quat_rel_pod",
        },
        "robot0_eef_quat_rel_pod_holder": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"motors": ["dqx", "dqy", "dqz", "dqw"]},
            "source_key": "robot0_eef_quat_rel_pod_holder",
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

        # Gripper (raw qpos/qvel) + standardized 1D
        # NOTE: your converter typically reads robot0_gripper_qpos then standardizes to 1D.
        "robot0_gripper_qpos": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["finger_0", "finger_1"]},
            "source_key": "robot0_gripper_qpos",
        },
        "robot0_gripper_qvel": {
            "dtype": "float32",
            "shape": (2,),
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

        # Joint states
        "robot0_joint_pos": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"joint_{i}" for i in range(7)]},
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
        # From your note: actions shape (T,7). Keep None if you want runtime inference,
        # or set (7,) explicitly.
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"motors": [f"a{i}" for i in range(7)]},
            "source_key": "actions",
        },
    },
}
