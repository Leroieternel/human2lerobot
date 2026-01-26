# mimicgen_config_hammer_kitchen.py
# Hammer-kitchen task config (robomind2lerobot-like style): images / states / actions.
# Obs keys & shapes are from demo_0/obs/* (T=323), so we store per-step shapes.

MIMICGEN_HAMMER_KITCHEN_CONFIG = {
    "images": {
        "camera_front": {
            "dtype": "video",
            "shape": (84, 84, 3),   # overwritten by actual frames at runtime
            "names": ["height", "width", "rgb"],
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
        # Object state (28 dims for this task)
        # "object": {
        #     "dtype": "float32",
        #     "shape": (28,),
        #     "names": {"state": [f"object_{i}" for i in range(28)]},
        #     "source_key": "object",
        # },

        # Contact & force scalar signals
        "robot0_contact": {
            "dtype": "bool",
            "shape": (1,),
            "names": {"signals": ["contact"]},
            "source_key": "robot0_contact",
        },
        "robot0_eef_force_norm": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"signals": ["eef_force_norm"]},
            "source_key": "robot0_eef_force_norm",
        },

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

        # Gripper raw + standardized 1D
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
        "robot0_gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"motors": ["gripper_open"]},
            "source_key": "robot0_gripper_qpos",
            "postprocess": "standardize_gripper",
        },

        # Joint states (7-DoF in this task)
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
        # action dim is often (7,) in MimicGen; keep None for robustness if unsure.
        "action": {
            "dtype": "float32",
            "shape": None,
            "names": None,
            "source_key": "actions",
        },
    },
}
