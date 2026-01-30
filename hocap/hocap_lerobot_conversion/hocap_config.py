HOCAP_CONFIG = {
    "images": {
        "camera_1": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "037522251142",
        },
        "camera_2": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "043422252387",
        },
        "camera_3": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "046122250168",
        },
        "camera_4": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "105322251225",
        },
        "camera_5": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "105322251564",
        },
        "camera_6": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "108222250342",
        },
        "camera_7": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "115422250549",
        },
        "camera_8": {
            "dtype": "video",
            "shape": (480, 640, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "117222250549",
        },
        "hololens": {
            "dtype": "video",
            "shape": (720, 1280, 3),   
            "names": ["height", "width", "rgb"],
            "source_key": "hololens_kv5h72",
        },
    },

    "states": {
        "left_delta_end_effector": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "source_key": "left_end_effector",
        },
        "right_delta_end_effector": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "source_key": "right_end_effector",
        },
    },

    "actions": {
        "left_delta_end_effector": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "source_key": "actions",
        },
        "right_delta_end_effector": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "source_key": "actions",
        },
    },
}
