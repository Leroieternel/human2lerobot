import torch
import numpy as np

def standardize_gripper(gripper_qpos, robot_type, gripper_type=None):
    """
    Standardizes gripper state to 1D.
    IMPORTANT!!! No scalling is applied!!!!!
    Args:
        gripper_qpos: numpy array or torch tensor of shape (..., D)
                      D=2 for PandaGripper (Panda). both entries increase when opening
                      D=6 for Robotiq85 (UR5e). 0dimension is the driver joint. Higher values mean closing.
                      D=6 for Robotiq140 (iiwa). 0dimension is the driver joint. Higher values mean closing.
                      D=2 for RethinkGripper (Sawyer). 0dimension increases when opening, 1dimension decreases when opening.
        robot_type: string specifying the robot type (e.g., "Panda", "UR5e", "iiwa", "sawyer").
        gripper_type: Optional string specifying the gripper type (e.g., "RethinkGripper", "PandaGripper", "Robotiq85", "Robotiq140").
                      If provided, allows for specific handling logic.
    Returns:
        standardized_gripper: shape (..., 1)
        PandaGripper and RethinkGripper: sum of absolute values of both joints (higher means more open)
        Robotiq85 and Robotiq140: negative of the driver joint (higher means more open)
    """
    is_torch = isinstance(gripper_qpos, torch.Tensor)
    
    # get gripper type if not provided. The pairing is based on common robot-gripper combinations:  https://robosuite.ai/docs/modules/robots.html
    valid_robot_types = ["Panda", "UR5e", "IIWA", "Sawyer"]
    valid_gripper_types = ["PandaGripper", "Robotiq85", "Robotiq140", "RethinkGripper"]
    robot_to_gripper = {
        "Panda": "PandaGripper",     # 2
        "UR5e": "Robotiq85",         # 6
        "IIWA": "Robotiq140",        # 6
        "Sawyer": "RethinkGripper",  # 2
    }
    expected_dims = {
        "PandaGripper": 2,
        "RethinkGripper": 2,
        "Robotiq85": 6,
        "Robotiq140": 6,
    }
    assert robot_type in valid_robot_types, f"Unknown robot type: {robot_type}"

    if gripper_type is None:
        gripper_type = robot_to_gripper[robot_type]
    else:
        assert gripper_type in valid_gripper_types, f"Unknown gripper type: {gripper_type}"
    # Check if the dimension of gripper_qpos matches expected for the gripper type
    input_dim = gripper_qpos.shape[-1]
    expected_dim = expected_dims[gripper_type]
    assert input_dim == expected_dim, (
        f"gripper_qpos has wrong dimension for {gripper_type}: "
        f"expected {expected_dim}, got {input_dim}"
    )    
     # Handler functions for each gripper type
    if is_torch:
        handlers = {
            "PandaGripper": lambda q: q.abs().sum(dim=-1, keepdim=True),
            "RethinkGripper": lambda q: q.abs().sum(dim=-1, keepdim=True),
            "Robotiq85": lambda q: -q[..., [0]], # negative of driver joint
            "Robotiq140": lambda q: -q[..., [0]], # negative of driver joint
        }
    else:
        handlers = {
            "PandaGripper": lambda q: np.abs(q).sum(axis=-1, keepdims=True),
            "RethinkGripper": lambda q: np.abs(q).sum(axis=-1, keepdims=True),
            "Robotiq85": lambda q: -q[..., [0]], # negative of driver joint
            "Robotiq140": lambda q: -q[..., [0]], # negative of driver joint
        }
    return handlers[gripper_type](gripper_qpos)
