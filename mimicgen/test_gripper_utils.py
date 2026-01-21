import numpy as np
import torch
from mimicgen.gripper_utils import standardize_gripper

def test_standardize_gripper_numpy():
    # PandaGripper: D=2, both entries increase when opening
    qpos = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = standardize_gripper(qpos, 'Panda')
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.abs(qpos).sum(axis=-1, keepdims=True))

    # Robotiq85: D=6, 0th entry is driver joint
    qpos = np.array([[0.5, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]])
    out = standardize_gripper(qpos, 'UR5e')
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, -qpos[:, [0]])

    # RethinkGripper: D=2
    qpos = np.array([[0.1, -0.1], [0.2, -0.2]])
    out = standardize_gripper(qpos, 'sawyer')
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out, np.abs(qpos).sum(axis=-1, keepdims=True))

def test_standardize_gripper_torch():
    # PandaGripper: D=2
    qpos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    out = standardize_gripper(qpos, 'Panda')
    assert out.shape == (2, 1)
    torch.testing.assert_close(out, qpos.abs().sum(dim=-1, keepdim=True))

    # Robotiq140: D=6
    qpos = torch.tensor([[0.5, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]])
    out = standardize_gripper(qpos, 'iiwa')
    assert out.shape == (2, 1)
    torch.testing.assert_close(out, -qpos[:, [0]])

if __name__ == "__main__":
    test_standardize_gripper_numpy()
    test_standardize_gripper_torch()
    print("All tests passed.")
