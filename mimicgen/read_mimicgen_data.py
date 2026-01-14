import h5py
import matplotlib.pyplot as plt

# h5_path = "/mnt/central_storage/data_pool_world/mimicgen_datasets/core/three_piece_assembly_d0.hdf5"
h5_path = "/mnt/central_storage/data_pool_world/mimicgen_datasets/core/coffee_d0.hdf5"
out_path = "agentview_image shape_demo7_frame0.png"

# with h5py.File(h5_path, "r") as f:
#     img = f["data/demo_7/obs/agentview_image"][10]  # 第一帧

# # img shape: (84, 84, 3), dtype=uint8
# plt.imsave(out_path, img)

with h5py.File(h5_path, "r") as f:
    demo = "demo_0"
    obs = f[f"data/{demo}/obs"]
    keys = sorted(list(obs.keys()))
    print("obs keys count =", len(keys))
    print("first 50 keys:", keys[:50])
    print("has robot0_eef_pos_rel_pod ?", "robot0_eef_pos_rel_pod" in obs)
    print("has robot0_eef_pos ?", "robot0_eef_pos" in obs)
    
    for key in keys:
        dset = obs[key]
        print(f"{key:35s} shape={dset.shape}, dtype={dset.dtype}")
    
    # 👉 读取 actions
    actions = f[f"data/{demo}/actions"][:]   # shape (T, 7)

    print("actions shape:", actions.shape)
    
    # 👉 打印最后一个维度（gripper）的前 100 个
    gripper = actions[:100, -1]
    print("action[-1] (gripper) first 100 values:")
    print(gripper)
