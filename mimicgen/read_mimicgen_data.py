import h5py
import matplotlib.pyplot as plt
import json

# h5_path = "/mnt/central_storage/data_pool_world/mimicgen_datasets/core/three_piece_assembly_d0.hdf5"
h5_path = "/mnt/central_storage/data_pool_world/mimicgen_datasets/core/coffee_d0.hdf5"
out_path = "agentview_image shape_demo7_frame100.png"

# save one image from a sample demo
with h5py.File(h5_path, "r") as f:
    img = f["data/demo_7/obs/agentview_image"][100] 

# img shape: (84, 84, 3), dtype=uint8
plt.imsave(out_path, img)

with h5py.File(h5_path, "r") as f:
    # check number of demos in the hdf5 
    data_grp = f["data"]
    demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])

    print("number of demos =", len(demo_keys))
    print("first 20 demos:", demo_keys[:20])
    
    # check obs and actions of one demo
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
    
    # read actions
    actions = f[f"data/{demo}/actions"][:]   # shape (T, 7)

    print("actions shape:", actions.shape)
    
    # print first 100 values of action[-1] (gripper)
    gripper = actions[:100, -1]
    print("action[-1] (gripper) first 100 values:")
    print(gripper)
    
    # check fps from env_args
    env_args = json.loads(f["data"].attrs["env_args"])
    print(env_args.keys())
    print(json.dumps(env_args, indent=2)[:2000])
