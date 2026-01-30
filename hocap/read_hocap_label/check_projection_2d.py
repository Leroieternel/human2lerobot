import numpy as np

npz_path = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/105322251564/label_000444.npz"
# npz_path = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/037522251142/label_000444.npz"
d = np.load(npz_path, allow_pickle=True)

K = d["cam_K"].astype(np.float32)                 # (3,3)
j3d = d["hand_joints_3d"][1].astype(np.float32)   # (21,3) 选slot=1
j2d = d["hand_joints_2d"][1].astype(np.float32)   # (21,2)

# 有效点mask：3D不是(-1,-1,-1) 且 2D不是(-1,-1)
valid = (j3d[:,0] > -0.5) & (j2d[:,0] > -0.5)

fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
X, Y, Z = j3d[:,0], j3d[:,1], j3d[:,2]
Z = np.maximum(Z, 1e-6)

u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
proj = np.stack([u, v], axis=1)

err = np.linalg.norm(proj[valid] - j2d[valid], axis=1)

print("valid joints:", int(valid.sum()), "/ 21")
print("mean pixel err:", float(err.mean()))
print("max  pixel err:", float(err.max()))
print("per-joint err:", np.round(err, 2))
