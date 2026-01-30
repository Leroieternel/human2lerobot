import numpy as np

p = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/037522251142/label_000000.npz"
p1 = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/105322251564/label_000000.npz"
d = np.load(p, allow_pickle=True)
d1 = np.load(p1, allow_pickle=True) 

print("keys:", d.files)
for k in d.files:
    print("key: ", k)
    v = d[k]
    try:
        print(k, type(v), getattr(v, "shape", None), v.dtype if hasattr(v, "dtype") else "")
    except Exception as e:
        print(k, type(v), "err:", e)
    
    if k == "hand_joints_3d" or k == "hand_joints_2d":
        print("first 2 entries:\n", v[:2])
        
        print("keys:", d.files)
        
for k in d1.files:
    print("key: ", k)
    v = d1[k]
    try:
        print(k, type(v), getattr(v, "shape", None), v.dtype if hasattr(v, "dtype") else "")
    except Exception as e:
        print(k, type(v), "err:", e)
    
    if k == "hand_joints_3d" or k == "hand_joints_2d":
        print("first 2 entries:\n", v[:2])
        

