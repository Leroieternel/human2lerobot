#!/usr/bin/env python3
"""
Validate whether hand_joints_3d from two cameras, at the same frame, map to (approximately)
the same 3D points in a common world frame using HO-Cap extrinsics.

It will:
1) Load extrinsics YAML (3x4 per camera, rs_master identity).
2) Load two label_XXXXXX.npz files (hand_joints_3d: (2,21,3)).
3) Convert camera-frame joints -> world-frame joints using T_world_cam.
4) Match hand slots across the two cameras (handles 1 or 2 hands).
5) Report per-joint distances, mean/max error (meters + centimeters).
6) Also tries the inverse transform (T_cam_world) and tells you which direction fits better.

Usage:
  python validate_hocap_world_consistency.py
"""

import numpy as np
import yaml
from itertools import product

EXTR_YAML = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/calibration/extrinsics/extrinsics_20231014.yaml"

NPZ_A = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/037522251142/label_000444.npz"
NPZ_B = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1/20231025_165502/105322251564/label_000444.npz"

SERIAL_A = "037522251142"
SERIAL_B = "105322251564"

SENTINEL = -0.5  # joints with x <= -0.5 treated as invalid (since invalid is -1,-1,-1)


def ext12_to_T(ext12):
    """Convert 12-number list (row-major 3x4) to 4x4 homogeneous transform."""
    ext12 = np.asarray(ext12, dtype=np.float32).reshape(3, 4)
    T = np.eye(4, dtype=np.float32)
    T[:3, :] = ext12
    return T


def load_extrinsics(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    extr = y["extrinsics"]
    rs_master = y.get("rs_master", None)
    # Convert only camera serial keys (strings of digits) if you want; tags can stay too.
    Ts = {}
    for k, v in extr.items():
        Ts[str(k)] = ext12_to_T(v)
    return Ts, rs_master


def load_hand_joints(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    j = d["hand_joints_3d"].astype(np.float32)  # (2,21,3)
    # validity per hand-slot: use wrist (joint0) x and z checks
    slot_valid = []
    for s in range(j.shape[0]):
        wrist = j[s, 0]
        ok = (wrist[0] > SENTINEL) and (wrist[2] > 1e-6)
        slot_valid.append(ok)
    return j, np.array(slot_valid, dtype=bool)


def transform_points(T, pts):
    """T: 4x4, pts: (21,3) -> (21,3)"""
    pts = np.asarray(pts, dtype=np.float32)
    valid = pts[:, 0] > SENTINEL
    out = np.full_like(pts, -1.0, dtype=np.float32)
    if valid.sum() == 0:
        return out, valid
    ones = np.ones((valid.sum(), 1), dtype=np.float32)
    pts_h = np.hstack([pts[valid], ones])  # (N,4)
    out_valid = (T @ pts_h.T).T[:, :3]
    out[valid] = out_valid
    return out, valid


def compare_two_hands(worldA, validA, worldB, validB):
    """Return per-joint distances for intersection-valid joints."""
    valid = validA & validB
    if valid.sum() == 0:
        return None
    dist = np.linalg.norm(worldA[valid] - worldB[valid], axis=1)
    return dist


def best_slot_matching(jA_world, vA_slots, jB_world, vB_slots):
    """
    If there are multiple valid hand slots, match slots by minimum mean distance in world.
    Returns: (slotA, slotB, dist_array)
    """
    a_slots = [i for i, ok in enumerate(vA_slots) if ok]
    b_slots = [i for i, ok in enumerate(vB_slots) if ok]
    if len(a_slots) == 0 or len(b_slots) == 0:
        return None

    best = None
    for sa, sb in product(a_slots, b_slots):
        dist = compare_two_hands(jA_world[sa][0], jA_world[sa][1], jB_world[sb][0], jB_world[sb][1])
        if dist is None:
            continue
        score = float(dist.mean())
        if best is None or score < best["score"]:
            best = {"sa": sa, "sb": sb, "dist": dist, "score": score}
    return best


def run(direction_name, T_wA, T_wB):
    """
    direction_name: label for printing
    T_wA, T_wB: transforms applied to camera-frame joints to produce 'world' points
    """
    jA_cam, vA_slots = load_hand_joints(NPZ_A)
    jB_cam, vB_slots = load_hand_joints(NPZ_B)

    # Transform each slot to world
    jA_world = []
    for s in range(jA_cam.shape[0]):
        pts_w, valid = transform_points(T_wA, jA_cam[s])
        jA_world.append((pts_w, valid))
    jB_world = []
    for s in range(jB_cam.shape[0]):
        pts_w, valid = transform_points(T_wB, jB_cam[s])
        jB_world.append((pts_w, valid))

    best = best_slot_matching(jA_world, vA_slots, jB_world, vB_slots)
    if best is None:
        return None

    dist = best["dist"]
    out = {
        "direction": direction_name,
        "slotA": best["sa"],
        "slotB": best["sb"],
        "mean_m": float(dist.mean()),
        "max_m": float(dist.max()),
        "per_joint_cm": (dist * 100.0),
        "n_joints": int(dist.shape[0]),
    }
    return out


def main():
    Ts, rs_master = load_extrinsics(EXTR_YAML)
    assert SERIAL_A in Ts, f"Missing extrinsic for {SERIAL_A}"
    assert SERIAL_B in Ts, f"Missing extrinsic for {SERIAL_B}"

    T_A = Ts[SERIAL_A]
    T_B = Ts[SERIAL_B]

    # Hypothesis 1: YAML gives T_world_cam (camera -> world)
    res1 = run("Assume T_world_cam (camera→world) from YAML", T_A, T_B)

    # Hypothesis 2: YAML gives T_cam_world (world -> camera), so invert to get camera->world
    T_A_inv = np.linalg.inv(T_A)
    T_B_inv = np.linalg.inv(T_B)
    res2 = run("Assume T_cam_world (world→camera) from YAML, so use inverse", T_A_inv, T_B_inv)

    print("=== Extrinsics file ===")
    print(EXTR_YAML)
    print("rs_master:", rs_master)
    print("cam A serial:", SERIAL_A, "npz:", NPZ_A)
    print("cam B serial:", SERIAL_B, "npz:", NPZ_B)
    print()

    if res1 is None and res2 is None:
        print("No valid hand slots found in one or both npz files (all -1).")
        return

    def pretty(res):
        print(f"--- {res['direction']} ---")
        print(f"matched slots: camA slot={res['slotA']}  camB slot={res['slotB']}")
        print(f"valid joints used: {res['n_joints']} / 21")
        print(f"mean distance: {res['mean_m']:.6f} m  ({res['mean_m']*100:.3f} cm)")
        print(f"max  distance: {res['max_m']:.6f} m  ({res['max_m']*100:.3f} cm)")
        # Show a compact per-joint list
        cm = res["per_joint_cm"]
        print("per-joint distances (cm):", np.round(cm, 3).tolist())
        print()

    if res1 is not None:
        pretty(res1)
    if res2 is not None:
        pretty(res2)

    # Choose best by mean distance (if both exist)
    candidates = [r for r in [res1, res2] if r is not None]
    best = min(candidates, key=lambda r: r["mean_m"])
    print("=== Best direction by mean distance ===")
    print(best["direction"])
    print(f"mean: {best['mean_m']*100:.3f} cm, max: {best['max_m']*100:.3f} cm")
    print("If this is still large (>5-10 cm), check: (1) frame sync, (2) hand slot mismatch, (3) wrong serial/extrinsic file.")


if __name__ == "__main__":
    main()
