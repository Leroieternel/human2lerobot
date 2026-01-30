#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan HO-Cap labels and report missing hand joints.

What it does:
- Traverse /mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_{1-9}/<episode_id>/<camera_id>/*.npz
- For each label npz:
  - read hand_joints_3d (expected shape (2,21,3))
  - detect joints where (x,y,z)==(-1,-1,-1)  (i.e., "three -1 in a row")
- Aggregate statistics per episode_id, per camera_id:
  - total steps (npz count)
  - for each step: number of missing joints (per hand slot and total)
  - list exact step ids (frame index from filename) and which joint indices are missing
- Save full report to JSON.

Usage:
  python scan_hocap_missing_joints.py \
    --base /mnt/central_storage/data_pool_world/HO-Cap/datasets \
    --out  /raid/xiangyi.jia/data_processing/hocap/missing_joints_report.json
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


LABEL_RE = re.compile(r"(?:label_)?(\d+)\.npz$", re.IGNORECASE)


def extract_step_id(npz_path: str) -> str:
    """Extract step/frame id from filename like label_000444.npz or 000444.npz."""
    name = os.path.basename(npz_path)
    m = LABEL_RE.search(name)
    return m.group(1) if m else name


def is_triple_minus_one(xyz: np.ndarray) -> np.ndarray:
    """
    xyz: (..., 3)
    Return boolean mask (...,) where all three coords == -1.
    """
    return np.all(xyz == -1.0, axis=-1)


def scan_episode_camera(camera_dir: str) -> dict:
    """
    Scan one camera folder of an episode, return stats.
    camera_dir: .../<episode_id>/<camera_id>/
    """
    npz_files = sorted(glob.glob(os.path.join(camera_dir, "*.npz")))
    total_steps = len(npz_files)

    step_details = {}
    steps_with_any_missing = 0
    total_missing_joints_over_steps = 0  # sum over steps (total missing joints across both hands)

    for npz_path in npz_files:
        step_id = extract_step_id(npz_path)

        try:
            d = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            step_details[step_id] = {
                "npz_path": npz_path,
                "error": f"failed_to_load: {repr(e)}",
            }
            continue

        if "hand_joints_3d" not in d.files:
            step_details[step_id] = {
                "npz_path": npz_path,
                "error": "missing_key: hand_joints_3d",
                "keys": list(d.files),
            }
            continue

        j3d = d["hand_joints_3d"]

        if not (isinstance(j3d, np.ndarray) and j3d.shape == (2, 21, 3)):
            step_details[step_id] = {
                "npz_path": npz_path,
                "error": "unexpected_shape_or_type",
                "shape": list(getattr(j3d, "shape", [])),
                "dtype": str(getattr(j3d, "dtype", "")),
            }
            continue

        miss_mask = is_triple_minus_one(j3d)  # (2,21)
        miss_count_per_hand = miss_mask.sum(axis=1).astype(int).tolist()  # [h0,h1]
        miss_total = int(miss_mask.sum())

        miss_idx_per_hand = []
        for h in range(2):
            idx = np.where(miss_mask[h])[0].astype(int).tolist()
            miss_idx_per_hand.append(idx)

        # Record only if any missing OR if you want all steps; user wants:
        # "每个step有多少个hand_joints_3d出现... 并给出具体是哪个step"
        # -> keep all steps but store missing=0 for clean ones.
        step_details[step_id] = {
            "npz_path": npz_path,
            "missing_joints_total": miss_total,
            "missing_joints_per_hand": miss_count_per_hand,
            "missing_joint_indices_per_hand": miss_idx_per_hand,
        }

        if miss_total > 0:
            steps_with_any_missing += 1
            total_missing_joints_over_steps += miss_total

    return {
        "total_steps": total_steps,
        "steps_with_any_missing": steps_with_any_missing,
        "total_missing_joints_over_steps": total_missing_joints_over_steps,
        "step_details": step_details,  # step_id -> detail
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="/mnt/central_storage/data_pool_world/HO-Cap/datasets",
        help="HO-Cap datasets root (contains subject_1..subject_9).",
    )
    ap.add_argument(
        "--out",
        default="missing_joints_report.json",
        help="Output JSON path.",
    )
    ap.add_argument(
        "--subjects",
        default="1-9",
        help='Subjects range, e.g. "1-9" or "1,2,5". Default: 1-9',
    )
    ap.add_argument(
        "--only_steps_with_missing",
        action="store_true",
        help="If set, JSON step_details will include only steps where missing_joints_total>0.",
    )
    args = ap.parse_args()

    # Parse subjects
    subjects = []
    s = args.subjects.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        subjects = list(range(int(a), int(b) + 1))
    else:
        subjects = [int(x) for x in s.split(",") if x.strip()]

    report = {
        "base": args.base,
        "subjects": subjects,
        "episodes": {},  # episode_id -> camera_id -> stats
    }

    # Traverse
    for subj in subjects:
        subj_dir = os.path.join(args.base, f"subject_{subj}")
        if not os.path.isdir(subj_dir):
            print(f"[WARN] not found: {subj_dir}")
            continue

        episode_ids = sorted(
            [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
        )

        for episode_id in episode_ids:
            episode_dir = os.path.join(subj_dir, episode_id)
            # camera folders are immediate children that are directories
            camera_ids = sorted(
                [d for d in os.listdir(episode_dir) if os.path.isdir(os.path.join(episode_dir, d))]
            )

            # skip obvious non-camera folders if needed (meta.yaml, poses_*.npy are files so already excluded)
            # We'll keep hololens_* too; it likely has 0 npz and will show total_steps=0.
            for camera_id in camera_ids:
                camera_dir = os.path.join(episode_dir, camera_id)

                stats = scan_episode_camera(camera_dir)

                # Optionally keep only missing steps
                if args.only_steps_with_missing:
                    filtered = {
                        k: v
                        for k, v in stats["step_details"].items()
                        if isinstance(v, dict) and v.get("missing_joints_total", 0) > 0
                    }
                    stats["step_details"] = filtered

                # Store into report
                report["episodes"].setdefault(episode_id, {})
                report["episodes"][episode_id].setdefault(camera_id, {})
                report["episodes"][episode_id][camera_id] = stats

    # Write JSON
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print compact summary to stdout
    print("\n=== Summary (episode_id / camera_id) ===")
    # Sort by episode then camera
    for episode_id in sorted(report["episodes"].keys()):
        cams = report["episodes"][episode_id]
        for camera_id in sorted(cams.keys()):
            st = cams[camera_id]
            ts = st["total_steps"]
            miss_steps = st["steps_with_any_missing"]
            miss_sum = st["total_missing_joints_over_steps"]
            print(
                f"{episode_id}  {camera_id:>12}  steps={ts:6d}  "
                f"steps_with_missing={miss_steps:6d}  total_missing_joints={miss_sum:8d}"
            )

    print(f"\n[OK] JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
