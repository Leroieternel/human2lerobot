#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import h5py


def iter_hdf5_files(root_dir: str):
    """Yield absolute paths of all .hdf5/.h5 files under root_dir recursively."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".hdf5", ".h5")):
                yield os.path.join(dirpath, fn)


def get_demos_and_sample(f: h5py.File) -> Tuple[int, Optional[str]]:
    """Return (#demos under /data, first demo name)."""
    if "data" not in f or not isinstance(f["data"], h5py.Group):
        return 0, None

    demos = [k for k in f["data"].keys() if k.startswith("demo_")]

    def demo_idx(s: str) -> int:
        try:
            return int(s.split("_")[1])
        except Exception:
            return 10**9

    demos = sorted(demos, key=demo_idx)
    return len(demos), (demos[0] if demos else None)


def list_top_level_keys(f: h5py.File) -> List[str]:
    return sorted(["/" + k for k in f.keys()])


def list_keys_under_group(f: h5py.File, group_path: str) -> List[str]:
    """List all keys (groups+datasets) under a specific group path, as absolute paths."""
    if group_path not in f:
        return []

    keys: List[str] = []
    grp = f[group_path]

    gp = group_path if group_path.startswith("/") else "/" + group_path
    keys.append(gp)

    def visitor(name, obj):
        if name == "":
            return
        full = (group_path.rstrip("/") + "/" + name).replace("//", "/")
        if not full.startswith("/"):
            full = "/" + full
        keys.append(full)

    grp.visititems(visitor)
    return sorted(set(keys))


def normalize_demo_in_paths(paths: List[str], sample_demo: Optional[str]) -> List[str]:
    """Replace '/data/<sample_demo>' with '/data/demo_*' for stable JSON."""
    if not sample_demo:
        return paths
    needle = f"/data/{sample_demo}"
    out = []
    for p in paths:
        if p.startswith(needle):
            out.append(p.replace(needle, "/data/demo_*", 1))
        else:
            out.append(p)
    return out


def get_key_meta(f: h5py.File, abs_key: str) -> Dict[str, Any]:
    """Return metadata for a key: kind, shape, dtype."""
    if abs_key not in f:
        return {"kind": "missing", "shape": None, "dtype": None}

    obj = f[abs_key]
    if isinstance(obj, h5py.Dataset):
        return {"kind": "dataset", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    return {"kind": "group", "shape": None, "dtype": None}


def read_env_args(f: h5py.File) -> Optional[Dict[str, Any]]:
    """Safely read data.attrs['env_args'] JSON string."""
    if "data" not in f:
        return None
    if "env_args" not in f["data"].attrs:
        return None
    try:
        return json.loads(f["data"].attrs["env_args"])
    except Exception:
        return None


def extract_env_robot_gripper(env_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract env/task/robot/gripper/controller/camera info from env_args."""
    if env_args is None:
        return {
            "env_name": None,
            "env_version": None,
            "env_type": None,
            "robots": None,
            "gripper_types": None,
            "controller_type": None,
            "control_freq": None,
            "camera_names": None,
            "camera_heights": None,
            "camera_widths": None,
        }

    env_kwargs = env_args.get("env_kwargs", {}) if isinstance(env_args, dict) else {}
    controller = env_kwargs.get("controller_configs", {}) if isinstance(env_kwargs, dict) else {}

    return {
        "env_name": env_args.get("env_name"),
        "env_version": env_args.get("env_version"),
        "env_type": env_args.get("type"),
        "robots": env_kwargs.get("robots"),
        "gripper_types": env_kwargs.get("gripper_types"),
        "controller_type": controller.get("type"),
        "control_freq": env_kwargs.get("control_freq"),
        "camera_names": env_kwargs.get("camera_names"),
        "camera_heights": env_kwargs.get("camera_heights"),
        "camera_widths": env_kwargs.get("camera_widths"),
    }


def infer_gripper_info(f: h5py.File, demo_path: str) -> Dict[str, Any]:
    """
    Gripper info priority:
      1) env_args.env_kwargs.gripper_types
      2) obs gripper key shape (fallback)
    """
    env_args = read_env_args(f)
    env_info = extract_env_robot_gripper(env_args)
    gripper_types = env_info.get("gripper_types")

    if isinstance(gripper_types, list) and len(gripper_types) > 0:
        return {
            "types": gripper_types,
            "num_grippers": len(gripper_types),
            "source": "env_args",
            "obs_key": None,
            "obs_shape": None,
            "dof": None,
        }

    obs_path = f"{demo_path}/obs"
    if obs_path not in f:
        return {
            "types": None,
            "num_grippers": 0,
            "source": "none",
            "obs_key": None,
            "obs_shape": None,
            "dof": None,
        }

    obs = f[obs_path]
    candidates = ["robot0_gripper_qpos", "robot0_gripper_pos", "gripper_qpos", "gripper_pos"]

    chosen = None
    for k in candidates:
        if k in obs and isinstance(obs[k], h5py.Dataset):
            chosen = k
            break

    if chosen is None:
        for k in obs.keys():
            if "gripper" in k and isinstance(obs[k], h5py.Dataset):
                chosen = k
                break

    if chosen is None:
        return {
            "types": None,
            "num_grippers": 0,
            "source": "unknown",
            "obs_key": None,
            "obs_shape": None,
            "dof": None,
        }

    ds = obs[chosen]
    shape = list(ds.shape)
    dof = 1 if len(shape) == 1 else (shape[1] if len(shape) >= 2 else None)

    return {
        "types": None,
        "num_grippers": 1,
        "source": "obs_shape",
        "obs_key": f"/{obs_path}/{chosen}" if not obs_path.startswith("/") else f"{obs_path}/{chosen}",
        "obs_shape": shape,
        "dof": dof,
    }


def _as_str_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def build_most_frequent_keys_and_diffs(file_to_keys: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Build JSON:
      {
        "most_frequent_keys": [...],   # keys with maximum frequency across files
        "<file1.hdf5>": {"missing_from_common": [...], "extra_vs_common": [...], ...},
        "<file2.hdf5>": ...
      }
    Notes:
      - Uses only the sampled & normalized key list per file.
      - "most_frequent_keys" is defined as keys whose occurrence count equals the global max count.
    """
    # frequency
    freq: Dict[str, int] = {}
    for _, keys in file_to_keys.items():
        for k in set(keys):
            freq[k] = freq.get(k, 0) + 1

    if not freq:
        # no keys available at all
        out: Dict[str, Any] = {"most_frequent_keys": []}
        for fn in sorted(file_to_keys.keys()):
            out[fn] = {
                "missing_from_common": [],
                "extra_vs_common": sorted(set(file_to_keys.get(fn, []))),
                "num_missing": 0,
                "num_extra": len(set(file_to_keys.get(fn, []))),
            }
        return out

    max_count = max(freq.values())
    common_keys = sorted([k for k, c in freq.items() if c == max_count])

    common_set = set(common_keys)
    out: Dict[str, Any] = {"most_frequent_keys": common_keys}

    for fn in sorted(file_to_keys.keys()):
        kset = set(file_to_keys.get(fn, []))
        missing = sorted(list(common_set - kset))
        extra = sorted(list(kset - common_set))
        out[fn] = {
            "missing_from_common": missing,
            "extra_vs_common": extra,
            "num_missing": len(missing),
            "num_extra": len(extra),
        }

    # (可选) 附加一些统计信息，方便你看 common_keys 的“强度”
    out["_meta"] = {
        "definition": "most_frequent_keys are keys with maximum occurrence count across HDF5 files",
        "max_count": max_count,
        "num_files_considered": len(file_to_keys),
        "num_unique_keys_total": len(freq),
    }
    return out


def build_index_and_stats_and_common(
    root_dir: str,
    keys_scope: str = "demo",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      index_json: per-file details
      stats_json: groupings by robot/gripper
      common_keys_json: most_frequent_keys + per-file diffs vs that set
    """
    datasets = []
    num_errors = 0
    all_files = sorted(iter_hdf5_files(root_dir))

    # stats containers
    by_robot: Dict[str, List[str]] = {}
    by_gripper: Dict[str, List[str]] = {}
    by_robot_and_gripper: Dict[str, List[str]] = {}

    # common keys containers (file basename -> normalized keys list)
    file_to_keys: Dict[str, List[str]] = {}

    def add_map(m: Dict[str, List[str]], k: str, fname: str):
        m.setdefault(k, []).append(fname)

    for abs_path in all_files:
        rel_path = os.path.relpath(abs_path, root_dir)
        fname_only = os.path.basename(abs_path)

        item: Dict[str, Any] = {
            "rel_path": rel_path,
            "abs_path": abs_path,
            "file": fname_only,
        }

        try:
            with h5py.File(abs_path, "r") as f:
                num_demos, sample_demo = get_demos_and_sample(f)
                item["num_demos"] = int(num_demos)
                item["sample_demo"] = sample_demo
                item["top_level_keys"] = list_top_level_keys(f)

                env_args = read_env_args(f)
                env_info = extract_env_robot_gripper(env_args)
                item["env"] = env_info

                # unified gripper info (prefers env_args)
                if sample_demo is not None:
                    demo_group = f"data/{sample_demo}"
                    gripper_info = infer_gripper_info(f, demo_group)
                else:
                    gripper_info = {
                        "types": None,
                        "num_grippers": 0,
                        "source": "none",
                        "obs_key": None,
                        "obs_shape": None,
                        "dof": None,
                    }
                item["gripper"] = gripper_info

                # ---- robot/gripper stats ----
                robot_types = _as_str_list(env_info.get("robots")) or ["__unknown__"]
                gripper_types = (
                    _as_str_list(env_info.get("gripper_types"))
                    or _as_str_list(gripper_info.get("types"))
                    or ["__unknown__"]
                )

                for r in robot_types:
                    add_map(by_robot, r, fname_only)
                for g in gripper_types:
                    add_map(by_gripper, g, fname_only)
                for r in robot_types:
                    for g in gripper_types:
                        add_map(by_robot_and_gripper, f"{r}|{g}", fname_only)

                # ---- keys + meta (only ONE demo) ----
                if sample_demo is not None:
                    if keys_scope == "obs":
                        group_path = f"data/{sample_demo}/obs"
                    else:
                        group_path = f"data/{sample_demo}"

                    keys_sample = list_keys_under_group(f, group_path)
                    keys_meta = {k: get_key_meta(f, k) for k in keys_sample}

                    keys_sample_norm = normalize_demo_in_paths(keys_sample, sample_demo)

                    needle = f"/data/{sample_demo}"
                    keys_meta_norm = {}
                    for k, meta in keys_meta.items():
                        kn = k.replace(needle, "/data/demo_*", 1) if k.startswith(needle) else k
                        keys_meta_norm[kn] = meta

                    item["keys_sample_demo"] = keys_sample_norm
                    item["keys_sample_demo_meta"] = keys_meta_norm
                    item["num_keys_sample_demo"] = len(keys_sample_norm)
                    item["keys_scope"] = keys_scope

                    # record for common-key analysis (basename -> keys)
                    file_to_keys[fname_only] = keys_sample_norm
                else:
                    item["keys_sample_demo"] = []
                    item["keys_sample_demo_meta"] = {}
                    item["num_keys_sample_demo"] = 0
                    item["keys_scope"] = keys_scope
                    file_to_keys[fname_only] = []

        except Exception as e:
            num_errors += 1
            item["error"] = f"{type(e).__name__}: {e}"
            item.setdefault("num_demos", 0)
            item.setdefault("sample_demo", None)
            item.setdefault("top_level_keys", [])
            item.setdefault("env", extract_env_robot_gripper(None))
            item.setdefault(
                "gripper",
                {
                    "types": None,
                    "num_grippers": 0,
                    "source": "none",
                    "obs_key": None,
                    "obs_shape": None,
                    "dof": None,
                },
            )
            item.setdefault("keys_sample_demo", [])
            item.setdefault("keys_sample_demo_meta", {})
            item.setdefault("num_keys_sample_demo", 0)
            item.setdefault("keys_scope", keys_scope)

            # still record in stats as unknown
            add_map(by_robot, "__unknown__", fname_only)
            add_map(by_gripper, "__unknown__", fname_only)
            add_map(by_robot_and_gripper, "__unknown__|__unknown__", fname_only)

            # for common keys
            file_to_keys[fname_only] = []

        datasets.append(item)

    index_json = {
        "root": os.path.abspath(root_dir),
        "datasets": datasets,
        "summary": {
            "num_files": len(datasets),
            "num_errors": num_errors,
        },
    }

    # sort lists for stable output
    for m in (by_robot, by_gripper, by_robot_and_gripper):
        for k in list(m.keys()):
            m[k] = sorted(set(m[k]))

    stats_json = {
        "by_robot": dict(sorted(by_robot.items(), key=lambda kv: kv[0])),
        "by_gripper": dict(sorted(by_gripper.items(), key=lambda kv: kv[0])),
        "by_robot_and_gripper": dict(sorted(by_robot_and_gripper.items(), key=lambda kv: kv[0])),
        "summary": {
            "num_files": len(datasets),
            "num_pairs": len(by_robot_and_gripper),
            "num_unique_robots": len(by_robot),
            "num_unique_grippers": len(by_gripper),
        },
    }

    # most frequent keys + per-file diffs
    common_keys_json = build_most_frequent_keys_and_diffs(file_to_keys)

    return index_json, stats_json, common_keys_json


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan HDF5 files; output (1) per-file index JSON, (2) robot/gripper stats JSON, "
            "(3) most-frequent-keys & per-file diffs JSON."
        )
    )
    parser.add_argument(
        "--root",
        default="/mnt/central_storage/data_pool_world/mimicgen_datasets",
        help="Root folder containing subfolders with HDF5 files",
    )
    parser.add_argument(
        "--out",
        default="mimicgen_hdf5_index_sample_demo_with_env_robot_gripper.json",
        help="Output JSON file path (per-file index)",
    )
    parser.add_argument(
        "--out-stats",
        default="mimicgen_robot_gripper_stats.json",
        help="Output JSON file path (robot/gripper -> [hdf5 filenames])",
    )
    parser.add_argument(
        "--out-common-keys",
        default="mimicgen_common_keys_and_diffs.json",
        help="Output JSON file path (most_frequent_keys + per-hdf5 diffs)",
    )
    parser.add_argument(
        "--keys-scope",
        choices=["demo", "obs"],
        default="demo",
        help="Which keys to collect for sample demo: 'demo' (all under /data/demo_x) or 'obs' (only obs)",
    )
    args = parser.parse_args()

    root_dir = args.root
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    index_json, stats_json, common_keys_json = build_index_and_stats_and_common(
        root_dir, keys_scope=args.keys_scope
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.out_stats) or ".", exist_ok=True)
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.out_common_keys) or ".", exist_ok=True)
    with open(args.out_common_keys, "w", encoding="utf-8") as f:
        json.dump(common_keys_json, f, ensure_ascii=False, indent=2)

    print(f"Wrote index JSON to: {args.out}")
    print(f"Wrote robot/gripper stats JSON to: {args.out_stats}")
    print(f"Wrote common keys & diffs JSON to: {args.out_common_keys}")
    print(f"Total HDF5 files: {index_json['summary']['num_files']}, Errors: {index_json['summary']['num_errors']}")


if __name__ == "__main__":
    main()
