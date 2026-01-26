import json
import numpy as np
from collections import defaultdict

# stats_path = "/raid/xiangyi.jia/data_processing/mimicgen/mimicgen_lerobot_0123/mimicgen_lerobot_debug_0124/coffee/meta/episodes_stats.jsonl" 
stats_path = "/raid/xiangyi.jia/data_processing/mimicgen/mimicgen_lerobot_0123/mimicgen_lerobot_debug_0124/general_7d_joint/meta/episodes_stats.jsonl"
# stats_path = "/raid/xiangyi.jia/data_processing/mimicgen/mimicgen_lerobot_0123/mimicgen_lerobot_debug_0124/three_assembly/meta/episodes_stats.jsonl"
# stats_path = "/raid/xiangyi.jia/data_processing/mimicgen/mimicgen_lerobot_0123/mimicgen_lerobot_debug_0124/hammer_kitchen/meta/episodes_stats.jsonl"
shapes = defaultdict(set)

with open(stats_path, "r") as f:
    for line in f:
        j = json.loads(line)
        stats = j["stats"]
        for k, v in stats.items():
            if "mean" not in v:
                continue
            mean = np.array(v["mean"])
            shapes[k].add(mean.shape)

bad = {k: list(v) for k, v in shapes.items() if len(v) > 1}
print("Inconsistent keys:", bad)
