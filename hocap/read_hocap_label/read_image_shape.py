import os
import cv2

BASE_DIR = "/mnt/central_storage/data_pool_world/HO-Cap/datasets/subject_1"
TARGET_IMG = "color_000000.jpg"


def main():
    for episode_id in sorted(os.listdir(BASE_DIR)):
        episode_dir = os.path.join(BASE_DIR, episode_id)
        if not os.path.isdir(episode_dir):
            continue

        print(f"\n=== Episode: {episode_id} ===")

        for camera_id in sorted(os.listdir(episode_dir)):
            camera_dir = os.path.join(episode_dir, camera_id)
            if not os.path.isdir(camera_dir):
                continue

            img_path = os.path.join(camera_dir, TARGET_IMG)

            if not os.path.isfile(img_path):
                print(f"[MISS] {camera_id}: {TARGET_IMG} not found")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[FAIL] {camera_id}: failed to read image")
                continue

            h, w, c = img.shape
            print(f"[OK]   {camera_id}: shape = ({h}, {w}, {c})")


if __name__ == "__main__":
    main()
