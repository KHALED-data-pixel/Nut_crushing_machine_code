import json
import os


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_structure(config):
    class_names = config.get("class_names", ["argan", "almond", "peanut"])
    dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/images/raw/argan",
        "dataset/images/raw/almond",
        "dataset/images/raw/peanut",
        "dataset/images/raw/calib",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  {d}")

    yaml_path = os.path.join("dataset", "dataset.yaml")
    content = (
        f"path:  {os.path.abspath('dataset').replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  {yaml_path}")

    print("\nClass IDs:")
    for i, name in enumerate(class_names):
        print(f"  {i} = {name}")

    print("\nNext steps:")
    print("  1. Capture: python capture_samples.py --class argan --count 200")
    print("  2. Annotate with labelImg")
    print("  3. Train:   python train_yolo.py")


if __name__ == "__main__":
    create_structure(load_config())
