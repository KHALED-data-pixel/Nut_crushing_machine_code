import argparse
import json
import os
import shutil
import sys


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dataset_yaml(config):
    yaml_path   = os.path.join("dataset", "dataset.yaml")
    class_names = config.get("class_names", ["argan", "almond", "peanut"])
    if os.path.isfile(yaml_path):
        print(f"Using existing {yaml_path}")
        return yaml_path

    os.makedirs("dataset", exist_ok=True)
    content = (
        f"path: {os.path.abspath('dataset')}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {yaml_path}")
    return yaml_path


def check_dataset():
    ok = True
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join("dataset", sub, split)
            if not os.path.isdir(d) or not os.listdir(d):
                print(f"[ERROR] Empty or missing: {d}")
                ok = False
            else:
                print(f"  [OK] {d}  ({len(os.listdir(d))} files)")
    return ok


def train(args, config):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Run: pip install ultralytics")
        sys.exit(1)

    yaml_path = ensure_dataset_yaml(config)

    print("\n--- Dataset check ---")
    if not check_dataset():
        print("\nFix dataset structure before training.")
        sys.exit(1)

    print(f"\nTraining YOLOv8{args.model_size}  epochs={args.epochs}  "
          f"batch={args.batch}  imgsz={args.imgsz}  device={args.device}\n")

    model = YOLO(f"yolov8{args.model_size}.pt")
    model.train(
        data=yaml_path, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
        workers=4, exist_ok=True,
    )

    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.4f}   mAP50-95: {metrics.box.map:.4f}")

    # Reload best.pt explicitly before exporting — guarantees best weights, not last epoch
    best_pt = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    if not os.path.isfile(best_pt):
        print(f"[WARN] best.pt not found at {best_pt} — exporting from current model state.")
        best_pt = None

    export_model = YOLO(best_pt) if best_pt else model
    export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)

    src = os.path.join("runs", "detect", args.name, "weights", "best.onnx")
    dst = config.get("model_file", "best.onnx")
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Exported -> {os.path.abspath(dst)}")
    else:
        print(f"[WARN] Could not find {src} — locate best.onnx manually.")

    print("\nRun inference with:  python infer_nut.py")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on nut dataset and export ONNX.")
    p.add_argument("--config",      default="config.json")
    p.add_argument("--epochs",      type=int,  default=100)
    p.add_argument("--batch",       type=int,  default=16)
    p.add_argument("--imgsz",       type=int,  default=640)
    p.add_argument("--device",      default="0", help="'0' = GPU, 'cpu' = CPU")
    p.add_argument("--model-size",  default="n", choices=["n","s","m","l","x"])
    p.add_argument("--name",        default="nut_detector")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args, load_config(args.config))
