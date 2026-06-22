import argparse
import json
import os
import sys
import time

import cv2


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def capture(class_name, count, config):
    cam_idx    = config.get("camera_index", 0)
    cam_w      = config.get("camera_width", 640)
    cam_h      = config.get("camera_height", 640)
    output_dir = os.path.join(config.get("capture_output_dir", "dataset/images/raw"), class_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_idx}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    existing = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    saved = 0

    print(f"Class: {class_name} | Target: {count} | Output: {os.path.abspath(output_dir)}")
    print("SPACE = save frame   Q / ESC = quit\n")

    win = f"Capture — {class_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cam_w, cam_h)

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display = frame.copy()
        cv2.putText(display, f"{class_name}  {saved}/{count}  (SPACE=save  Q=quit)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("Aborted.")
            break
        elif key == ord(" "):
            idx = existing + saved + 1
            path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"  [{saved}/{count}] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone — {saved} images saved.")


def parse_args():
    p = argparse.ArgumentParser(description="Capture nut / calibration images from a webcam.")
    p.add_argument("--class", dest="class_name", required=True,
                   help="argan | almond | peanut | calib")
    p.add_argument("--count",  type=int, default=200)
    p.add_argument("--config", default="config.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture(args.class_name, args.count, load_config(args.config))
