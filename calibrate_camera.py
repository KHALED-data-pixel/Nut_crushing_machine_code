import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calibrate(images_dir, config, output_path):
    cols   = config.get("chessboard_cols", 9)
    rows   = config.get("chessboard_rows", 6)
    sq_mm  = config.get("chessboard_square_mm", 10.0)

    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm

    obj_points, img_points = [], []
    image_size = None

    paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not paths:
        print(f"[ERROR] No images in: {images_dir}")
        print("        Run: python capture_samples.py --class calib --count 30")
        sys.exit(1)

    print(f"Chessboard: {cols}×{rows}  |  Square: {sq_mm} mm  |  Images: {len(paths)}\n")

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            print(f"  [FAIL] {os.path.basename(p)}")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners2)
        print(f"  [OK]   {os.path.basename(p)}")

    good = len(obj_points)
    print(f"\nUsable: {good}/{len(paths)}")
    if good < 3:
        print("[ERROR] Need at least 3 good images.")
        sys.exit(1)
    if good < 10:
        print("[WARN]  < 10 images — calibration may be inaccurate.")

    rms, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    print(f"\nRMS error : {rms:.4f} px  (target < 1.0)")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion: {dist.ravel()}")

    np.savez(output_path, mtx=mtx, dist=dist, rms=rms)
    print(f"\nSaved -> {os.path.abspath(output_path)}")

    # Show before/after on first image
    sample = cv2.imread(paths[0])
    if sample is not None:
        undist = cv2.undistort(sample, mtx, dist)
        side = np.hstack([sample, undist])
        cv2.putText(side, "Original",     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(side, "Undistorted", (image_size[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.namedWindow("Calibration Check", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Check", side)
        print("Press any key to close …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Camera calibration from chessboard images.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--images", default=None, help="Directory of chessboard images")
    p.add_argument("--output", default=None, help="Output .npz path")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)

    images_dir  = args.images or os.path.join(
        config.get("capture_output_dir", "dataset/images/raw"), "calib"
    )
    output_path = args.output or config.get("calib_file", "calib.npz")

    calibrate(images_dir, config, output_path)
