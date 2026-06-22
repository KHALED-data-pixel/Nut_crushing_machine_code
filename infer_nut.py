import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from force_table import get_force

CLASS_COLORS = [
    (0,   200, 255),   # argan
    (50,  220,  80),   # almond
    (255,  80,  80),   # peanut
]


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_calib(path):
    if not os.path.isfile(path):
        print(f"[WARN] No calibration file: {path} — undistortion disabled.")
        return None, None
    d = np.load(path)
    return d["mtx"], d["dist"]


def load_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] Run: pip install onnxruntime-gpu")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}  Run: python train_yolo.py")
        sys.exit(1)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    print(f"ONNX provider: {providers[0]}")
    return ort.InferenceSession(model_path, providers=providers)


def open_serial(port, baud):
    try:
        import serial
        s = serial.Serial(port, baud, timeout=1)
        print(f"Serial: {port} @ {baud}")
        return s
    except Exception as e:
        print(f"[WARN] Serial unavailable: {e}")
        return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(output, conf_thr, iou_thr, w, h, num_cls):
    # output is already the raw ndarray (shape: 1, 5+num_cls, 8400)
    pred = output.T                             # (8400, 5+num_cls)
    cx = sigmoid(pred[:, 0]) * w
    cy = sigmoid(pred[:, 1]) * h
    bw = sigmoid(pred[:, 2]) * w
    bh = sigmoid(pred[:, 3]) * h
    obj = sigmoid(pred[:, 4])
    cls_logits = pred[:, 5: 5 + num_cls]
    cls_ids    = np.argmax(cls_logits, axis=1)
    cls_conf   = sigmoid(cls_logits[np.arange(len(cls_logits)), cls_ids])
    scores     = obj * cls_conf

    mask = scores >= conf_thr
    if not np.any(mask):
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    scores  = scores[mask]
    cls_ids = cls_ids[mask]

    x1 = np.clip((cx - bw / 2).astype(int), 0, w - 1)
    y1 = np.clip((cy - bh / 2).astype(int), 0, h - 1)
    x2 = np.clip((cx + bw / 2).astype(int), 0, w - 1)
    y2 = np.clip((cy + bh / 2).astype(int), 0, h - 1)

    boxes = list(zip(x1.tolist(), y1.tolist(),
                     (x2 - x1).tolist(), (y2 - y1).tolist()))
    idxs  = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf_thr, iou_thr)

    results = []
    for i in (idxs.flatten() if len(idxs) else []):
        results.append({
            "class_id":   int(cls_ids[i]),
            "confidence": float(scores[i]),
            "bbox":       [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
        })
    return results


def draw(frame, det, class_names, mm_per_px, force):
    x1, y1, x2, y2 = det["bbox"]
    cls   = class_names[det["class_id"]]
    conf  = det["confidence"]
    w_mm  = (x2 - x1) * mm_per_px
    color = CLASS_COLORS[det["class_id"] % len(CLASS_COLORS)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{cls} {conf:.2f}  {w_mm:.1f}mm  {int(force)}N"
    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - lh - bl - 4), (x1 + lw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def preprocess(frame, size):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), 0)


def run(args):
    cfg       = load_config(args.config)
    cam_idx   = cfg["camera_index"]
    cam_w     = cfg["camera_width"]
    cam_h     = cfg["camera_height"]
    mm_per_px = cfg["mm_per_px"]
    conf_thr  = cfg["confidence_threshold"]
    iou_thr   = cfg["iou_threshold"]
    cls_names = cfg.get("class_names", ["argan", "almond", "peanut"])

    mtx, dist = load_calib(cfg.get("calib_file", "calib.npz"))
    session   = load_session(cfg.get("model_file", "best.onnx"))
    inp_name  = session.get_inputs()[0].name

    ser = None
    if not args.no_serial and cfg.get("serial_enabled", True):
        ser = open_serial(cfg["serial_port"], cfg["serial_baud"])

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_idx} unavailable")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    display = not args.no_display
    if display:
        cv2.namedWindow("Nut Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Nut Vision", cam_w, cam_h)

    map1 = map2 = None
    if mtx is not None:
        map1, map2 = cv2.initUndistortRectifyMap(
            mtx, dist, None, mtx, (cam_w, cam_h), cv2.CV_16SC2)

    fps_t, fps_n, fps = time.time(), 0, 0.0
    print("Running — press Q or ESC to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            ud = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) if map1 is not None else frame.copy()

            raw_output = session.run(None, {inp_name: preprocess(ud, cam_w)})
            detections = decode(
                raw_output[0],          # shape: (1, 5+num_cls, 8400)
                conf_thr, iou_thr, cam_w, cam_h, len(cls_names)
            )

            for det in detections:
                cls   = cls_names[det["class_id"]]
                w_mm  = (det["bbox"][2] - det["bbox"][0]) * mm_per_px
                force = get_force(cls, w_mm)
                print(f"  {cls:<10} conf={det['confidence']:.2f}  {w_mm:.1f}mm  {int(force)}N")
                if ser:
                    try:
                        ser.write(f"FORCE={int(force)}\n".encode())
                    except Exception as e:
                        print(f"  [WARN] Serial: {e}")
                if display:
                    draw(ud, det, cls_names, mm_per_px, force)

            fps_n += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps   = fps_n / elapsed
                fps_n = 0
                fps_t = time.time()

            if display:
                cv2.putText(ud, f"FPS {fps:.1f}", (cam_w - 110, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow("Nut Vision", ud)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # parentheses required
                    break
    finally:
        cap.release()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Live nut detection -> force command.")
    p.add_argument("--config",      default="config.json")
    p.add_argument("--no-serial",   action="store_true")
    p.add_argument("--no-display",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
