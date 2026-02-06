import os, re, glob
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================
# PATH AYARLARI (MP4)
# =========================
VIDEO_PATH = r"C:\Users\Burak\yolov5\testvideo.mp4"
LABEL_DIR  = r"C:\Users\Burak\yolov5\runs\detect\exp3\labels"
OUT_DIR    = r"C:\Users\Burak\yolov5\runs\track"
OUT_NAME   = "deepsort_exp3_conf050_mxcsine014.mp4"

CONF_THRES = 0.50  # senin iyi dediğin değer

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LABEL MAP: testvideo_137.txt -> 137
# =========================
frame_re = re.compile(r"_(\d+)\.txt$")
label_map = {}

for p in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    m = frame_re.search(os.path.basename(p))
    if m:
        label_map[int(m.group(1))] = p

print("LABEL COUNT:", len(label_map))

# =========================
# VIDEO OPEN
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
print("CAP OPENED:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("WIDTH:", W, "HEIGHT:", H)

if not cap.isOpened() or W == 0 or H == 0:
    print("\n❌ VIDEO OPEN FAILED")
    print("OpenCV bu mp4 dosyasını okuyamıyor.")
    cap.release()
    exit()

# =========================
# MP4 WRITER (H.264)
# =========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    os.path.join(OUT_DIR, OUT_NAME),
    fourcc,
    fps if fps > 0 else 25,
    (W, H)
)

if not out.isOpened():
    print("❌ VideoWriter açılamadı")
    cap.release()
    exit()

# =========================
# DEEPSORT
# =========================
tracker = DeepSort(
    max_age=50,
    n_init=2,
    max_iou_distance=0.5,
    max_cosine_distance=0.14,
    nn_budget=None
)

frame_idx = 0
written = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    txt_path = label_map.get(frame_idx) or label_map.get(frame_idx + 1)

    if txt_path:
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue

                x, y, bw, bh = map(float, p[1:5])
                conf = float(p[5]) if len(p) >= 6 else 1.0

                if conf < CONF_THRES:
                    continue

                x1 = (x - bw/2) * W
                y1 = (y - bh/2) * H
                x2 = (x + bw/2) * W
                y2 = (y + bh/2) * H

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {t.track_id}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    out.write(frame)
    written += 1
    frame_idx += 1

cap.release()
out.release()

print("✅ DONE")
print("Written frames:", written)
print("OUTPUT:", os.path.join(OUT_DIR, OUT_NAME))
