import os, re, glob
import cv2
import numpy as np

from trackers.sort.sort import Sort  # senin mevcut sort.py

VIDEO_PATH = r"testvideo.mp4"
LABEL_DIR  = r"runs\detect\exp3\labels"
OUT_DIR    = r"runs\track"
OUT_NAME   = "sort_exp3_conf045_FIXED.mp4"

os.makedirs(OUT_DIR, exist_ok=True)

# 1) label dosyalarını frame index -> dosya yolu şeklinde map’le
#    ör: testvideo_000123.txt -> 123
frame_re = re.compile(r"_(\d+)\.txt$")
label_map = {}

for p in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    base = os.path.basename(p)
    m = frame_re.search(base)
    if m:
        idx = int(m.group(1))
        label_map[idx] = p
    else:
        # Eğer isim formatın farklıysa buraya düşer
        # base'i print edip regexi ona göre güncelleriz
        pass

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(os.path.join(OUT_DIR, OUT_NAME),
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# 2) SORT parametreleri (stabil)
mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = []

    # 3) Bu frame için txt var mı?
    txt_path = label_map.get(frame_idx, None)
    if txt_path is not None:
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # YOLO format: cls x y w h [conf]
                cls = int(float(parts[0]))
                if cls != 0:
                    continue

                x, y, bw, bh = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else 1.0

                # Normalized xywh -> pixel xyxy
                x1 = (x - bw/2) * W
                y1 = (y - bh/2) * H
                x2 = (x + bw/2) * W
                y2 = (y + bh/2) * H

                # güvenlik: clamp
                x1 = max(0, min(W-1, x1))
                y1 = max(0, min(H-1, y1))
                x2 = max(0, min(W-1, x2))
                y2 = max(0, min(H-1, y2))

                dets.append([x1, y1, x2, y2, conf])

    dets = np.array(dets, dtype=np.float32) if len(dets) else np.empty((0, 5), dtype=np.float32)

    tracks = mot_tracker.update(dets)

    for x1, y1, x2, y2, tid in tracks:
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("DONE:", os.path.join(OUT_DIR, OUT_NAME))
print("Label files mapped:", len(label_map))
