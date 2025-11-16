# app.py
from flask import Flask, render_template, Response
import cv2
import supervision as sv
from ultralytics import YOLO
import threading
import time
import numpy as np

app = Flask(__name__)

# -------------------------------------------------
# 1. Load model
# -------------------------------------------------
model = YOLO("best.pt")

# -------------------------------------------------
# 2. Video sources
# -------------------------------------------------
VIDEO_1 = "demo.mp4"
VIDEO_2 = "demo2.mp4"

# -------------------------------------------------
# 3. Annotators (bbox + centered label)
# -------------------------------------------------
# a nice palette
colors = sv.ColorPalette.from_hex(["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"])

bbox_annotator = sv.BoundingBoxAnnotator(thickness=3, color=colors)

# ---- custom centered-label annotator ----
class CenterLabelAnnotator:
    """Draws class label exactly in the middle of the bbox."""
    def __init__(self, text_scale=0.7, text_thickness=2, color=colors):
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.color = color

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        img = scene.copy()
        for (x1, y1, x2, y2), class_id, conf in zip(
            detections.xyxy.astype(int),
            detections.class_id,
            detections.confidence,
        ):
            label = f"{model.names[int(class_id)]} {conf:.2f}"
            # centre point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # measure text size
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            # background rectangle (slightly larger than text)
            pad = 4
            cv2.rectangle(
                img,
                (cx - tw // 2 - pad, cy - th // 2 - pad),
                (cx + tw // 2 + pad, cy + th // 2 + pad),
                self.color.by_idx(class_id).as_bgr(),
                -1,
            )
            # white text
            cv2.putText(
                img,
                label,
                (cx - tw // 2, cy + th // 2 - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                (255, 255, 255),
                self.text_thickness,
                cv2.LINE_AA,
            )
        return img

label_annotator = CenterLabelAnnotator(text_scale=0.7, text_thickness=2, color=colors)

# -------------------------------------------------
# 4. Frame storage (thread-safe)
# -------------------------------------------------
frame_lock = threading.Lock()
frame_cam1 = None
frame_cam2 = None


# -------------------------------------------------
# 5. Video processing worker
# -------------------------------------------------
def process_video(source: str, storage: str):
    global frame_cam1, frame_cam2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
            continue

        # ---- inference ----
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # ---- annotate ----
        annotated = bbox_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)

        # ---- timestamp ----
        cv2.putText(
            annotated,
            time.strftime("%H:%M:%S"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # ---- store frame ----
        with frame_lock:
            if storage == "cam1":
                frame_cam1 = annotated.copy()
            else:
                frame_cam2 = annotated.copy()


# -------------------------------------------------
# 6. MJPEG stream generator
# -------------------------------------------------
def generate_stream(storage: str):
    global frame_cam1, frame_cam2
    while True:
        with frame_lock:
            frame = frame_cam1 if storage == "cam1" else frame_cam2
            if frame is None:
                time.sleep(0.01)
                continue
            # compress to JPEG (quality 85 → good balance)
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# -------------------------------------------------
# 7. Flask routes
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/cam1")
def cam1():
    return Response(generate_stream("cam1"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cam2")
def cam2():
    return Response(generate_stream("cam2"), mimetype="multipart/x-mixed-replace; boundary=frame")


# -------------------------------------------------
# 8. Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # start two background workers
    threading.Thread(target=process_video, args=(VIDEO_1, "cam1"), daemon=True).start()
    threading.Thread(target=process_video, args=(VIDEO_2, "cam2"), daemon=True).start()

    print("Pyresearch Dashboard LIVE → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)