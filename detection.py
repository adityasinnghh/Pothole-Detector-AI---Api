from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import traceback

router = APIRouter(prefix="/detect", tags=["Detection"])

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"           # make sure this path is correct
UPLOAD_DIR = "uploads"
SAMPLE_RATE = 10                 # every Nth frame for video
CONFIDENCE = 0.4

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model ONCE
model = YOLO(MODEL_PATH)

# ======================================================
# IMAGE DETECTION
# ======================================================
@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("cv2.imdecode failed")

        results = model(
            img,
            conf=CONFIDENCE,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        detections = _parse_yolo(results)

        return {
            "type": "image",
            "width": img.shape[1],
            "height": img.shape[0],
            "detections": detections
        }

    except Exception as e:
        traceback.print_exc()
        return _error_response(e)


# ======================================================
# VIDEO DETECTION
# ======================================================


from fastapi import UploadFile, File, Request
import cv2
from ultralytics import YOLO

@router.post("/video")
async def detect_video(
    request: Request,
    file: UploadFile = File(...)
):
    video_path = f"uploads/{file.filename}"
    out_path = f"outputs/pothole_{file.filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    model = YOLO("best.pt")
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()

    # 🔥 DYNAMIC URL (THIS IS THE KEY FIX)
    video_url = str(request.base_url) + out_path

    return {
        "video_url": video_url
    }

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def _parse_yolo(results):
    detections = []
    r = results[0]

    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(box.conf[0]),
                "class_id": cls,
                "label": model.names.get(cls, str(cls))
            })
    return detections


def _error_response(e: Exception):
    return {
        "error": str(e),
        "detections": []
    }
