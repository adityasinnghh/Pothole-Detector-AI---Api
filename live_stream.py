from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO

router = APIRouter(prefix="/live", tags=["Live Detection"])

# Load model ONCE (very important for speed)
model = YOLO("best.pt")

@router.post("/frame")
async def detect_live_frame(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Invalid image")

    # Read bytes
    image_bytes = await file.read()

    # Decode image
    img_np = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w, _ = frame.shape

    # YOLO inference
    results = model(
        frame,
        conf=0.4,
        imgsz=640,
        device=0 if model.device.type != "cpu" else "cpu",
        verbose=False
    )

    detections = []

    r = results[0]
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": round(confidence, 3),
                "class_id": cls,
                "label": model.names[cls]
            })

    return {
        "frame_width": w,
        "frame_height": h,
        "detections": detections
    }
