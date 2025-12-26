import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "best.pt"     # Your trained pothole model
IMAGE_PATH = "test.jpg"   # Image test (optional)
VIDEO_PATH = "test.mp4"   # Video test (optional)
CONF_THRESHOLD = 0.4      # Confidence threshold

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# FUNCTION: DRAW BOXES
# =========================
def draw_boxes(frame, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return frame

# =========================
# IMAGE TEST
# =========================
def test_image():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ Image not found")
        return

    results = model(img)
    img = draw_boxes(img, results)

    cv2.imshow("Pothole Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# VIDEO TEST
# =========================
def test_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Video not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        frame = draw_boxes(frame, results)

        cv2.imshow("Pothole Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# LIVE CAMERA TEST
# =========================
def test_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("🎥 Press ESC to exit camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)
        frame = draw_boxes(frame, results)

        cv2.imshow("Pothole Detection - Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("""
Choose Test Mode:
1 - Image
2 - Video
3 - Live Camera
    """)

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        test_image()
    elif choice == "2":
        test_video()
    elif choice == "3":
        test_camera()
    else:
        print("❌ Invalid choice")
