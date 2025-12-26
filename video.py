from ultralytics import YOLO

model = YOLO("best.pt")
results = model("test.jpg", conf=0.25, iou=0.45)

for r in results:
    for box in r.boxes:
        print(
            "xyxy =", box.xyxy.cpu().numpy(),
            "conf =", float(box.conf)
        )
