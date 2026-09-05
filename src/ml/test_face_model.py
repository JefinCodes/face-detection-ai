from ultralytics import YOLO


# Load face detection model
model = YOLO("models/best.pt")


# Run inference
results = model("https://ultralytics.com/images/bus.jpg")


# Get first image result
result = results[0]


# Get detections
boxes = result.boxes


print("Number of faces:", len(boxes))

for i in range(len(boxes)):

    x1, y1, x2, y2 = boxes.xyxy[i]

    confidence = boxes.conf[i]

    print(
        f"Face {i + 1}: "
        f"confidence={confidence:.2f}, "
        f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
    )
