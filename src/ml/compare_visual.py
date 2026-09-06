import cv2
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE = "test.jpeg"

OLD_MODEL = "models/yolo26n-face.pt"
NEW_MODEL = "models/best.pt"

CONF = 0.5


# --------------------------------------------------
# Draw detections
# --------------------------------------------------

def draw_detections(image, model, title):
    result = model.predict(
        source=image,
        conf=CONF,
        verbose=False
    )[0]

    output = image.copy()

    for box, confidence in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"Face {float(confidence):.2f}"

        cv2.putText(
            output,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        output,
        f"{title} | Faces: {len(result.boxes)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return output


# --------------------------------------------------
# Main
# --------------------------------------------------

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(f"Could not read {IMAGE}")

old_model = YOLO(OLD_MODEL)
new_model = YOLO(NEW_MODEL)

old_result = draw_detections(
    image,
    old_model,
    "PREVIOUS MODEL"
)

new_result = draw_detections(
    image,
    new_model,
    "TRAINED MODEL"
)

# Side by side
comparison = cv2.hconcat([
    old_result,
    new_result
])

cv2.imwrite(
    "model_comparison.jpeg",
    comparison
)

print("Saved: model_comparison.jpeg")

cv2.imshow("Model Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
