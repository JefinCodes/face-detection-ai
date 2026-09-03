import cv2
from ultralytics import YOLO
import time


CONFIDENCE_THRESHOLD = 0.5


# -------------------------
# Load face detection model
# -------------------------

model = YOLO("models/yolo26n-face.pt")


# -------------------------
# Open webcam
# -------------------------

cap = cv2.VideoCapture(0)


prev_time = time.time()

while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break


    # -------------------------
    # Run face detection
    # -------------------------

    results = model(frame)

    result = results[0]

    boxes = result.boxes


    # -------------------------
    # Process each face
    # -------------------------

    for i in range(len(boxes)):

        x1, y1, x2, y2 = boxes.xyxy[i]

        confidence = boxes.conf[i]
        
        if confidence < CONFIDENCE_THRESHOLD:
            continue


        # Convert tensor values to integers
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Display confidence
        text = f"Face {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # -------------------------
    # Calculate FPS
    # -------------------------

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # -------------------------
    # Display FPS
    # -------------------------
    
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -------------------------
    # Display
    # -------------------------

    cv2.imshow("Face Detection", frame)


    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()
