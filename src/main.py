import cv2
import time


# Open the default webcam
camera = cv2.VideoCapture(0)

# Load the pretrained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


while True:

    start_time = time.time()

    # Capture one frame
    success, frame = camera.read()

    if not success:
        print("Failed to read frame from camera")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw a rectangle around every detected face
    for (x, y, width, height) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )
        
    end_time = time.time()
    
    fps = 1 / (end_time - start_time)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
camera.release()
cv2.destroyAllWindows()
