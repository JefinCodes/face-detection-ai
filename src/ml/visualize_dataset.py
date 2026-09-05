from pathlib import Path
import random

import cv2


# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "face_dataset" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "data" / "face_dataset" / "labels" / "train"


# -------------------------
# Pick random images
# -------------------------

images = list(IMAGE_DIR.glob("*.jpg"))

if not images:
    print("No images found!")
    exit()

random.shuffle(images)

# Show 5 random images
images = images[:5]


# -------------------------
# Process images
# -------------------------

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                # Convert normalized YOLO coordinates
                # back into pixel coordinates

                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Draw bounding box

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "face",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Resize large images for display

    display_width = 1000

    if width > display_width:

        scale = display_width / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale
        )

    cv2.imshow("Dataset Verification", image)

    print("Showing:", image_path.name)
    print("Press any key for next image.")

    cv2.waitKey(0)


cv2.destroyAllWindows()
