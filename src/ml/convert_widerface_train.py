from pathlib import Path
from PIL import Image
import shutil


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WIDER_ROOT = PROJECT_ROOT / "data" / "widerface"
DATASET_ROOT = PROJECT_ROOT / "data" / "face_dataset"

IMAGE_ROOT = WIDER_ROOT / "WIDER_train" / "images"

ANNOTATION_FILE = (
    WIDER_ROOT
    / "wider_face_split"
    / "wider_face_train_bbx_gt.txt"
)

OUTPUT_IMAGES = DATASET_ROOT / "images" / "train"
OUTPUT_LABELS = DATASET_ROOT / "labels" / "train"


# ============================================================
# Create output directories
# ============================================================

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read annotation file
# ============================================================

with open(ANNOTATION_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]


# ============================================================
# Counters
# ============================================================

i = 0

image_count = 0
skipped_count = 0
malformed_count = 0
missing_count = 0
corrupt_count = 0


# ============================================================
# Helper: determine whether a line looks like an image path
# ============================================================

def looks_like_image_path(line):
    """
    WIDER FACE image paths look like:

        0--Parade/0_Parade_marchingband_1_5.jpg

    We use the .jpg extension as the main indicator.
    """

    return line.lower().endswith((".jpg", ".jpeg", ".png"))


# ============================================================
# Main parsing loop
# ============================================================

while i < len(lines):

    # --------------------------------------------------------
    # Find the next image path
    # --------------------------------------------------------

    if not looks_like_image_path(lines[i]):

        malformed_count += 1

        print(
            f"[WARNING] Unexpected line at index {i}: "
            f"{lines[i]}"
        )

        i += 1
        continue

    image_relative_path = lines[i]
    i += 1


    # --------------------------------------------------------
    # Read number of faces
    # --------------------------------------------------------

    if i >= len(lines):
        print(
            f"[WARNING] Missing face count for: "
            f"{image_relative_path}"
        )

        skipped_count += 1
        break


    try:

        num_faces = int(lines[i])

    except ValueError:

        print(
            f"[WARNING] Invalid face count for: "
            f"{image_relative_path}"
        )

        print(
            f"          Found: {lines[i]}"
        )

        skipped_count += 1

        # ----------------------------------------------------
        # Recovery:
        # The parser is out of sync.
        #
        # Search forward until we find the next image path.
        # ----------------------------------------------------

        while i < len(lines):

            if looks_like_image_path(lines[i]):
                break

            i += 1

        continue

    i += 1


    # --------------------------------------------------------
    # Construct image path
    # --------------------------------------------------------

    image_path = IMAGE_ROOT / image_relative_path


    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not image_path.exists():

        print(
            f"[WARNING] Image not found: "
            f"{image_relative_path}"
        )

        missing_count += 1
        skipped_count += 1

        # Skip its annotation lines
        i += num_faces

        continue


    # --------------------------------------------------------
    # Read image dimensions
    # --------------------------------------------------------

    try:

        with Image.open(image_path) as image:

            image_width, image_height = image.size

    except Exception as e:

        print(
            f"[WARNING] Could not open image: "
            f"{image_relative_path}"
        )

        print(f"          Error: {e}")

        corrupt_count += 1
        skipped_count += 1

        # Skip annotation lines
        i += num_faces

        continue


    # --------------------------------------------------------
    # Convert bounding boxes
    # --------------------------------------------------------

    yolo_labels = []

    box_error = False

    for _ in range(num_faces):

        # Safety check
        if i >= len(lines):

            print(
                f"[WARNING] Unexpected end of annotation "
                f"while processing {image_relative_path}"
            )

            box_error = True
            break


        values = lines[i].split()
        i += 1


        # ----------------------------------------------------
        # WIDER FACE bounding box normally contains:
        #
        # x y width height blur expression illumination
        # invalid occlusion pose
        #
        # We only need first four values.
        # ----------------------------------------------------

        if len(values) < 4:

            print(
                f"[WARNING] Malformed bounding box in "
                f"{image_relative_path}: {values}"
            )

            box_error = True
            continue


        try:

            x = float(values[0])
            y = float(values[1])
            width = float(values[2])
            height = float(values[3])

        except ValueError:

            print(
                f"[WARNING] Invalid bounding box in "
                f"{image_relative_path}: {values}"
            )

            box_error = True
            continue


        # ----------------------------------------------------
        # Ignore invalid boxes
        # ----------------------------------------------------

        if width <= 0 or height <= 0:

            continue


        # ----------------------------------------------------
        # Convert:
        #
        # x, y, width, height
        #
        # ->
        #
        # x_center, y_center, width, height
        # ----------------------------------------------------

        x_center = x + width / 2
        y_center = y + height / 2


        # ----------------------------------------------------
        # Normalize to 0-1
        # ----------------------------------------------------

        x_center /= image_width
        y_center /= image_height

        width /= image_width
        height /= image_height


        # ----------------------------------------------------
        # Clamp values to valid YOLO range
        # ----------------------------------------------------

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))

        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))


        # ----------------------------------------------------
        # Class 0 = face
        # ----------------------------------------------------

        yolo_labels.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )


    # --------------------------------------------------------
    # If something went seriously wrong with this block,
    # don't generate potentially incorrect labels.
    # --------------------------------------------------------

    if box_error and not yolo_labels:

        skipped_count += 1
        continue


    # --------------------------------------------------------
    # Flatten directory structure
    #
    # Example:
    #
    # 0--Parade/abc.jpg
    #
    # becomes:
    #
    # 0--Parade_abc.jpg
    # --------------------------------------------------------

    output_name = image_relative_path.replace("/", "_")


    output_image = OUTPUT_IMAGES / output_name

    output_label = OUTPUT_LABELS / (
        Path(output_name).stem + ".txt"
    )


    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    try:

        shutil.copy2(
            image_path,
            output_image
        )

    except Exception as e:

        print(
            f"[WARNING] Failed to copy "
            f"{image_relative_path}"
        )

        print(f"          Error: {e}")

        skipped_count += 1
        continue


    # --------------------------------------------------------
    # Write YOLO labels
    # --------------------------------------------------------

    try:

        with open(output_label, "w") as f:

            f.write(
                "\n".join(yolo_labels)
            )

    except Exception as e:

        print(
            f"[WARNING] Failed to write label "
            f"for {image_relative_path}"
        )

        print(f"          Error: {e}")

        # Remove image because its label couldn't be created
        if output_image.exists():
            output_image.unlink()

        skipped_count += 1
        continue


    # --------------------------------------------------------
    # Successfully processed
    # --------------------------------------------------------

    image_count += 1


    if image_count % 500 == 0:

        print(
            f"Processed {image_count} images "
            f"(skipped: {skipped_count})"
        )


# ============================================================
# Final report
# ============================================================

print()
print("=" * 60)
print("Conversion complete!")
print("=" * 60)

print(f"Images converted : {image_count}")
print(f"Images skipped   : {skipped_count}")
print(f"Missing images   : {missing_count}")
print(f"Corrupt images   : {corrupt_count}")
print(f"Malformed blocks : {malformed_count}")

print("=" * 60)

