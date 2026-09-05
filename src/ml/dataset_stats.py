from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

for split in ["train", "val"]:

    label_dir = (
        PROJECT_ROOT
        / "data"
        / "face_dataset"
        / "labels"
        / split
    )

    total_images = 0
    total_faces = 0

    for label_file in label_dir.glob("*.txt"):

        total_images += 1

        with open(label_file, "r") as f:
            faces = [
                line
                for line in f
                if line.strip()
            ]

        total_faces += len(faces)

    print(f"{split}:")
    print(f"  images = {total_images}")
    print(f"  faces  = {total_faces}")
    print()
