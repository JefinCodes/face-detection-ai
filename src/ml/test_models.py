import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

OLD_MODEL_PATH = "models/yolo26n-face.pt"
NEW_MODEL_PATH = "models/best.pt"

AFW_DIR = Path("data/afw/extracted/testimages")

NUM_IMAGES = 100

CONF = 0.50
IOU_THRESHOLD = 0.50
IMG_SIZE = 640

RANDOM_SEED = 42


# ============================================================
# READ MATLAB STRING
# ============================================================

def read_mat_string(f, ref):

    data = np.asarray(
        f[ref][()]
    ).flatten()

    return "".join(
        chr(int(x))
        for x in data
    )


# ============================================================
# READ AFW ANNOTATIONS
# ============================================================

def read_afw_annotations():

    mat_path = AFW_DIR / "anno.mat"

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Could not find {mat_path}"
        )

    entries = []

    with h5py.File(mat_path, "r") as f:

        anno = f["anno"]

        print(
            f"AFW annotation matrix: {anno.shape}"
        )

        # anno = 4 x 205
        #
        # row 0 = filename
        # row 1 = bounding boxes
        # row 2 = pose information
        # row 3 = landmarks

        for i in range(anno.shape[1]):

            # ------------------------------------------------
            # IMAGE NAME
            # ------------------------------------------------

            filename_ref = anno[0, i]

            filename = read_mat_string(
                f,
                filename_ref,
            )

            filename = Path(filename).name

            image_path = AFW_DIR / filename

            if not image_path.exists():
                continue

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            bbox_cell_ref = anno[1, i]

            bbox_cell = f[bbox_cell_ref]

            bbox_ref = bbox_cell[0, 0]

            bbox_data = np.asarray(
                f[bbox_ref][()],
                dtype=float,
            )

            boxes = []

            # AFW format:
            #
            # [[x1, x2],
            #  [y1, y2]]
            #
            # For multiple faces it can contain more columns.

            if bbox_data.ndim == 2:

                for j in range(bbox_data.shape[1]):

                    if bbox_data.shape[0] >= 2:

                        x1 = bbox_data[0, j]
                        y1 = bbox_data[1, j]

                        # In the AFW annotation,
                        # the second coordinate pair gives
                        # the bottom-right corner.
                        #
                        # For the observed structure:
                        # shape = (2, 2)
                        #
                        # [[196, 472],
                        #  [291, 555]]
                        #
                        # => [196,291,472,555]

                        if bbox_data.shape[1] == 2:

                            x2 = bbox_data[0, 1]
                            y2 = bbox_data[1, 1]

                            boxes = [[
                                float(bbox_data[0, 0]),
                                float(bbox_data[1, 0]),
                                float(x2),
                                float(y2),
                            ]]

                            break

            if not boxes:
                continue

            entries.append(
                {
                    "image": image_path,
                    "boxes": boxes,
                }
            )

    print(
        f"AFW images with annotations: {len(entries)}"
    )

    return entries


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1)
        *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(predictions, ground_truth):

    matched_gt = set()

    tp = 0
    fp = 0

    matched_ious = []

    # Highest confidence first
    predictions = sorted(
        predictions,
        key=lambda x: x[1],
        reverse=True,
    )

    for pred_box, confidence in predictions:

        best_iou = 0.0
        best_gt = -1

        for i, gt_box in enumerate(ground_truth):

            if i in matched_gt:
                continue

            iou = calculate_iou(
                pred_box,
                gt_box,
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = i

        if best_iou >= IOU_THRESHOLD:

            tp += 1
            matched_gt.add(best_gt)

            matched_ious.append(
                best_iou
            )

        else:

            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn, matched_ious


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    test_entries,
    model_name,
):

    print()
    print("=" * 60)
    print(f"Evaluating: {model_name}")
    print("=" * 60)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_ious = []
    all_confidences = []

    total_time = 0.0

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    model.predict(
        str(test_entries[0]["image"]),
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False,
    )

    # --------------------------------------------------------
    # Test images
    # --------------------------------------------------------

    for index, entry in enumerate(
        test_entries,
        start=1,
    ):

        image_path = str(
            entry["image"]
        )

        ground_truth = entry["boxes"]

        start = time.perf_counter()

        results = model.predict(
            image_path,
            imgsz=IMG_SIZE,
            conf=CONF,
            verbose=False,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for box, confidence, cls in zip(
                boxes,
                confidences,
                classes,
            ):

                # Face class
                if int(cls) != 0:
                    continue

                predictions.append(
                    (
                        box.tolist(),
                        float(confidence),
                    )
                )

                all_confidences.append(
                    float(confidence)
                )

        tp, fp, fn, ious = match_predictions(
            predictions,
            ground_truth,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ious.extend(ious)

        if (
            index % 10 == 0
            or index == len(test_entries)
        ):

            print(
                f"Processed "
                f"{index}/{len(test_entries)}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        total_tp
        /
        (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    recall = (
        total_tp
        /
        (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    average_iou = (
        sum(all_ious)
        /
        len(all_ious)
        if all_ious
        else 0
    )

    average_confidence = (
        sum(all_confidences)
        /
        len(all_confidences)
        if all_confidences
        else 0
    )

    average_time = (
        total_time
        /
        len(test_entries)
    )

    fps = (
        1.0 / average_time
        if average_time > 0
        else 0
    )

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": average_iou,
        "confidence": average_confidence,
        "time": average_time,
        "fps": fps,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_comparison(
    old_stats,
    new_stats,
    num_images,
):

    print()
    print("=" * 75)
    print("FINAL FACE DETECTION MODEL COMPARISON")
    print("=" * 75)

    print(
        f"Dataset              : AFW"
    )

    print(
        f"Images tested        : {num_images}"
    )

    print(
        f"IoU threshold        : {IOU_THRESHOLD}"
    )

    print(
        f"Confidence threshold : {CONF}"
    )

    print()

    print(
        f"{'Metric':<25}"
        f"{'Previous Model':>20}"
        f"{'New Model':>20}"
    )

    print("-" * 75)

    print(
        f"{'True Positives':<25}"
        f"{old_stats['tp']:>20}"
        f"{new_stats['tp']:>20}"
    )

    print(
        f"{'False Positives':<25}"
        f"{old_stats['fp']:>20}"
        f"{new_stats['fp']:>20}"
    )

    print(
        f"{'False Negatives':<25}"
        f"{old_stats['fn']:>20}"
        f"{new_stats['fn']:>20}"
    )

    print(
        f"{'Precision':<25}"
        f"{old_stats['precision']:>20.4f}"
        f"{new_stats['precision']:>20.4f}"
    )

    print(
        f"{'Recall':<25}"
        f"{old_stats['recall']:>20.4f}"
        f"{new_stats['recall']:>20.4f}"
    )

    print(
        f"{'F1 Score':<25}"
        f"{old_stats['f1']:>20.4f}"
        f"{new_stats['f1']:>20.4f}"
    )

    print(
        f"{'Average IoU':<25}"
        f"{old_stats['iou']:>20.4f}"
        f"{new_stats['iou']:>20.4f}"
    )

    print(
        f"{'Average Confidence':<25}"
        f"{old_stats['confidence']:>20.4f}"
        f"{new_stats['confidence']:>20.4f}"
    )

    print(
        f"{'Inference Time (sec)':<25}"
        f"{old_stats['time']:>20.4f}"
        f"{new_stats['time']:>20.4f}"
    )

    print(
        f"{'FPS':<25}"
        f"{old_stats['fps']:>20.2f}"
        f"{new_stats['fps']:>20.2f}"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    print()
    print("IMPROVEMENT")
    print("-" * 75)

    print(
        f"Precision : "
        f"{new_stats['precision'] - old_stats['precision']:+.4f}"
    )

    print(
        f"Recall    : "
        f"{new_stats['recall'] - old_stats['recall']:+.4f}"
    )

    print(
        f"F1        : "
        f"{new_stats['f1'] - old_stats['f1']:+.4f}"
    )

    print(
        f"IoU       : "
        f"{new_stats['iou'] - old_stats['iou']:+.4f}"
    )

    print(
        f"FPS       : "
        f"{new_stats['fps'] - old_stats['fps']:+.2f}"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FACE DETECTION MODEL COMPARISON")
    print("=" * 75)

    # --------------------------------------------------------
    # Read AFW
    # --------------------------------------------------------

    print()
    print("Loading AFW annotations...")

    entries = read_afw_annotations()

    if not entries:

        raise RuntimeError(
            "Could not parse AFW annotations."
        )

    # --------------------------------------------------------
    # Select 100 random images
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    if len(entries) > NUM_IMAGES:

        test_entries = random.sample(
            entries,
            NUM_IMAGES,
        )

    else:

        test_entries = entries

    print()
    print(
        f"Testing on "
        f"{len(test_entries)} images."
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading previous model...")

    old_model = YOLO(
        OLD_MODEL_PATH
    )

    print("Loading new model...")

    new_model = YOLO(
        NEW_MODEL_PATH
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    old_stats = evaluate_model(
        old_model,
        test_entries,
        "Previous Model",
    )

    new_stats = evaluate_model(
        new_model,
        test_entries,
        "New Model",
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print_comparison(
        old_stats,
        new_stats,
        len(test_entries),
    )


if __name__ == "__main__":
    main()
