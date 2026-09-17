import cv2
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_PATH = "room12.jpg"

SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
YOLO_MODEL = "yolo11n-seg.pt"

FLOOR_CLASS_ID = 3

YOLO_CONFIDENCE = 0.25


# =========================================================
# 1. Load Floor Segmentation Model
# =========================================================

print("\nLoading floor segmentation model...")

processor = AutoImageProcessor.from_pretrained(
    SEG_MODEL
)

segformer = SegformerForSemanticSegmentation.from_pretrained(
    SEG_MODEL
)

print("Floor segmentation model loaded")


# =========================================================
# 2. Load YOLO Segmentation Model
# =========================================================

print("\nLoading YOLO segmentation model...")

yolo = YOLO(
    YOLO_MODEL
)

print("YOLO segmentation model loaded")


# =========================================================
# 3. Load Image
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_np = np.array(image)

height, width = image_np.shape[:2]

print(
    f"\nImage size: {width} x {height}"
)


# =========================================================
# 4. SEGFORMER → FLOOR MASK
# =========================================================

print("\nRunning floor segmentation...")

inputs = processor(
    images=image,
    return_tensors="pt"
)


with torch.no_grad():

    outputs = segformer(
        **inputs
    )


# ---------------------------------------------------------
# Resize segmentation output to original image size
# ---------------------------------------------------------

logits = outputs.logits

logits = torch.nn.functional.interpolate(
    logits,
    size=(height, width),
    mode="bilinear",
    align_corners=False
)


# ---------------------------------------------------------
# Get predicted class for every pixel
# ---------------------------------------------------------

prediction = logits.argmax(
    dim=1
)[0].cpu().numpy()


# ---------------------------------------------------------
# Extract floor
# ---------------------------------------------------------

floor_mask = (
    prediction == FLOOR_CLASS_ID
)


print(
    "Raw floor pixels:",
    int(floor_mask.sum())
)


# =========================================================
# 5. CLEAN FLOOR MASK
# =========================================================

floor_mask_uint8 = (
    floor_mask.astype(np.uint8)
    * 255
)


# Remove small noise
kernel = np.ones(
    (5, 5),
    np.uint8
)


floor_mask_uint8 = cv2.morphologyEx(
    floor_mask_uint8,
    cv2.MORPH_OPEN,
    kernel
)


# Fill small gaps
floor_mask_uint8 = cv2.morphologyEx(
    floor_mask_uint8,
    cv2.MORPH_CLOSE,
    kernel
)


floor_mask = (
    floor_mask_uint8 > 0
)


print(
    "Clean floor pixels:",
    int(floor_mask.sum())
)


# =========================================================
# 6. SAVE FLOOR MASK
# =========================================================

cv2.imwrite(
    "floor_mask.png",
    floor_mask_uint8
)


print(
    "Saved: floor_mask.png"
)


# =========================================================
# 7. YOLO → FURNITURE MASKS
# =========================================================

print("\nRunning YOLO furniture segmentation...")


results = yolo(
    IMAGE_PATH,
    conf=YOLO_CONFIDENCE
)


occupied_mask = np.zeros(
    (height, width),
    dtype=bool
)


detected_objects = []


for result in results:

    if result.masks is None:

        print(
            "No segmentation masks detected."
        )

        continue


    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )


    for index, mask in enumerate(masks):

        # ---------------------------------------------
        # Resize mask to original image dimensions
        # ---------------------------------------------

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )


        mask = (
            mask > 0.5
        )


        # ---------------------------------------------
        # Object class
        # ---------------------------------------------

        class_id = int(
            result.boxes.cls[index]
        )


        confidence = float(
            result.boxes.conf[index]
        )


        class_name = result.names[
            class_id
        ]


        detected_objects.append({
            "name": class_name,
            "confidence": confidence
        })


        # ---------------------------------------------
        # Add object to global occupied mask
        # ---------------------------------------------

        occupied_mask |= mask


# =========================================================
# 8. PRINT DETECTED OBJECTS
# =========================================================

print(
    "\nDetected objects:"
)


if detected_objects:

    for obj in detected_objects:

        print(
            f"- {obj['name']} "
            f"({obj['confidence']:.3f})"
        )

else:

    print(
        "- None"
    )


print(
    "\nFurniture mask pixels:",
    int(occupied_mask.sum())
)


# =========================================================
# 9. FIND FURNITURE ACTUALLY ON FLOOR
# =========================================================

occupied_floor = (
    floor_mask
    & occupied_mask
)


print(
    "Occupied floor pixels:",
    int(occupied_floor.sum())
)


# =========================================================
# 10. FIND FREE FLOOR
# =========================================================

free_floor = (
    floor_mask
    & ~occupied_floor
)


print(
    "Free floor pixels:",
    int(free_floor.sum())
)


# =========================================================
# 11. SAVE FREE FLOOR MASK
# =========================================================

free_mask = (
    free_floor.astype(np.uint8)
    * 255
)


cv2.imwrite(
    "free_space_mask.png",
    free_mask
)


print(
    "Saved: free_space_mask.png"
)


# =========================================================
# 12. CREATE FLOOR OVERLAY
# =========================================================

original = cv2.imread(
    IMAGE_PATH
)


floor_overlay = (
    original.copy()
)


floor_overlay[
    floor_mask
] = (
    0,
    255,
    0
)


floor_result = cv2.addWeighted(
    original,
    0.65,
    floor_overlay,
    0.35,
    0
)


cv2.imwrite(
    "floor_overlay.jpg",
    floor_result
)


print(
    "Saved: floor_overlay.jpg"
)


# =========================================================
# 13. CREATE FREE SPACE OVERLAY
# =========================================================

free_overlay = (
    original.copy()
)


free_overlay[
    free_floor
] = (
    0,
    255,
    0
)


free_result = cv2.addWeighted(
    original,
    0.65,
    free_overlay,
    0.35,
    0
)


cv2.imwrite(
    "free_space_overlay.jpg",
    free_result
)


print(
    "Saved: free_space_overlay.jpg"
)


# =========================================================
# 14. CREATE OCCUPIED FLOOR OVERLAY
# =========================================================

occupied_overlay = (
    original.copy()
)


occupied_overlay[
    occupied_floor
] = (
    0,
    0,
    255
)


occupied_result = cv2.addWeighted(
    original,
    0.65,
    occupied_overlay,
    0.35,
    0
)


cv2.imwrite(
    "occupied_floor_overlay.jpg",
    occupied_result
)


print(
    "Saved: occupied_floor_overlay.jpg"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print(
    "\n======================================"
)

print(
    "ROOM SPACE ANALYSIS"
)

print(
    "======================================"
)

print(
    f"Total pixels:          {width * height}"
)

print(
    f"Floor pixels:          {int(floor_mask.sum())}"
)

print(
    f"Furniture pixels:      {int(occupied_mask.sum())}"
)

print(
    f"Occupied floor:        {int(occupied_floor.sum())}"
)

print(
    f"Free floor pixels:     {int(free_floor.sum())}"
)

print(
    "======================================"
)

print(
    "\nAnalysis complete."
)