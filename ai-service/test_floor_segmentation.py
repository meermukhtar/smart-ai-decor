import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation
)


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


# ---------------------------------------
# Load model
# ---------------------------------------

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

print("Model loaded")


# ---------------------------------------
# Load image
# ---------------------------------------

image_path = "room12.jpg"

image = Image.open(
    image_path
).convert("RGB")


# ---------------------------------------
# Prepare image
# ---------------------------------------

inputs = processor(
    images=image,
    return_tensors="pt"
)


# ---------------------------------------
# Run segmentation
# ---------------------------------------

with torch.no_grad():

    outputs = model(**inputs)


# ---------------------------------------
# Convert predictions
# ---------------------------------------

logits = outputs.logits

# Resize segmentation to original image size

logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False
)

prediction = logits.argmax(
    dim=1
)[0]


prediction = prediction.cpu().numpy()


# ---------------------------------------
# Find FLOOR class
# ---------------------------------------

floor_class_id = None

for class_id, label in model.config.id2label.items():

    if label.lower() == "floor":

        floor_class_id = int(class_id)

        break


print(
    "Floor class ID:",
    floor_class_id
)


if floor_class_id is None:

    raise RuntimeError(
        "Floor class was not found"
    )


# ---------------------------------------
# Create floor mask
# ---------------------------------------

floor_mask = (
    prediction == floor_class_id
)


print(
    "Floor pixels:",
    floor_mask.sum()
)


# ---------------------------------------
# Save binary mask
# ---------------------------------------

mask = (
    floor_mask.astype(np.uint8)
    * 255
)


cv2.imwrite(
    "floor_mask.png",
    mask
)


print(
    "Saved: floor_mask.png"
)


# ---------------------------------------
# Create visualization
# ---------------------------------------

original = cv2.imread(
    image_path
)

overlay = original.copy()

# Highlight floor

overlay[floor_mask] = (
    0,
    255,
    0
)


# Blend original + floor

result = cv2.addWeighted(
    original,
    0.65,
    overlay,
    0.35,
    0
)


cv2.imwrite(
    "floor_overlay.jpg",
    result
)


print(
    "Saved: floor_overlay.jpg"
)