import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

IMAGE_PATH = room12.jpg

DEPTH_RAW_PATH = room12.jpg_depth.npy"

DEPTH_GRAY_PATH = room12.jpg_depth.png"

DEPTH_VIS_PATH = room12.jpg_depth_visual.png"


# =========================================================
# Load model
# =========================================================

print("Loading depth model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Depth model loaded")


# =========================================================
# Load image
# =========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_width, original_height = image.size

print(
    "Image size:",
    original_width,
    "x",
    original_height
)


# =========================================================
# Prepare input
# =========================================================

inputs = processor(
    images=image,
    return_tensors="pt"
)


# =========================================================
# Run depth estimation
# =========================================================

print("Running depth estimation...")

with torch.no_grad():

    outputs = model(**inputs)


# =========================================================
# Get predicted depth
# =========================================================

depth = outputs.predicted_depth


# =========================================================
# Resize to original image size
# =========================================================

depth = torch.nn.functional.interpolate(

    depth.unsqueeze(1),

    size=(
        original_height,
        original_width
    ),

    mode="bicubic",

    align_corners=False
).squeeze()


# =========================================================
# Convert to numpy
# =========================================================

depth = depth.cpu().numpy().astype(
    np.float32
)


# =========================================================
# Remove invalid values
# =========================================================

depth = np.nan_to_num(
    depth,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# =========================================================
# Get depth range
# =========================================================

depth_min = float(depth.min())

depth_max = float(depth.max())


print(
    "Raw depth min:",
    depth_min
)

print(
    "Raw depth max:",
    depth_max
)


# =========================================================
# Normalize depth
# =========================================================

if depth_max > depth_min:

    depth_normalized = (
        depth - depth_min
    ) / (
        depth_max - depth_min
    )

else:

    depth_normalized = np.zeros_like(
        depth
    )


# =========================================================
# Save RAW depth
# =========================================================

np.save(
    DEPTH_RAW_PATH,
    depth
)

print(
    "Saved:",
    DEPTH_RAW_PATH
)


# =========================================================
# Save GRAYSCALE depth
#
# IMPORTANT:
# This is the file reconstruction.py should use.
# =========================================================

depth_uint8 = (
    depth_normalized * 255
).clip(
    0,
    255
).astype(
    np.uint8
)


cv2.imwrite(
    DEPTH_GRAY_PATH,
    depth_uint8
)

print(
    "Saved:",
    DEPTH_GRAY_PATH
)


# =========================================================
# Save color visualization
#
# This is ONLY for humans to inspect.
# =========================================================

plt.imsave(
    DEPTH_VIS_PATH,
    depth_normalized,
    cmap="inferno"
)

print(
    "Saved:",
    DEPTH_VIS_PATH
)


# =========================================================
# Final information
# =========================================================

print()
print(
    "=============================="
)

print(
    "DEPTH COMPLETE"
)

print(
    "=============================="
)

print(
    "Raw depth:",
    DEPTH_RAW_PATH
)

print(
    "Processing depth:",
    DEPTH_GRAY_PATH
)

print(
    "Visualization:",
    DEPTH_VIS_PATH
)

print(
    "Depth shape:",
    depth.shape
)