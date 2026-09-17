# room_analysis/depth.py

import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME
)


def estimate_depth(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = depth.cpu().numpy()

    # Normalize to 0-1
    depth_min = depth.min()
    depth_max = depth.max()

    normalized = (
        depth - depth_min
    ) / (
        depth_max - depth_min
    )

    return normalized