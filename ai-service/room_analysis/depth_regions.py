import numpy as np


def analyze_region_depth(
    depth,
    region
):

    x = region["bbox"]["x"]
    y = region["bbox"]["y"]

    width = region["bbox"]["width"]
    height = region["bbox"]["height"]

    region_depth = depth[
        y:y + height,
        x:x + width
    ]

    if region_depth.size == 0:

        return {
            "min_depth": None,
            "max_depth": None,
            "mean_depth": None
        }

    return {

        "min_depth": float(
            np.min(region_depth)
        ),

        "max_depth": float(
            np.max(region_depth)
        ),

        "mean_depth": float(
            np.mean(region_depth)
        )
    }