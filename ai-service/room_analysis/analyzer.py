import numpy as np


def analyze_room(depth):

    height, width = depth.shape

    # Take lower 40% of image
    # as our initial floor candidate.
    floor_region = depth[
        int(height * 0.60):,
        :
    ]

    return {
        "image_width": width,
        "image_height": height,
        "floor_candidate": {
            "x": 0,
            "y": int(height * 0.60),
            "width": width,
            "height": int(height * 0.40)
        }
    }