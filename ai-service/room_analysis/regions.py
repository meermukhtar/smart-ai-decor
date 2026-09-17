import cv2
import numpy as np


MIN_AREA = 3000


def find_free_regions(mask_path):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(mask_path)

    binary = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    # Clean small noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    regions = []

    for label in range(1, num_labels):

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < MIN_AREA:
            continue

        aspect_ratio = width / max(
            height,
            1
        )

        center_x, center_y = (
            centroids[label]
        )

        if aspect_ratio > 1.5:
            shape = "wide"

        elif aspect_ratio < 0.67:
            shape = "tall"

        else:
            shape = "balanced"

        regions.append({

            "id": int(label),

            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            },

            "area_pixels": int(area),

            "center": {
                "x": round(
                    float(center_x),
                    2
                ),
                "y": round(
                    float(center_y),
                    2
                )
            },

            "aspect_ratio": round(
                float(aspect_ratio),
                2
            ),

            "shape": shape
        })

    regions.sort(
        key=lambda r: r["area_pixels"],
        reverse=True
    )

    return regions