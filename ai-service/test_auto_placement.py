from room_analysis.scale import calculate_scale

from room_analysis.auto_placement import (
    analyze_furniture_placement
)


# =====================================================
# Configuration
# =====================================================

IMAGE_PATH = "room12.jpg"

FREE_MASK = "free_space_mask.png"


# =====================================================
# Image / scale
# =====================================================

import cv2


image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


image_height, image_width = (
    image.shape[:2]
)


# Current prototype assumption
ROOM_WIDTH_METERS = 4.5


meters_per_pixel = calculate_scale(
    image_width,
    ROOM_WIDTH_METERS
)


print(
    "Image:",
    image_width,
    "x",
    image_height
)

print(
    "Meters per pixel:",
    meters_per_pixel
)


# =====================================================
# Furniture
# =====================================================

furniture_name = "2_seater_sofa"

furniture_width_m = 1.6
furniture_depth_m = 0.9


# =====================================================
# Analyze
# =====================================================

results = analyze_furniture_placement(
    FREE_MASK,
    furniture_width_m,
    furniture_depth_m,
    meters_per_pixel
)


# =====================================================
# Display results
# =====================================================

print(
    "\n================================"
)

print(
    "AUTOMATIC FURNITURE ANALYSIS"
)

print(
    "================================"
)


print(
    "\nFurniture:",
    furniture_name
)

print(
    "Size:",
    furniture_width_m,
    "m x",
    furniture_depth_m,
    "m"
)


for result in results:

    region = result["region"]

    bbox = region["bbox"]

    fit = result["fit"]

    print(
        "\n--------------------------------"
    )

    print(
        "Region:",
        region["id"]
    )

    print(
        "Position:",
        bbox["x"],
        ",",
        bbox["y"]
    )

    print(
        "Region size:",
        bbox["width"],
        "x",
        bbox["height"]
    )

    print(
        "Normal fit:",
        fit["normal_fit"]
    )

    print(
        "Rotated fit:",
        fit["rotated_fit"]
    )

    print(
        "Overall fit:",
        fit["fits"]
    )