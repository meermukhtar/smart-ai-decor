import cv2

from room_analysis.scale import (
    calculate_scale,
    meters_to_pixels
)

from room_analysis.placement import (
    check_bounding_box_fit
)


# =====================================================
# Image
# =====================================================

IMAGE_PATH = "room12.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


image_height, image_width = image.shape[:2]


print(
    "Image size:",
    image_width,
    "x",
    image_height
)


# =====================================================
# Room scale
# =====================================================

# IMPORTANT:
# This is currently an assumed real-world width.
#
# Later we will replace this with depth/camera-based
# scale estimation.

ROOM_WIDTH_METERS = 4.5


meters_per_pixel = calculate_scale(
    image_width,
    ROOM_WIDTH_METERS
)


print(
    "Meters per pixel:",
    meters_per_pixel
)


# =====================================================
# Main free region
# =====================================================

# From test_regions_v2.py
region_width_px = 944
region_height_px = 461


# =====================================================
# Sofa
# =====================================================

sofa_width_m = 1.6
sofa_depth_m = 0.9


# Convert real-world dimensions to pixels

sofa_width_px = meters_to_pixels(
    sofa_width_m,
    meters_per_pixel
)

sofa_depth_px = meters_to_pixels(
    sofa_depth_m,
    meters_per_pixel
)


# =====================================================
# Print information
# =====================================================

print(
    "\n=============================="
)

print(
    "GEOMETRIC FIT"
)

print(
    "=============================="
)


print(
    "\nFree region:"
)

print(
    "Pixel dimensions:",
    region_width_px,
    "x",
    region_height_px
)


print(
    "Real dimensions:",
    round(
        region_width_px * meters_per_pixel,
        2
    ),
    "m x",
    round(
        region_height_px * meters_per_pixel,
        2
    ),
    "m"
)


print(
    "\nSofa:"
)

print(
    "Real dimensions:",
    sofa_width_m,
    "m x",
    sofa_depth_m,
    "m"
)


print(
    "Pixel dimensions:",
    sofa_width_px,
    "x",
    sofa_depth_px
)


# =====================================================
# Check fit
# =====================================================

result = check_bounding_box_fit(
    region_width_px,
    region_height_px,
    sofa_width_px,
    sofa_depth_px
)


# =====================================================
# Results
# =====================================================

print(
    "\nNormal orientation:",
    result["normal_fit"]
)

print(
    "Rotated orientation:",
    result["rotated_fit"]
)

print(
    "Overall fit:",
    result["fits"]
)


# =====================================================
# Final recommendation
# =====================================================

if result["fits"]:

    print(
        "\n✓ Sofa can theoretically fit "
        "inside this region."
    )

else:

    print(
        "\n✗ Sofa does not fit "
        "inside this region."
    )