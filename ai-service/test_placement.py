from room_analysis.scale import (
    calculate_scale,
    meters_to_pixels
)

from room_analysis.placement import (
    find_placement_points
)


# =====================================================
# IMAGE
# =====================================================

IMAGE_WIDTH = 597

FREE_MASK = "free_space_mask.png"


# =====================================================
# ROOM SCALE
# =====================================================

ROOM_WIDTH_METERS = 4.5

meters_per_pixel = calculate_scale(
    IMAGE_WIDTH,
    ROOM_WIDTH_METERS
)


print(
    "Meters per pixel:",
    meters_per_pixel
)


# =====================================================
# FURNITURE
# =====================================================

furniture_name = "2_seater_sofa"

furniture_width_m = 1.6
furniture_depth_m = 0.9


# =====================================================
# Convert furniture dimensions
# =====================================================

furniture_width_px = meters_to_pixels(
    furniture_width_m,
    meters_per_pixel
)

furniture_height_px = meters_to_pixels(
    furniture_depth_m,
    meters_per_pixel
)


print(
    "\nFurniture:",
    furniture_name
)

print(
    "Real dimensions:",
    furniture_width_m,
    "m x",
    furniture_depth_m,
    "m"
)

print(
    "Pixel dimensions:",
    furniture_width_px,
    "x",
    furniture_height_px
)


# =====================================================
# FIND PLACEMENTS
# =====================================================

points = find_placement_points(
    FREE_MASK,
    furniture_width_px,
    furniture_height_px
)


print(
    "\n=============================="
)

print(
    "PLACEMENT ANALYSIS"
)

print(
    "=============================="
)


print(
    "Valid placement points:",
    len(points)
)


# =====================================================
# SHOW FIRST FEW
# =====================================================

for point in points[:20]:

    print(
        "x:",
        point["x"],
        "y:",
        point["y"]
    )