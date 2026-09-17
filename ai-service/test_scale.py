from room_analysis.regions import (
    find_free_regions
)

from room_analysis.scale import (
    calculate_scale,
    region_dimensions
)


IMAGE_WIDTH = 1543

# Example:
# User says the room is approximately 4.5 meters wide.

ROOM_WIDTH_METERS = 4.5


# =========================================================
# Calculate scale
# =========================================================

meters_per_pixel = calculate_scale(
    IMAGE_WIDTH,
    ROOM_WIDTH_METERS
)


print(
    "\nMeters per pixel:",
    meters_per_pixel
)


# =========================================================
# Find regions
# =========================================================

regions = find_free_regions(
    "free_space_mask.png"
)


print(
    "\n=============================="
)

print(
    "REAL WORLD REGIONS"
)

print(
    "=============================="
)


# =========================================================
# Convert regions
# =========================================================

for region in regions:

    dimensions = region_dimensions(
        region,
        meters_per_pixel
    )

    print(
        f"\nRegion {region['id']}"
    )

    print(
        "Pixel dimensions:",
        region["bbox"]["width"],
        "x",
        region["bbox"]["height"]
    )

    print(
        "Approx dimensions:",
        dimensions["width_m"],
        "m x",
        dimensions["height_m"],
        "m"
    )