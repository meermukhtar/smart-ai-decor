from room_analysis.regions import find_free_regions
from room_analysis.scale import (
    calculate_scale,
    region_dimensions
)
from room_analysis.furniture import FURNITURE
from room_analysis.placement import check_bounding_box_fit

# =========================================================
# Image information
# =========================================================

IMAGE_WIDTH = 597

ROOM_WIDTH_METERS = 4.5


# =========================================================
# Calculate scale
# =========================================================

meters_per_pixel = calculate_scale(
    IMAGE_WIDTH,
    ROOM_WIDTH_METERS
)


# =========================================================
# Get free regions
# =========================================================

regions = find_free_regions(
    "free_space_mask.png"
)


if not regions:
    print("No usable free regions found.")
    exit()


# =========================================================
# Test living room
# =========================================================

room_type = "living_room"

furniture_list = FURNITURE[
    room_type
]


# =========================================================
# Analyze each region
# =========================================================

for region in regions:

    dimensions = region_dimensions(
        region,
        meters_per_pixel
    )

    region_width = dimensions[
        "width_m"
    ]

    region_height = dimensions[
        "height_m"
    ]

    print("\n================================")
    print(
        f"REGION {region['id']}"
    )
    print("================================")

    print(
        f"Available: "
        f"{region_width}m × "
        f"{region_height}m"
    )


    for furniture in furniture_list:

        result = check_bounding_box_fit(
        region_width,
        region_height,
        furniture["width"],
        furniture["depth"]
    )

        status = (
            "✓ FITS"
            if result["fits"]
            else "✗ DOES NOT FIT"
        )

        print(
            f"{furniture['name']}: "
            f"{status}"
        )