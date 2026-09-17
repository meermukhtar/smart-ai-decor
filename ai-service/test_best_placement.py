import cv2

from room_analysis.regions import (
    find_free_regions
)

from room_analysis.scale import (
    calculate_scale,
    meters_to_pixels
)

from room_analysis.placement import (
    find_best_position
)


# =====================================================
# Configuration
# =====================================================

IMAGE_PATH = "room12.jpg"

MASK_PATH = "free_space_mask.png"

ROOM_WIDTH_METERS = 4.5

SOFA_WIDTH_M = 1.6

SOFA_DEPTH_M = 0.9


# =====================================================
# Load image
# =====================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]


print(
    "Image:",
    width,
    "x",
    height
)


# =====================================================
# Calculate scale
# =====================================================

meters_per_pixel = calculate_scale(
    width,
    ROOM_WIDTH_METERS
)

print(
    "Meters per pixel:",
    meters_per_pixel
)


# =====================================================
# Convert sofa dimensions
# =====================================================

sofa_width_px = meters_to_pixels(
    SOFA_WIDTH_M,
    meters_per_pixel
)

sofa_depth_px = meters_to_pixels(
    SOFA_DEPTH_M,
    meters_per_pixel
)


print(
    "Sofa:",
    SOFA_WIDTH_M,
    "m x",
    SOFA_DEPTH_M,
    "m"
)

print(
    "Sofa pixel size:",
    sofa_width_px,
    "x",
    sofa_depth_px
)


# =====================================================
# Load free-space mask
# =====================================================

free_mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if free_mask is None:
    raise FileNotFoundError(
        f"Could not load mask: {MASK_PATH}"
    )


free_mask = free_mask > 0


# =====================================================
# Detect free regions
# =====================================================

regions = find_free_regions(
    MASK_PATH
)


print("\n================================")
print("BEST FURNITURE PLACEMENT")
print("================================")


# =====================================================
# Prepare output image
# =====================================================

output = image.copy()


# =====================================================
# Find best placement in each region
# =====================================================

placements = []


for region in regions:

    bbox = region["bbox"]

    print(
        f"\nRegion {region['id']}"
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


    # -------------------------------------------------
    # Find actual placement
    # -------------------------------------------------

    placement = find_best_position(

        free_mask,

        region,

        sofa_width_px,

        sofa_depth_px,

        step=10
    )


    # -------------------------------------------------
    # No placement
    # -------------------------------------------------

    if placement is None:

        print(
            "Result: NO VALID PLACEMENT"
        )

        continue


    # -------------------------------------------------
    # Placement found
    # -------------------------------------------------

    print(
        "Result: VALID PLACEMENT"
    )

    print(
        "X:",
        placement["x"]
    )

    print(
        "Y:",
        placement["y"]
    )

    print(
        "Width:",
        placement["width"]
    )

    print(
        "Height:",
        placement["height"]
    )

    print(
        "Rotation:",
        placement["rotation"]
    )


    placements.append(
        placement
    )


    # -------------------------------------------------
    # Draw placement
    # -------------------------------------------------

    x = placement["x"]
    y = placement["y"]

    w = placement["width"]
    h = placement["height"]


    cv2.rectangle(

        output,

        (x, y),

        (x + w, y + h),

        (0, 255, 0),

        4
    )


    # -------------------------------------------------
    # Label
    # -------------------------------------------------

    label_y = max(
        y - 10,
        25
    )


    cv2.putText(

        output,

        "SOFA",

        (x, label_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )


# =====================================================
# Save preview
# =====================================================

cv2.imwrite(
    "placement_preview.jpg",
    output
)


print(
    "\nSaved: placement_preview.jpg"
)


# =====================================================
# Summary
# =====================================================

print("\n================================")
print("SUMMARY")
print("================================")

print(
    "Regions detected:",
    len(regions)
)

print(
    "Valid placements:",
    len(placements)
)