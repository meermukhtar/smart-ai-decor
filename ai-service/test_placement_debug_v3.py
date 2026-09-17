import cv2
import numpy as np

from room_analysis.regions import find_free_regions
from room_analysis.scale import calculate_scale, meters_to_pixels


IMAGE_PATH = "room12.jpg"
MASK_PATH = "free_space_mask.png"

ROOM_WIDTH_METERS = 4.5

SOFA_WIDTH_M = 1.6
SOFA_DEPTH_M = 0.9


# =====================================================
# Load image
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]


# =====================================================
# Scale
# =====================================================

meters_per_pixel = calculate_scale(
    width,
    ROOM_WIDTH_METERS
)


sofa_width_px = meters_to_pixels(
    SOFA_WIDTH_M,
    meters_per_pixel
)

sofa_height_px = meters_to_pixels(
    SOFA_DEPTH_M,
    meters_per_pixel
)


print("Image:", width, "x", height)

print(
    "Sofa:",
    sofa_width_px,
    "x",
    sofa_height_px
)


# =====================================================
# Load mask
# =====================================================

mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(MASK_PATH)


free_mask = (
    mask > 0
).astype(np.uint8)


# =====================================================
# Find regions
# =====================================================

regions = find_free_regions(
    MASK_PATH
)


# =====================================================
# Debug every region
# =====================================================

for region in regions:

    bbox = region["bbox"]

    x = bbox["x"]
    y = bbox["y"]

    w = bbox["width"]
    h = bbox["height"]


    print("\n================================")
    print("REGION", region["id"])
    print("================================")

    print(
        "Bounding box:",
        x,
        y,
        w,
        h
    )

    print(
        "Free area:",
        region["area_pixels"]
    )


    # -------------------------------------------------
    # Extract actual region
    # -------------------------------------------------

    region_mask = free_mask[
        y:y + h,
        x:x + w
    ]


    actual_free = int(
        np.count_nonzero(region_mask)
    )


    bbox_area = (
        w * h
    )


    free_percentage = (
        actual_free
        /
        bbox_area
        *
        100
    )


    print(
        "Actual free pixels:",
        actual_free
    )

    print(
        "Bounding box pixels:",
        bbox_area
    )

    print(
        "Free percentage:",
        round(
            free_percentage,
            2
        ),
        "%"
    )


    # -------------------------------------------------
    # Test sofa with erosion
    # -------------------------------------------------

    kernel = np.ones(
        (
            sofa_height_px,
            sofa_width_px
        ),
        dtype=np.uint8
    )


    eroded = cv2.erode(
        region_mask,
        kernel,
        iterations=1
    )


    valid_count = int(
        np.count_nonzero(eroded)
    )


    print(
        "Valid sofa positions:",
        valid_count
    )


    # -------------------------------------------------
    # Try rotated sofa
    # -------------------------------------------------

    rotated_kernel = np.ones(
        (
            sofa_width_px,
            sofa_height_px
        ),
        dtype=np.uint8
    )


    rotated_eroded = cv2.erode(
        region_mask,
        rotated_kernel,
        iterations=1
    )


    rotated_count = int(
        np.count_nonzero(rotated_eroded)
    )


    print(
        "Valid rotated positions:",
        rotated_count
    )


    # -------------------------------------------------
    # Create debug image
    # -------------------------------------------------

    debug = image.copy()


    # Draw region bounding box

    cv2.rectangle(
        debug,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        3
    )


    # -------------------------------------------------
    # Draw valid positions
    # -------------------------------------------------

    ys, xs = np.where(
        eroded > 0
    )


    for px, py in zip(xs[::20], ys[::20]):

        cv2.circle(
            debug,
            (
                int(px + x),
                int(py + y)
            ),
            2,
            (0, 255, 0),
            -1
        )


    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    output_name = (
        f"placement_debug_region_{region['id']}.jpg"
    )


    cv2.imwrite(
        output_name,
        debug
    )


    print(
        "Saved:",
        output_name
    )