import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

MASK_PATH = "free_space_mask.png"

MIN_AREA = 3000


# =========================================================
# Load free-space mask
# =========================================================

mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        f"Could not load {MASK_PATH}"
    )


# =========================================================
# Binary mask
# =========================================================

binary = np.where(
    mask > 0,
    255,
    0
).astype(np.uint8)


# =========================================================
# Clean mask
# =========================================================

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


# =========================================================
# Find connected regions
# =========================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )
)


print("\n======================================")
print("FREE SPACE REGIONS")
print("======================================")


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

    center_x, center_y = (
        centroids[label]
    )


    # Ignore tiny noise
    if area < MIN_AREA:
        continue


    region = {

        "id": int(label),

        "x": int(x),
        "y": int(y),

        "width": int(width),
        "height": int(height),

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
        }
    }


    regions.append(
        region
    )


# =========================================================
# Sort largest → smallest
# =========================================================

regions.sort(
    key=lambda r: r["area_pixels"],
    reverse=True
)


# =========================================================
# Print regions
# =========================================================

for region in regions:

    print(
        f"\nRegion {region['id']}"
    )

    print(
        f"  Position: "
        f"({region['x']}, {region['y']})"
    )

    print(
        f"  Size: "
        f"{region['width']} x "
        f"{region['height']}"
    )

    print(
        f"  Area: "
        f"{region['area_pixels']} pixels"
    )

    print(
        f"  Center: "
        f"{region['center']}"
    )


print(
    "\nTotal usable regions:",
    len(regions)
)


# =========================================================
# Create visualization
# =========================================================

original = cv2.imread(
    room12.jpg
)

output = original.copy()


for index, region in enumerate(
    regions,
    start=1
):

    x = region["x"]
    y = region["y"]

    w = region["width"]
    h = region["height"]


    # Draw bounding box

    cv2.rectangle(
        output,

        (x, y),

        (x + w, y + h),

        (0, 255, 0),

        2
    )


    # Center

    cx = int(
        region["center"]["x"]
    )

    cy = int(
        region["center"]["y"]
    )


    cv2.circle(
        output,

        (cx, cy),

        5,

        (0, 0, 255),

        -1
    )


    # Label

    cv2.putText(
        output,

        f"Region {index}",

        (x, max(y - 10, 20)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 255, 0),

        2
    )


# =========================================================
# Save
# =========================================================

cv2.imwrite(
    "free_regions.jpg",
    output
)


print(
    "\nSaved: free_regions.jpg"
)

print(
    "\n======================================"
)
print("DONE")
print("======================================")