import cv2
import numpy as np


MASK = "free_space_mask.png"

mask = cv2.imread(
    MASK,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(MASK)


free = mask > 0


print("Image size:")
print(
    mask.shape[1],
    "x",
    mask.shape[0]
)

print(
    "Free pixels:",
    int(free.sum())
)


# =====================================================
# Find contours
# =====================================================

contours, _ = cv2.findContours(
    free.astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


print(
    "\nNumber of free regions:",
    len(contours)
)


for i, contour in enumerate(contours):

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(
        contour
    )

    print(
        f"\nRegion {i + 1}"
    )

    print(
        "Area:",
        area
    )

    print(
        "Bounding box:",
        w,
        "x",
        h
    )


# =====================================================
# Create visualization
# =====================================================

image = cv2.cvtColor(
    mask,
    cv2.COLOR_GRAY2BGR
)


for i, contour in enumerate(contours):

    x, y, w, h = cv2.boundingRect(
        contour
    )

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Region {i + 1}",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


cv2.imwrite(
    "placement_debug.jpg",
    image
)

print(
    "\nSaved: placement_debug.jpg"
)