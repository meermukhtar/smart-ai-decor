import cv2

from room_analysis.detection import detect_furniture
from room_analysis.movement import find_move_position


# ============================================================
# IMAGE
# ============================================================

IMAGE_PATH = "room12.jpg"


# ============================================================
# MASKS
# ============================================================

FLOOR_MASK_PATH = "floor_mask.png"
FREE_SPACE_MASK_PATH = "free_space_mask.png"


# ============================================================
# DETECT FURNITURE
# ============================================================

detections = detect_furniture(
    IMAGE_PATH,
    confidence=0.5
)


print("\n" + "=" * 60)
print("FURNITURE DETECTION")
print("=" * 60)


for detection in detections:

    print(
        f"{detection['name']} "
        f"→ "
        f"{detection['catalog_name']}"
    )

    print(
        f"Confidence: "
        f"{detection['confidence']:.3f}"
    )

    print(
        f"BBox: "
        f"{detection['bbox']}"
    )

    if "mask" in detection:

        print(
            f"Mask shape: "
            f"{detection['mask'].shape}"
        )


# ============================================================
# FIND BED
# ============================================================

bed = None

for detection in detections:

    if detection["catalog_name"] == "double_bed":

        bed = detection
        break


if bed is None:

    raise RuntimeError(
        "double_bed was not detected."
    )


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise FileNotFoundError(
        IMAGE_PATH
    )


image_height, image_width = image.shape[:2]


print("\n" + "=" * 60)
print("IMAGE")
print("=" * 60)

print(
    f"Width: {image_width}"
)

print(
    f"Height: {image_height}"
)


# ============================================================
# LOAD FLOOR MASK
# ============================================================

floor_mask = cv2.imread(
    FLOOR_MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)


if floor_mask is None:

    raise FileNotFoundError(
        f"Could not load floor mask: "
        f"{FLOOR_MASK_PATH}"
    )


# ============================================================
# LOAD FREE SPACE MASK
# ============================================================

free_space_mask = cv2.imread(
    FREE_SPACE_MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)


if free_space_mask is None:

    raise FileNotFoundError(
        f"Could not load free-space mask: "
        f"{FREE_SPACE_MASK_PATH}"
    )


# ============================================================
# MASK INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MASK INFORMATION")
print("=" * 60)


print(
    f"Floor mask: "
    f"{floor_mask.shape[1]} x "
    f"{floor_mask.shape[0]}"
)


print(
    f"Free-space mask: "
    f"{free_space_mask.shape[1]} x "
    f"{free_space_mask.shape[0]}"
)


# ============================================================
# VALIDATE MASK DIMENSIONS
# ============================================================

if floor_mask.shape[:2] != (
    image_height,
    image_width
):

    raise ValueError(
        "Floor mask dimensions do not match "
        "the original image.\n"
        f"Image: {image_width} x {image_height}\n"
        f"Floor mask: "
        f"{floor_mask.shape[1]} x "
        f"{floor_mask.shape[0]}"
    )


if free_space_mask.shape[:2] != (
    image_height,
    image_width
):

    raise ValueError(
        "Free-space mask dimensions do not match "
        "the original image.\n"
        f"Image: {image_width} x {image_height}\n"
        f"Free-space mask: "
        f"{free_space_mask.shape[1]} x "
        f"{free_space_mask.shape[0]}"
    )


print(
    "Mask dimensions: OK"
)


# ============================================================
# TEST MOVE
# ============================================================

instruction = (
    "Move the bed slightly away from the wall."
)


print("\n" + "=" * 60)
print("MOVEMENT TEST")
print("=" * 60)


print(
    f"Instruction: "
    f"{instruction}"
)


# ============================================================
# FIND BEST MOVE POSITION
# ============================================================

result = find_move_position(

    detection=bed,

    detections=detections,

    image_width=image_width,

    image_height=image_height,

    instruction=instruction,

    floor_mask=floor_mask,

    free_space_mask=free_space_mask,

    step=20,

    max_distance=300,

    clearance=30,

    minimum_floor_coverage=0.85,

    minimum_free_space_coverage=0.85
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("MOVEMENT RESULT")
print("=" * 60)


print(
    f"Success: "
    f"{result['success']}"
)


if result["success"]:

    print(
        f"Direction: "
        f"{result['direction']}"
    )


    print(
        f"Movement distance: "
        f"{result['distance']} px"
    )


    print(
        f"Score: "
        f"{result['score']:.4f}"
    )


    # ========================================================
    # OLD POSITION
    # ========================================================

    print(
        "\nOLD POSITION:"
    )

    print(
        result["old_bbox"]
    )


    # ========================================================
    # NEW POSITION
    # ========================================================

    print(
        "\nNEW POSITION:"
    )

    print(
        result["new_bbox"]
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation = result[
        "validation"
    ]


    print("\n" + "=" * 60)
    print("MOVEMENT VALIDATION")
    print("=" * 60)


    # --------------------------------------------------------
    # FLOOR
    # --------------------------------------------------------

    floor_result = validation[
        "floor"
    ]

    print(
        "\nFLOOR:"
    )

    print(
        f"Valid: "
        f"{floor_result['valid']}"
    )

    print(
        f"Coverage: "
        f"{floor_result['coverage']}"
    )

    print(
        f"Reason: "
        f"{floor_result['reason']}"
    )


    # --------------------------------------------------------
    # FREE SPACE
    # --------------------------------------------------------

    free_result = validation[
        "free_space"
    ]

    print(
        "\nFREE SPACE:"
    )

    print(
        f"Valid: "
        f"{free_result['valid']}"
    )

    print(
        f"Coverage: "
        f"{free_result['coverage']}"
    )

    print(
        f"Reason: "
        f"{free_result['reason']}"
    )


    # --------------------------------------------------------
    # FURNITURE MASK COLLISION
    # --------------------------------------------------------

    collision_result = validation[
        "mask_collision"
    ]

    print(
        "\nFURNITURE COLLISION:"
    )

    print(
        f"Valid: "
        f"{collision_result['valid']}"
    )

    print(
        f"Reason: "
        f"{collision_result['reason']}"
    )


else:

    print(
        "\nMOVEMENT FAILED"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )
