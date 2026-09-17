import numpy as np
import cv2


# ============================================================
# BASIC BBOX HELPERS
# ============================================================

def bbox_center(bbox):
    return (
        (bbox["x1"] + bbox["x2"]) / 2.0,
        (bbox["y1"] + bbox["y2"]) / 2.0
    )


def create_bbox(x1, y1, width, height):
    x1 = int(round(x1))
    y1 = int(round(y1))
    width = int(round(width))
    height = int(round(height))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x1 + width,
        "y2": y1 + height,
        "width": width,
        "height": height
    }


def boxes_overlap(box1, box2):
    return not (
        box1["x2"] <= box2["x1"]
        or box1["x1"] >= box2["x2"]
        or box1["y2"] <= box2["y1"]
        or box1["y1"] >= box2["y2"]
    )


def distance_between_boxes(box1, box2):
    cx1, cy1 = bbox_center(box1)
    cx2, cy2 = bbox_center(box2)

    return float(
        np.sqrt(
            (cx1 - cx2) ** 2 +
            (cy1 - cy2) ** 2
        )
    )


# ============================================================
# MOVEMENT DIRECTION
# ============================================================

def get_movement_direction(instruction):

    if not instruction:
        return "unknown"

    instruction = instruction.lower().strip()

    # LEFT
    if (
        "move left" in instruction
        or "to the left" in instruction
        or "toward left" in instruction
        or "towards left" in instruction
    ):
        return "left"

    # RIGHT
    if (
        "move right" in instruction
        or "to the right" in instruction
        or "toward right" in instruction
        or "towards right" in instruction
    ):
        return "right"

    # AWAY FROM WALL
    if (
        "away from wall" in instruction
        or "away from the wall" in instruction
        or "away from a wall" in instruction
        or "move away from wall" in instruction
        or "move away from the wall" in instruction
    ):
        return "away_from_wall"

    # TOWARD WALL
    if (
        "toward wall" in instruction
        or "towards wall" in instruction
        or "toward the wall" in instruction
        or "towards the wall" in instruction
        or "move toward wall" in instruction
        or "move towards wall" in instruction
        or "move toward the wall" in instruction
        or "move towards the wall" in instruction
    ):
        return "toward_wall"

    # UP
    if (
        "move up" in instruction
        or "upward" in instruction
        or "upwards" in instruction
    ):
        return "up"

    # DOWN
    if (
        "move down" in instruction
        or "downward" in instruction
        or "downwards" in instruction
    ):
        return "down"

    # CENTER
    if (
        "move to center" in instruction
        or "move toward center" in instruction
        or "move towards center" in instruction
        or "center the" in instruction
    ):
        return "center"

    return "unknown"


# ============================================================
# CANDIDATE POSITION
# ============================================================

def generate_candidate(
    bbox,
    direction,
    distance
):

    x = bbox["x1"]
    y = bbox["y1"]

    width = bbox["width"]
    height = bbox["height"]

    if direction == "left":
        x -= distance

    elif direction == "right":
        x += distance

    elif direction == "up":
        y -= distance

    elif direction == "down":
        y += distance

    elif direction == "away_from_wall":
        # Temporary assumption:
        # wall is behind object, so move toward camera/down.
        y += distance

    elif direction == "toward_wall":
        y -= distance

    else:
        return None

    return create_bbox(
        x,
        y,
        width,
        height
    )


# ============================================================
# IMAGE BOUNDARY CHECK
# ============================================================

def is_inside_image(
    bbox,
    image_width,
    image_height
):

    return (
        bbox["x1"] >= 0
        and bbox["y1"] >= 0
        and bbox["x2"] <= image_width
        and bbox["y2"] <= image_height
    )


# ============================================================
# BBOX COLLISION
# ============================================================

def has_collision(
    candidate_bbox,
    detections,
    current_detection
):

    for detection in detections:

        if detection is current_detection:
            continue

        other_bbox = detection.get("bbox")

        if other_bbox is None:
            continue

        if boxes_overlap(
            candidate_bbox,
            other_bbox
        ):
            return True

    return False


# ============================================================
# MASK COVERAGE
# ============================================================

def calculate_mask_coverage(
    bbox,
    mask
):

    if mask is None:
        return 0.0

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be a 2D grayscale array."
        )

    mask_height, mask_width = mask.shape[:2]

    x1 = max(
        0,
        int(bbox["x1"])
    )

    y1 = max(
        0,
        int(bbox["y1"])
    )

    x2 = min(
        mask_width,
        int(bbox["x2"])
    )

    y2 = min(
        mask_height,
        int(bbox["y2"])
    )

    if x1 >= x2 or y1 >= y2:
        return 0.0

    region = mask[
        y1:y2,
        x1:x2
    ]

    if region.size == 0:
        return 0.0

    occupied_pixels = np.count_nonzero(
        region > 0
    )

    total_pixels = region.size

    return float(
        occupied_pixels / total_pixels
    )


# ============================================================
# CREATE MOVEMENT-AWARE FREE SPACE
# ============================================================

def create_movement_free_space(
    free_space_mask,
    current_detection
):
    """
    The current furniture occupies space that is not marked
    as free in the original free-space mask.

    While moving this furniture, its OWN current area should
    be considered available.

    Therefore:

        movement_free_space =
            free_space_mask OR current furniture mask
    """

    if free_space_mask is None:
        return None

    movement_free_space = free_space_mask.copy()

    current_mask = current_detection.get("mask")

    if current_mask is None:
        return movement_free_space

    if current_mask.shape != movement_free_space.shape:
        raise ValueError(
            "Current furniture mask and free-space mask "
            "must have the same dimensions."
        )

    movement_free_space = cv2.bitwise_or(
        movement_free_space,
        current_mask
    )

    return movement_free_space


# ============================================================
# FLOOR VALIDATION
# ============================================================

def check_floor_position(
    candidate_bbox,
    floor_mask,
    minimum_coverage=0.85
):

    if floor_mask is None:
        return {
            "valid": True,
            "coverage": None,
            "reason": "No floor mask provided"
        }

    coverage = calculate_mask_coverage(
        candidate_bbox,
        floor_mask
    )

    valid = coverage >= minimum_coverage

    return {
        "valid": valid,
        "coverage": coverage,
        "reason": (
            "Enough floor coverage"
            if valid
            else "Candidate is not sufficiently on floor"
        )
    }


# ============================================================
# FREE SPACE VALIDATION
# ============================================================

def check_free_space(
    candidate_bbox,
    free_space_mask,
    minimum_coverage=0.85
):

    if free_space_mask is None:
        return {
            "valid": True,
            "coverage": None,
            "reason": "No free-space mask provided"
        }

    coverage = calculate_mask_coverage(
        candidate_bbox,
        free_space_mask
    )

    valid = coverage >= minimum_coverage

    return {
        "valid": valid,
        "coverage": coverage,
        "reason": (
            "Enough free space"
            if valid
            else "Candidate overlaps occupied space"
        )
    }


# ============================================================
# FURNITURE MASK COLLISION
# ============================================================

def check_furniture_masks(
    candidate_bbox,
    detections,
    current_detection,
    maximum_overlap_ratio=0.10
):

    candidate_width = candidate_bbox["width"]
    candidate_height = candidate_bbox["height"]

    candidate_area = (
        candidate_width *
        candidate_height
    )

    if candidate_area <= 0:
        return {
            "valid": False,
            "reason": "Invalid candidate area"
        }

    for detection in detections:

        if detection is current_detection:
            continue

        other_mask = detection.get("mask")

        if other_mask is None:
            continue

        mask_height, mask_width = other_mask.shape[:2]

        x1 = max(
            0,
            candidate_bbox["x1"]
        )

        y1 = max(
            0,
            candidate_bbox["y1"]
        )

        x2 = min(
            mask_width,
            candidate_bbox["x2"]
        )

        y2 = min(
            mask_height,
            candidate_bbox["y2"]
        )

        if x1 >= x2 or y1 >= y2:
            continue

        overlap = other_mask[
            y1:y2,
            x1:x2
        ]

        overlap_pixels = np.count_nonzero(
            overlap > 0
        )

        overlap_ratio = (
            overlap_pixels /
            candidate_area
        )

        if overlap_ratio > maximum_overlap_ratio:

            return {
                "valid": False,
                "reason": (
                    f"Collision with "
                    f"{detection['catalog_name']}"
                ),
                "overlap_ratio": float(
                    overlap_ratio
                )
            }

    return {
        "valid": True,
        "reason": "No furniture mask collision",
        "overlap_ratio": 0.0
    }


# ============================================================
# CLEARANCE VALIDATION
# ============================================================

def apply_clearance_to_mask(
    mask,
    clearance
):

    if mask is None:
        return None

    if clearance <= 0:
        return mask

    kernel_size = int(clearance * 2 + 1)

    kernel_size = max(
        3,
        kernel_size
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    return cv2.dilate(
        mask,
        kernel,
        iterations=1
    )


def check_clearance(
    candidate_bbox,
    detections,
    current_detection,
    clearance=0
):

    if clearance <= 0:
        return {
            "valid": True,
            "reason": "No clearance requested"
        }

    candidate_width = candidate_bbox["width"]
    candidate_height = candidate_bbox["height"]

    candidate_area = (
        candidate_width *
        candidate_height
    )

    if candidate_area <= 0:
        return {
            "valid": False,
            "reason": "Invalid candidate area"
        }

    for detection in detections:

        if detection is current_detection:
            continue

        other_mask = detection.get("mask")

        if other_mask is None:
            continue

        expanded_mask = apply_clearance_to_mask(
            other_mask,
            clearance
        )

        mask_height, mask_width = expanded_mask.shape[:2]

        x1 = max(
            0,
            candidate_bbox["x1"]
        )

        y1 = max(
            0,
            candidate_bbox["y1"]
        )

        x2 = min(
            mask_width,
            candidate_bbox["x2"]
        )

        y2 = min(
            mask_height,
            candidate_bbox["y2"]
        )

        if x1 >= x2 or y1 >= y2:
            continue

        overlap = expanded_mask[
            y1:y2,
            x1:x2
        ]

        overlap_pixels = np.count_nonzero(
            overlap > 0
        )

        if overlap_pixels > 0:

            return {
                "valid": False,
                "reason": (
                    f"Clearance violation near "
                    f"{detection['catalog_name']}"
                )
            }

    return {
        "valid": True,
        "reason": "Clearance satisfied"
    }


# ============================================================
# CANDIDATE SCORE
# ============================================================

def calculate_candidate_score(
    distance,
    floor_result,
    free_space_result
):

    floor_score = (
        floor_result["coverage"]
        if floor_result["coverage"] is not None
        else 1.0
    )

    free_space_score = (
        free_space_result["coverage"]
        if free_space_result["coverage"] is not None
        else 1.0
    )

    movement_penalty = (
        float(distance) * 0.001
    )

    score = (
        floor_score * 0.45
        + free_space_score * 0.45
        - movement_penalty
    )

    return float(score)


# ============================================================
# FIND MOVE POSITION
# ============================================================

def find_move_position(
    detection,
    detections,
    image_width,
    image_height,
    instruction,
    floor_mask=None,
    free_space_mask=None,
    step=20,
    max_distance=300,
    clearance=30,
    minimum_floor_coverage=0.85,
    minimum_free_space_coverage=0.85
):

    # ========================================================
    # DETERMINE DIRECTION
    # ========================================================

    direction = get_movement_direction(
        instruction
    )

    if direction == "unknown":

        return {
            "success": False,
            "direction": "unknown",
            "reason": (
                "Could not determine movement direction "
                f"from instruction: '{instruction}'"
            )
        }

    # ========================================================
    # CURRENT POSITION
    # ========================================================

    current_bbox = detection.get("bbox")

    if current_bbox is None:

        return {
            "success": False,
            "direction": direction,
            "reason": "Detection does not contain a bbox"
        }

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if step <= 0:

        return {
            "success": False,
            "direction": direction,
            "reason": "step must be greater than 0"
        }

    if max_distance < step:

        return {
            "success": False,
            "direction": direction,
            "reason": "max_distance must be >= step"
        }

    # ========================================================
    # MOVEMENT-AWARE FREE SPACE
    # ========================================================

    movement_free_space = create_movement_free_space(
        free_space_mask,
        detection
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    best_candidate = None
    best_score = -float("inf")

    candidates_checked = 0

    candidates_rejected = {
        "boundary": 0,
        "bbox_collision": 0,
        "floor": 0,
        "free_space": 0,
        "mask_collision": 0,
        "clearance": 0
    }

    # ========================================================
    # GENERATE CANDIDATES
    # ========================================================

    for distance in range(
        step,
        max_distance + 1,
        step
    ):

        candidates_checked += 1

        candidate_bbox = generate_candidate(
            current_bbox,
            direction,
            distance
        )

        if candidate_bbox is None:
            continue

        # ====================================================
        # 1. IMAGE BOUNDARY
        # ====================================================

        if not is_inside_image(
            candidate_bbox,
            image_width,
            image_height
        ):

            candidates_rejected[
                "boundary"
            ] += 1

            continue

        # ====================================================
        # 2. BBOX COLLISION
        # ====================================================

        if has_collision(
            candidate_bbox,
            detections,
            detection
        ):

            candidates_rejected[
                "bbox_collision"
            ] += 1

            continue

        # ====================================================
        # 3. FLOOR
        # ====================================================

        floor_result = check_floor_position(
            candidate_bbox,
            floor_mask,
            minimum_floor_coverage
        )

        if not floor_result["valid"]:

            candidates_rejected[
                "floor"
            ] += 1

            continue

        # ====================================================
        # 4. FREE SPACE
        # ====================================================

        free_space_result = check_free_space(
            candidate_bbox,
            movement_free_space,
            minimum_free_space_coverage
        )

        if not free_space_result["valid"]:

            candidates_rejected[
                "free_space"
            ] += 1

            continue

        # ====================================================
        # 5. FURNITURE MASK COLLISION
        # ====================================================

        mask_result = check_furniture_masks(
            candidate_bbox,
            detections,
            detection
        )

        if not mask_result["valid"]:

            candidates_rejected[
                "mask_collision"
            ] += 1

            continue

        # ====================================================
        # 6. CLEARANCE
        # ====================================================

        clearance_result = check_clearance(
            candidate_bbox,
            detections,
            detection,
            clearance
        )

        if not clearance_result["valid"]:

            candidates_rejected[
                "clearance"
            ] += 1

            continue

        # ====================================================
        # 7. SCORE
        # ====================================================

        score = calculate_candidate_score(
            distance,
            floor_result,
            free_space_result
        )

        # ====================================================
        # 8. BEST CANDIDATE
        # ====================================================

        if score > best_score:

            best_score = score

            best_candidate = {
                "bbox": candidate_bbox,
                "distance": distance,
                "score": score,
                "floor": floor_result,
                "free_space": free_space_result,
                "mask_collision": mask_result,
                "clearance": clearance_result
            }

    # ========================================================
    # NO VALID POSITION
    # ========================================================

    if best_candidate is None:

        return {
            "success": False,
            "direction": direction,
            "reason": "No valid movement position found",
            "debug": {
                "candidates_checked": candidates_checked,
                "candidates_rejected": candidates_rejected
            }
        }

    # ========================================================
    # SUCCESS
    # ========================================================

    return {
        "success": True,

        "direction": direction,

        "distance": best_candidate["distance"],

        "old_bbox": current_bbox,

        "new_bbox": best_candidate["bbox"],

        "score": best_candidate["score"],

        "validation": {
            "floor": best_candidate["floor"],
            "free_space": best_candidate["free_space"],
            "mask_collision": best_candidate["mask_collision"],
            "clearance": best_candidate["clearance"]
        },

        "debug": {
            "candidates_checked": candidates_checked,
            "candidates_rejected": candidates_rejected
        }
    }