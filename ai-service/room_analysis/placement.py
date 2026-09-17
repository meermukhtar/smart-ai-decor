import cv2
import numpy as np


def check_bounding_box_fit(
    region_width,
    region_height,
    furniture_width,
    furniture_height
):
    """
    Check whether furniture can theoretically fit
    inside a rectangular region.
    """

    normal_fit = (
        furniture_width <= region_width
        and
        furniture_height <= region_height
    )

    rotated_fit = (
        furniture_height <= region_width
        and
        furniture_width <= region_height
    )

    return {
        "normal_fit": normal_fit,
        "rotated_fit": rotated_fit,
        "fits": normal_fit or rotated_fit
    }


def find_best_position(
    free_mask,
    region,
    furniture_width,
    furniture_height,
    step=10,
    allow_rotation=True
):
    """
    Find the best position for furniture inside
    a free-space region.
    """

    bbox = region["bbox"]

    region_x = bbox["x"]
    region_y = bbox["y"]

    region_width = bbox["width"]
    region_height = bbox["height"]


    # =================================================
    # Make sure mask is binary
    # =================================================

    free_mask = (
        free_mask > 0
    ).astype(np.uint8)


    # =================================================
    # Furniture orientations
    # =================================================

    orientations = [
        (
            furniture_width,
            furniture_height,
            0
        )
    ]

    if allow_rotation:
        orientations.append(
            (
                furniture_height,
                furniture_width,
                90
            )
        )


    best_position = None

    best_score = -float("inf")


    # =================================================
    # Check orientations
    # =================================================

    for fw, fh, rotation in orientations:

        # -------------------------------------------------
        # Basic geometric check
        # -------------------------------------------------

        if fw > region_width:
            continue

        if fh > region_height:
            continue


        # -------------------------------------------------
        # Furniture footprint
        # -------------------------------------------------

        kernel = np.ones(
            (
                fh,
                fw
            ),
            dtype=np.uint8
        )


        # Find valid top-left positions with anchor=(0,0)
        # so rectangle [y:y+fh, x:x+fw] is strictly inside free_mask
        eroded = cv2.erode(
            free_mask,
            kernel,
            anchor=(0, 0),
            iterations=1
        )


        # -------------------------------------------------
        # Restrict to current region
        # -------------------------------------------------

        region_area = np.zeros_like(
            eroded
        )

        region_area[
            region_y:
            region_y + region_height,

            region_x:
            region_x + region_width
        ] = 1


        valid_positions = (
            eroded
            &
            region_area
        )


        # -------------------------------------------------
        # Get coordinates
        # -------------------------------------------------

        ys, xs = np.where(
            valid_positions > 0
        )


        if len(xs) == 0:
            continue


        # -------------------------------------------------
        # Sample coordinates
        #
        # Instead of requiring x/y to be divisible by
        # step, simply take every Nth valid point.
        # -------------------------------------------------

        xs = xs[::step]
        ys = ys[::step]


        # -------------------------------------------------
        # Region center
        # -------------------------------------------------

        region_center_x = (
            region_x
            + region_width / 2
        )

        region_center_y = (
            region_y
            + region_height / 2
        )


        # -------------------------------------------------
        # Find best point
        # -------------------------------------------------

        img_h, img_w = free_mask.shape[:2]

        for x, y in zip(xs, ys):
            if x < 15 or (x + fw) > (img_w - 15) or (y + fh) > (img_h - 15):
                continue

            furniture_center_x = (
                x
                + fw / 2
            )

            furniture_center_y = (
                y
                + fh / 2
            )


            distance = np.sqrt(

                (
                    furniture_center_x
                    - region_center_x
                ) ** 2

                +

                (
                    furniture_center_y
                    - region_center_y
                ) ** 2

            )


            # Closer to center = better

            score = -distance


            if score > best_score:

                best_score = score

                best_position = {

                    "x": int(x),

                    "y": int(y),

                    "width": int(fw),

                    "height": int(fh),

                    "rotation": int(rotation),

                    "score": float(score)

                }


    return best_position