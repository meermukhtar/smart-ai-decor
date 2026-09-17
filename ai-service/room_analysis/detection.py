from ultralytics import YOLO
import cv2
import numpy as np


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = "yolo11n-seg.pt"

model = YOLO(MODEL_PATH)


# ============================================================
# FURNITURE MAPPING
# ============================================================

FURNITURE_MAPPING = {
    "bed": "double_bed",
    "couch": "3_seater_sofa",
    "chair": "armchair",
    "dining table": "dining_table",
    "dining chair": "dining_chair",
}


# ============================================================
# CREATE ORIGINAL-RESOLUTION MASK
# ============================================================

def create_original_mask(
    result,
    mask_index
):
    """
    Convert YOLO segmentation polygon into a binary
    mask matching the original image resolution.

    YOLO's result.masks.xy contains polygon coordinates
    already mapped to the original image dimensions.

    Returns:
        numpy array with shape:

        (original_height, original_width)

    Values:

        0   = background
        255 = furniture
    """

    # --------------------------------------------------------
    # ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    original_height, original_width = result.orig_shape


    # --------------------------------------------------------
    # EMPTY MASK
    # --------------------------------------------------------

    mask = np.zeros(
        (
            original_height,
            original_width
        ),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # GET POLYGON
    # --------------------------------------------------------

    polygon = result.masks.xy[mask_index]


    if polygon is None or len(polygon) == 0:

        return mask


    # --------------------------------------------------------
    # CONVERT POLYGON TO INTEGER PIXELS
    # --------------------------------------------------------

    polygon = np.asarray(
        polygon,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # FILL POLYGON
    # --------------------------------------------------------

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )


    return mask


# ============================================================
# DETECT FURNITURE
# ============================================================

def detect_furniture(
    image_path,
    confidence=0.5
):
    """
    Detect supported furniture in a room.

    Returns:
        [
            {
                "name": "bed",

                "catalog_name": "double_bed",

                "confidence": 0.69,

                "bbox": {
                    "x1": 567,
                    "y1": 532,
                    "x2": 1189,
                    "y2": 809,
                    "width": 622,
                    "height": 277
                },

                "mask": <original resolution mask>,

                "mask_shape": (
                    1066,
                    1543
                )
            }
        ]

    Important:

    - bbox coordinates are already in ORIGINAL image
      coordinates because Ultralytics scales them back.
    - mask is converted to ORIGINAL image resolution.
    - only supported furniture is returned.
    """

    # ========================================================
    # RUN YOLO
    # ========================================================

    results = model(
        image_path,
        conf=confidence
    )

    detections = []


    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    for result in results:

        # ----------------------------------------------------
        # NO DETECTIONS
        # ----------------------------------------------------

        if result.boxes is None:
            continue


        # ----------------------------------------------------
        # ORIGINAL IMAGE SIZE
        # ----------------------------------------------------

        original_height, original_width = (
            result.orig_shape
        )


        # ====================================================
        # PROCESS EACH DETECTION
        # ====================================================

        for i, box in enumerate(result.boxes):

            # ------------------------------------------------
            # CLASS ID
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            score = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # YOLO CLASS NAME
            # ------------------------------------------------

            name = result.names[
                class_id
            ]


            # ------------------------------------------------
            # IGNORE NON-FURNITURE
            # ------------------------------------------------

            if name not in FURNITURE_MAPPING:
                continue


            # ------------------------------------------------
            # MAP TO OUR CATALOG
            # ------------------------------------------------

            catalog_name = FURNITURE_MAPPING[
                name
            ]


            # =================================================
            # BOUNDING BOX
            # =================================================

            # IMPORTANT:
            #
            # Ultralytics already maps xyxy back to the
            # ORIGINAL image dimensions.
            #
            # Therefore we do NOT manually resize these
            # coordinates.

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # ------------------------------------------------
            # CLAMP BBOX TO ORIGINAL IMAGE
            # ------------------------------------------------

            x1 = max(
                0,
                min(
                    x1,
                    original_width - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    original_height - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    original_width
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    original_height
                )
            )


            # ------------------------------------------------
            # BBOX DIMENSIONS
            # ------------------------------------------------

            width = int(
                x2 - x1
            )

            height = int(
                y2 - y1
            )


            # =================================================
            # CREATE DETECTION
            # =================================================

            detection = {

                "name": name,

                "catalog_name": catalog_name,

                "confidence": score,

                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": width,
                    "height": height
                }
            }


            # =================================================
            # SEGMENTATION MASK
            # =================================================

            if result.masks is not None:

                # ---------------------------------------------
                # CREATE ORIGINAL-RESOLUTION MASK
                # ---------------------------------------------

                mask = create_original_mask(
                    result,
                    i
                )


                detection["mask"] = mask


                # ---------------------------------------------
                # MASK SHAPE
                # ---------------------------------------------

                detection["mask_shape"] = (
                    int(mask.shape[0]),
                    int(mask.shape[1])
                )


                # ---------------------------------------------
                # MASK AREA
                # ---------------------------------------------

                detection["mask_area"] = int(
                    np.count_nonzero(mask)
                )


            # =================================================
            # ADD DETECTION
            # =================================================

            detections.append(
                detection
            )


    # ========================================================
    # RETURN
    # ========================================================

    return detections
