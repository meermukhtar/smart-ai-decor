import os
import cv2
import numpy as np

AI_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_furniture(path):
    """
    Load furniture image with transparency.
    """
    resolved_path = path
    if not os.path.isabs(path):
        candidate = os.path.join(AI_SERVICE_DIR, path)
        if os.path.exists(candidate):
            resolved_path = candidate

    image = cv2.imread(
        resolved_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load furniture: {path} (resolved: {resolved_path})"
        )

    if image.shape[2] != 4:
        raise ValueError(
            "Furniture image must have an alpha channel (RGBA)"
        )

    return image


def crop_transparent(image):
    """
    Remove transparent borders around furniture.
    """

    alpha = image[:, :, 3]

    coords = cv2.findNonZero(alpha)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    return image[
        y:y+h,
        x:x+w
    ]


def resize_furniture(
    furniture,
    width,
    height
):
    """
    Resize furniture to target pixel dimensions.
    """

    return cv2.resize(
        furniture,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA
    )


def rotate_furniture(
    furniture,
    angle
):
    """
    Rotate furniture while preserving transparency.
    """

    if angle == 0:
        return furniture

    h, w = furniture.shape[:2]

    center = (
        w / 2,
        h / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int(
        h * sin + w * cos
    )

    new_height = int(
        h * cos + w * sin
    )

    matrix[0, 2] += (
        new_width / 2
        - center[0]
    )

    matrix[1, 2] += (
        new_height / 2
        - center[1]
    )

    return cv2.warpAffine(
        furniture,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def create_shadow(
    furniture,
    offset=(10, 10),
    blur=15,
    opacity=0.35
):
    """
    Create soft shadow from furniture alpha.
    """

    alpha = furniture[:, :, 3]

    shadow = np.zeros_like(
        furniture
    )

    shadow[:, :, 3] = alpha

    shadow = cv2.GaussianBlur(
        shadow,
        (0, 0),
        blur
    )

    shadow[:, :, 3] = (
        shadow[:, :, 3]
        * opacity
    ).astype(np.uint8)

    return shadow


def overlay_image(
    background,
    foreground,
    x,
    y
):
    """
    Alpha composite foreground onto background.
    """

    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # Clip if furniture goes outside image
    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(
        x + fg_w,
        bg_w
    )

    y2 = min(
        y + fg_h,
        bg_h
    )

    if x1 >= x2 or y1 >= y2:
        return background

    fg_x1 = x1 - x
    fg_y1 = y1 - y

    fg_x2 = fg_x1 + (
        x2 - x1
    )

    fg_y2 = fg_y1 + (
        y2 - y1
    )

    foreground_crop = foreground[
        fg_y1:fg_y2,
        fg_x1:fg_x2
    ]

    alpha = (
        foreground_crop[:, :, 3]
        / 255.0
    )

    alpha = alpha[:, :, None]

    bg_region = background[
        y1:y2,
        x1:x2
    ]

    foreground_rgb = (
        foreground_crop[:, :, :3]
    )

    background[
        y1:y2,
        x1:x2
    ] = (
        foreground_rgb * alpha
        +
        bg_region * (1 - alpha)
    ).astype(np.uint8)

    return background


def render_furniture(
    background,
    furniture_path,
    x,
    y,
    width,
    height,
    rotation=0,
    add_shadow=True
):
    """
    Render one furniture item onto a room image.
    """

    furniture = load_furniture(
        furniture_path
    )

    furniture = crop_transparent(
        furniture
    )

    furniture = resize_furniture(
        furniture,
        width,
        height
    )

    furniture = rotate_furniture(
        furniture,
        rotation
    )

    if add_shadow:

        shadow = create_shadow(
            furniture
        )

        overlay_image(
            background,
            shadow,
            x + 8,
            y + 8
        )

    overlay_image(
        background,
        furniture,
        x,
        y
    )

    return background