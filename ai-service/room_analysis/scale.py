def calculate_scale(
    image_width_pixels,
    real_width_meters
):
    """
    Calculate approximate meters per pixel.
    """

    if image_width_pixels <= 0:
        raise ValueError(
            "Image width must be greater than 0"
        )

    if real_width_meters <= 0:
        raise ValueError(
            "Real width must be greater than 0"
        )

    meters_per_pixel = (
        real_width_meters
        / image_width_pixels
    )

    return meters_per_pixel


def pixels_to_meters(
    pixels,
    meters_per_pixel
):
    """
    Convert pixels to approximate meters.
    """

    return (
        pixels
        * meters_per_pixel
    )


def meters_to_pixels(
    meters,
    meters_per_pixel
):
    """
    Convert meters to approximate pixels.
    """

    if meters < 0:
        raise ValueError(
            "Meters cannot be negative"
        )

    if meters_per_pixel <= 0:
        raise ValueError(
            "meters_per_pixel must be greater than 0"
        )

    return int(
        round(
            meters / meters_per_pixel
        )
    )


def region_dimensions(
    region,
    meters_per_pixel
):

    width_pixels = (
        region["bbox"]["width"]
    )

    height_pixels = (
        region["bbox"]["height"]
    )

    width_meters = pixels_to_meters(
        width_pixels,
        meters_per_pixel
    )

    height_meters = pixels_to_meters(
        height_pixels,
        meters_per_pixel
    )

    return {

        "width_m": round(
            width_meters,
            2
        ),

        "height_m": round(
            height_meters,
            2
        )
    }