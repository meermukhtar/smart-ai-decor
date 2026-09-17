from room_analysis.regions import find_free_regions
from room_analysis.scale import meters_to_pixels
from room_analysis.placement import check_bounding_box_fit


def analyze_furniture_placement(
    mask_path,
    furniture_width_m,
    furniture_depth_m,
    meters_per_pixel
):
    """
    Analyze all detected free regions and determine
    where a piece of furniture can theoretically fit.
    """

    regions = find_free_regions(
        mask_path
    )

    # Convert furniture dimensions
    furniture_width_px = meters_to_pixels(
        furniture_width_m,
        meters_per_pixel
    )

    furniture_depth_px = meters_to_pixels(
        furniture_depth_m,
        meters_per_pixel
    )

    results = []

    for region in regions:

        bbox = region["bbox"]

        region_width = bbox["width"]
        region_height = bbox["height"]

        fit = check_bounding_box_fit(
            region_width,
            region_height,
            furniture_width_px,
            furniture_depth_px
        )

        results.append({

            "region_id": region["id"],

            "region": region,

            "furniture": {
                "width_m": furniture_width_m,
                "depth_m": furniture_depth_m,

                "width_px": furniture_width_px,
                "depth_px": furniture_depth_px
            },

            "fit": fit

        })

    return results