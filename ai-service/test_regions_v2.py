from room_analysis.regions import (
    find_free_regions
)


regions = find_free_regions(
    "free_space_mask.png"
)


print("\n==============================")
print("FREE SPACE ANALYSIS")
print("==============================")


for region in regions:

    print("\nRegion:", region["id"])

    print(
        "Bounding box:",
        region["bbox"]
    )

    print(
        "Area:",
        region["area_pixels"]
    )

    print(
        "Center:",
        region["center"]
    )

    print(
        "Aspect ratio:",
        region["aspect_ratio"]
    )

    print(
        "Shape:",
        region["shape"]
    )


print(
    "\nTotal regions:",
    len(regions)
)