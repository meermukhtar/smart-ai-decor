from room_analysis.depth import estimate_depth
from room_analysis.analyzer import analyze_room


image_path = "rooms.jpeg"


depth = estimate_depth(
    image_path
)


analysis = analyze_room(
    depth
)


print("\nROOM ANALYSIS")
print("====================")

print(
    "Image:",
    analysis["image_width"],
    "x",
    analysis["image_height"]
)

print(
    "Floor candidate:",
    analysis["floor_candidate"]
)