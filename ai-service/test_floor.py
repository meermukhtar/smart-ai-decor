import cv2

from room_analysis.depth import estimate_depth
from room_analysis.analyzer import analyze_room


image_path = "rooms.jpeg"

# -----------------------------
# Run AI analysis
# -----------------------------

depth = estimate_depth(image_path)

analysis = analyze_room(depth)

print("\n========== ANALYSIS ==========")
print(analysis)

print("\n========== AVAILABLE KEYS ==========")
print(analysis.keys())


# -----------------------------
# Load original
# -----------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# -----------------------------
# Save original
# -----------------------------

cv2.imwrite(
    "original.jpg",
    image
)

print("\nSaved: original.jpg")


# -----------------------------
# Generate candidate images
# -----------------------------

for name, floor in analysis.items():

    if not name.startswith("floor_candidate"):
        continue

    output = image.copy()

    x = floor["x"]
    y = floor["y"]
    w = floor["width"]
    h = floor["height"]

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        4
    )

    output_path = f"{name}.jpg"

    cv2.imwrite(
        output_path,
        output
    )

    print(f"Saved: {output_path}")


print("\n========== DONE ==========")