from room_analysis.detection import detect_furniture


detections = detect_furniture(
    "room12.jpg",
    confidence=0.5
)


print("\nDetected furniture:\n")


for item in detections:

    print("Name:", item["name"])
    print("Confidence:", round(item["confidence"], 3))
    print("BBox:", item["bbox"])

    if "mask" in item:
        print("Mask shape:", item["mask"].shape)

    print("-" * 40)
