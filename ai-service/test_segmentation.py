from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

results = model(
    "rooms.jpeg",
    conf=0.25,
)

for result in results:

    print("Number of detections:", len(result.boxes))

    if result.masks is not None:
        print("Number of masks:", len(result.masks))

    result.save(
        filename="rooms_segmented.jpg"
    )

    print("Saved segmented image")