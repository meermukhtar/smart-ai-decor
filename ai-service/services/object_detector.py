from ultralytics import YOLO

model = YOLO("yolo11n.pt")


def detect_objects(image_path):

    results = model(image_path)

    objects = []

    for result in results:

        for box in result.boxes:

            objects.append({
                "label": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })

    return objects