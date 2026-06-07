from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("videos/traffic.mp4")

# COCO class IDs
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:

    success, frame = video.read()

    if not success:
        break

    results = model(frame)

    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    for box in results[0].boxes:

        cls = int(box.cls[0])

        if cls in VEHICLE_CLASSES:

            vehicle_type = VEHICLE_CLASSES[cls]

            counts[vehicle_type] += 1

    total = sum(counts.values())

    print(
        f"Cars:{counts['Car']} | "
        f"Bikes:{counts['Motorcycle']} | "
        f"Buses:{counts['Bus']} | "
        f"Trucks:{counts['Truck']} | "
        f"Total:{total}"
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()