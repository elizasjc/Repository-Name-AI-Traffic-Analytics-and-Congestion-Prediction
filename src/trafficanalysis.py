from ultralytics import YOLO
import cv2
import pandas as pd

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("videos/traffic.mp4")
print("Video loaded")

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

records = []

frame_number = 0

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    if frame_number % 100 != 0:
        continue
    print(f"Processing frame {frame_number}")
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
            counts[VEHICLE_CLASSES[cls]] += 1

    total = sum(counts.values())

    records.append({
        "frame": frame_number,
        "cars": counts["Car"],
        "motorcycles": counts["Motorcycle"],
        "buses": counts["Bus"],
        "trucks": counts["Truck"],
        "total_vehicles": total
    })

video.release()

df = pd.DataFrame(records)

df.to_csv("data/traffic_counts.csv", index=False)

print("CSV file created successfully")