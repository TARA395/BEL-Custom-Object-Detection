from ultralytics import YOLO
import os

model = YOLO("runs/detect/runs/animal_detect/train1/weights/best.pt")

file_path = input("Enter image or video path: ")

if os.path.exists(file_path):
    results = model(
        file_path,
        save=True,
        project="runs",
        name="detect_test",
        exist_ok=True
    )

    print("Detection completed!")
else:
    print("File not found!")