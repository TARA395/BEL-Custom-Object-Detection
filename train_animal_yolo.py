"""
train_animal_yolo.py

Trains a YOLOv8 nano model on the custom animal dataset
(Kangaroo, Penguin, Squirrel, Panda, Tiger).

This is tuned for CPU training, since AMD GPUs don't support CUDA
on Windows (which PyTorch/YOLOv8 needs for GPU acceleration there).

USAGE:
    1. First time only - install ultralytics:
        pip install ultralytics

    2. Run this script:
        python train_animal_yolo.py
"""

from ultralytics import YOLO


def main():
    # Load a pretrained YOLOv8 nano model as the starting point.
    # "nano" = smallest/fastest YOLOv8 variant, best choice for CPU
    # training and for a dataset this size (~200 images).
    model = YOLO("yolov8n.pt")

    # Train on the custom dataset.
    results = model.train(
        data="animal_dataset.yaml",   # path to the dataset config
        epochs=50,                    # reasonable for a small dataset
        imgsz=416,                    # smaller image size = faster on CPU
        batch=8,                      # modest batch size, CPU-friendly
        device="cpu",                 # explicit: no CUDA GPU available
        project="runs/animal_detect", # where results/checkpoints are saved
        name="train1",                # this run's subfolder name
        patience=15,                  # stop early if no improvement
        workers=2,                    # data-loading threads, conservative for CPU
    )

    print("\nTraining complete.")
    print("Best weights saved at: runs/animal_detect/train1/weights/best.pt")


if __name__ == "__main__":
    main()
