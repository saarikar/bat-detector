"""Step 4: train a bat-only detector.

Choices that address best.pt's problems:
- single 'bat' class, started from COCO-pretrained weights (not from best.pt, which carries its Batsman bias)
- imgsz 1280: bats are thin and small in broadcast frames; 640 erased them
- blur/scale/colour augmentation to survive motion-blurred swings and varied camera zoom
- empty-label (negative) frames in the dataset teach it to stay quiet on crowds and grass

Use a GPU: on this CPU-only laptop imgsz 1280 is impractically slow. Run it on Google Colab
(upload data/yolo and this script) or any CUDA machine.

Usage: python 4_train.py --data data/yolo/data.yaml [--model yolov8m.pt] [--epochs 100] [--device 0]
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="0 for first GPU, 'cpu' otherwise")
    a = ap.parse_args()

    YOLO(a.model).train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
        patience=25, project="runs", name="bat",
        # augmentation
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8, translate=0.15, scale=0.6,
        fliplr=0.5, mosaic=1.0, close_mosaic=10, mixup=0.1,
        # keep going after plateau, save best by val mAP50-95
        cos_lr=True, plots=True,
    )
    print("Best weights: runs/bat/weights/best.pt")


if __name__ == "__main__":
    main()
