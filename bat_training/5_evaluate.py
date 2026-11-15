"""Step 5: honest evaluation on held-out videos.

Reports mAP/precision/recall on the val split, plus the two numbers that actually failed before:
  - recall frames: share of bat-containing val frames where a bat was found
  - false-alarm rate: share of bat-free val frames (crowd, scoreboard, grass) where a bat box fired

Usage: python 5_evaluate.py --weights runs/bat/weights/best.pt --data data/yolo/data.yaml
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/bat/weights/best.pt")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    m = YOLO(a.weights)
    r = m.val(data=a.data, imgsz=a.imgsz, conf=0.001, verbose=False)
    print(f"mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}  P {r.box.mp:.3f}  R {r.box.mr:.3f}")

    root = Path(a.data).parent
    pos = neg = hit = fa = 0
    for img in sorted((root / "images" / "val").glob("*.jpg")):
        has_bat = bool((root / "labels" / "val" / (img.stem + ".txt")).read_text().strip())
        found = len(m.predict(str(img), imgsz=a.imgsz, conf=a.conf, verbose=False)[0].boxes) > 0
        if has_bat:
            pos += 1
            hit += found
        else:
            neg += 1
            fa += found
    print(f"@conf {a.conf}: bat found in {hit}/{pos} bat frames ({100 * hit / max(pos, 1):.0f}%)")
    print(f"@conf {a.conf}: false alarms on {fa}/{neg} bat-free frames ({100 * fa / max(neg, 1):.0f}%)")
    print("Targets: recall >= 90%, false alarms <= 5%. Frame-level 'found' can hide wrong boxes; "
          "also eyeball a sample of val predictions.")


if __name__ == "__main__":
    main()
