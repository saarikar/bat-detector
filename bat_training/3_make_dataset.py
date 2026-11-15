"""Step 3: build the YOLO dataset, splitting by SOURCE VIDEO.

Neighbouring frames of one video are near-duplicates, so a random frame split leaks and gives
inflated scores (likely why best.pt reported mAP50 0.995 but failed on new footage).
Held-out videos go entirely to val.

Usage: python 3_make_dataset.py --val-videos 5 7 --images data/images_all --labels data/labels_all
"""
import argparse
import shutil
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images_all")
    ap.add_argument("--labels", default="data/labels_all")
    ap.add_argument("--val-videos", nargs="+", required=True, help="video stems held out for validation, e.g. 5 7")
    ap.add_argument("--out", default="data/yolo")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    stats = {"train": [0, 0], "val": [0, 0]}  # [images, images with a bat]
    for img in sorted(Path(a.images).glob("*.jpg")):
        lab = Path(a.labels) / (img.stem + ".txt")
        if not lab.exists():
            continue  # unreviewed / unlabelled frames are skipped, not treated as negatives
        split = "val" if img.stem.split("__")[0] in a.val_videos else "train"
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out / "images" / split / img.name)
        shutil.copy(lab, out / "labels" / split / lab.name)
        stats[split][0] += 1
        stats[split][1] += bool(lab.read_text().strip())

    for s, (n, p) in stats.items():
        print(f"{s}: {n} images, {p} with a bat, {n - p} negatives")
    if not stats["val"][0] or not stats["train"][0]:
        raise SystemExit("Empty train or val split; check --val-videos and that labels exist.")
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": {0: "bat"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg))
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
