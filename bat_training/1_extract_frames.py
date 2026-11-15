"""Step 1: sample frames from cricket videos for labelling.

Skips near-duplicate frames so the dataset isn't 50 copies of the same moment.
Frames are named <video>__<frame>.jpg so step 3 can split by source video.

Usage: python 1_extract_frames.py ../0.mp4 ../1.mp4 ../7.mp4 --every 5 --out data/images_all
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=int, default=5, help="consider every Nth frame")
    ap.add_argument("--min-diff", type=float, default=4.0,
                    help="skip a frame whose mean abs difference from the last kept frame is below this")
    ap.add_argument("--max-width", type=int, default=1920, help="downscale larger frames (saves disk)")
    ap.add_argument("--out", default="data/images_all")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        cap = cv2.VideoCapture(v)
        stem, i, kept, last = Path(v).stem, 0, 0, None
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
                if last is None or np.abs(g - last).mean() >= a.min_diff:
                    if f.shape[1] > a.max_width:
                        s = a.max_width / f.shape[1]
                        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out / f"{stem}__{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    last, kept = g, kept + 1
            i += 1
        cap.release()
        print(f"{v}: {kept} frames kept of {i}")


if __name__ == "__main__":
    main()
