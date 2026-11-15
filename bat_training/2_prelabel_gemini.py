"""Step 2 (optional but saves hours): pre-label frames with Gemini, one image per request.

Writes YOLO txt files (class 0 = bat) to data/labels_all. Frames with no bat get an EMPTY
txt file, which is what teaches the model to ignore crowds and grass.
You MUST review and correct these in a labelling tool (CVAT / Roboflow / labelImg) before training.

Usage: $env:GEMINI_API_KEY="..."; python 2_prelabel_gemini.py --images data/images_all --out data/labels_all
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel


class Result(BaseModel):
    bat_visible: bool
    boxes_2d: list[list[int]]  # each [ymin, xmin, ymax, xmax], 0-1000; [] if no bat


PROMPT = """Cricket footage frame. Find every cricket bat held by a batter.
Return a tight box [ymin, xmin, ymax, xmax] normalised to 0-1000 around each whole bat
(blade + handle, including when motion-blurred). Ignore stumps, pads, gloves, and bats lying idle on the ground.
If no bat is in view (crowd, scoreboard, close-up of a face, empty field) return bat_visible=false and boxes_2d=[]."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images_all")
    ap.add_argument("--out", default="data/labels_all")
    ap.add_argument("--model", default="gemini-3.8-flash")
    a = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first.")
    client = genai.Client(api_key=key)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Result, temperature=0)

    imgs = sorted(Path(a.images).glob("*.jpg"))
    pos = 0
    for n, p in enumerate(imgs, 1):
        txt = out / (p.stem + ".txt")
        if txt.exists():  # resume support
            continue
        img = cv2.imread(str(p))
        small = cv2.resize(img, (1024, int(img.shape[0] * 1024 / img.shape[1]))) if img.shape[1] > 1024 else img
        data = cv2.imencode(".jpg", small)[1].tobytes()
        lines = None
        for _ in range(3):
            try:
                r = client.models.generate_content(
                    model=a.model, config=cfg,
                    contents=[PROMPT, types.Part.from_bytes(data=data, mime_type="image/jpeg")]).parsed
                lines = []
                for ymin, xmin, ymax, xmax in (r.boxes_2d if r.bat_visible else []):
                    if ymax > ymin and xmax > xmin:
                        lines.append(f"0 {(xmin + xmax) / 2000:.6f} {(ymin + ymax) / 2000:.6f} "
                                     f"{(xmax - xmin) / 1000:.6f} {(ymax - ymin) / 1000:.6f}")
                break
            except Exception as e:
                if any(t in str(e) for t in ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                    sys.exit(f"Gemini rejected the API key: {e}")
                print(f"  {p.name}: {e}", file=sys.stderr)
        if lines is None:
            continue  # leave unlabelled so a re-run retries it
        txt.write_text("\n".join(lines))
        pos += bool(lines)
        print(f"  [{n}/{len(imgs)}] {p.name}: {len(lines)} bat(s) | {pos} frames with a bat so far", flush=True)
    print(f"Labels in {out}. Review them before training.")


if __name__ == "__main__":
    main()
