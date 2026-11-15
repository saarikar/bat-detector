"""Detect the bat and bat-ball impact frames in cricket footage using Gemini.

Frames are sampled from the video, sent to Gemini in chunks as labelled images,
and Gemini returns a bat bounding box per frame plus any impact frames.
Boxes for unsampled frames are linearly interpolated, then an annotated video is written.

Usage:
    set GEMINI_API_KEY=your-key        (PowerShell: $env:GEMINI_API_KEY="your-key")
    python cricket_gemini.py input.mp4 [-o out.mp4] [--stride 1] [--model gemini-2.5-flash]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

SEND_WIDTH = 768  # frames are downscaled to this width before upload


class BatBox(BaseModel):
    frame: int
    visible: bool
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], normalised 0-1000; [] if not visible


class ChunkResult(BaseModel):
    bats: list[BatBox]
    impact_frames: list[int]


PROMPT = """You are analysing cricket footage. Each image below is preceded by its frame number.
For EVERY frame listed, report where the cricket bat is:
- visible=true and box_2d=[ymin, xmin, ymax, xmax] normalised to 0-1000, tightly around the whole bat
  (blade and handle, including when motion-blurred or partly hidden by the batter's hands);
- visible=false and box_2d=[] if no bat is in view (e.g. crowd shot, close-up, replay graphic).
Only the bat held by a batter counts; ignore stumps, pads and gloves.

Also report impact_frames: the frame number(s) where the bat makes contact with the ball.
Contact is the moment the ball touches the bat and abruptly changes direction or speed.
Use only frame numbers shown below. Return an empty list if no contact happens in these frames.
Frames in this chunk: {frames}"""


def read_sampled(cap, stride):
    """Yield (frame_index, frame) for every `stride`-th frame."""
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            return
        if i % stride == 0:
            yield i, f
        i += 1


def encode(frame):
    h, w = frame.shape[:2]
    s = SEND_WIDTH / w
    small = cv2.resize(frame, (SEND_WIDTH, int(h * s))) if s < 1 else frame
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def analyse_chunk(client, model, chunk):
    nums = [i for i, _ in chunk]
    contents = [PROMPT.format(frames=nums)]
    for i, f in chunk:
        contents.append(f"Frame {i}:")
        contents.append(types.Part.from_bytes(data=encode(f), mime_type="image/jpeg"))
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ChunkResult,
        temperature=0,
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=cfg)
            return resp.parsed or ChunkResult(**json.loads(resp.text))
        except Exception as e:  # network / quota / malformed JSON: retry, then give up on chunk
            if any(t in str(e) for t in ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                sys.exit(f"Gemini rejected the API key: {e}")
            print(f"    chunk {nums[0]}-{nums[-1]} attempt {attempt + 1} failed: {e}", file=sys.stderr)
    return ChunkResult(bats=[], impact_frames=[])


def to_pixels(box, w, h):
    ymin, xmin, ymax, xmax = box
    return np.array([xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h])


def interpolate(samples, n_frames, max_gap):
    """samples: {frame: box or None}. Linearly interpolate between consecutive detected samples
    if they are at most `max_gap` frames apart; otherwise leave the frames empty."""
    boxes = [None] * n_frames
    keys = sorted(samples)
    for a, b in zip(keys, keys[1:] + [None]):
        if samples[a] is not None:
            boxes[a] = samples[a]
        if b is not None and samples[a] is not None and samples[b] is not None and b - a <= max_gap:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                boxes[k] = (1 - t) * samples[a] + t * samples[b]
    if keys and samples[keys[-1]] is not None:
        boxes[keys[-1]] = samples[keys[-1]]
    return boxes


def merge_impacts(frames, min_gap):
    out = []
    for f in sorted(set(frames)):
        if not out or f - out[-1] >= min_gap:
            out.append(f)
    return out


def render(video, out_path, boxes, impacts, hold):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t = max(2, round(h / 300))
    scale = h / 720
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < len(boxes) and boxes[i] is not None:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), t)
            cv2.putText(frame, "bat", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, (0, 255, 0), t)
        cv2.putText(frame, f"frame {i}", (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (255, 255, 255), t)
        recent = [f for f in impacts if 0 <= i - f < hold]
        if recent:
            cv2.putText(frame, f"IMPACT @ frame {recent[-1]}", (10, int(70 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 255), t + 1)
            if i == recent[-1]:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6 * t)
        wr.write(frame)
        i += 1
    cap.release()
    wr.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyse every Nth frame (boxes are interpolated between); use 1 for short clips")
    ap.add_argument("--chunk", type=int, default=24, help="frames per Gemini request")
    ap.add_argument("--overlap", type=int, default=4, help="frames shared between consecutive chunks")
    a = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set the GEMINI_API_KEY environment variable first.")
    client = genai.Client(api_key=key)

    video = Path(a.video)
    out = Path(a.output) if a.output else video.with_name(video.stem + "_gemini.mp4")
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    frames = list(read_sampled(cap, a.stride))
    cap.release()
    n_total = max(n_total, frames[-1][0] + 1 if frames else 0)
    print(f"{video.name}: {n_total} frames, analysing {len(frames)} (stride {a.stride})")

    samples, impact_raw = {}, []
    step = max(1, a.chunk - a.overlap)
    for s in range(0, len(frames), step):
        chunk = frames[s:s + a.chunk]
        print(f"  frames {chunk[0][0]}-{chunk[-1][0]}", flush=True)
        res = analyse_chunk(client, a.model, chunk)
        valid = {i for i, _ in chunk}
        for b in res.bats:
            if b.frame in valid:
                ok = b.visible and len(b.box_2d) == 4 and b.box_2d[2] > b.box_2d[0] and b.box_2d[3] > b.box_2d[1]
                samples[b.frame] = to_pixels(b.box_2d, w, h) if ok else None
        impact_raw += [f for f in res.impact_frames if f in valid]
        if s + a.chunk >= len(frames):
            break

    if not samples:
        sys.exit("Gemini returned no results for any chunk; not writing a video.")
    boxes = interpolate(samples, n_total, max_gap=a.stride * 2)
    impacts = merge_impacts(impact_raw, min_gap=max(10, 3 * a.stride))
    print("Rendering...")
    render(video, out, boxes, impacts, hold=20)

    print(f"Bat boxed in {sum(b is not None for b in boxes)}/{n_total} frames")
    print(f"Impact frames: {impacts if impacts else 'none detected'}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
