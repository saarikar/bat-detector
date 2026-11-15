"""Track the bat through a cricket video and mark the frame where the bat hits the ball.

Models:
  - bat  : your trained single-class detector (runs/bat/weights/best.pt, class 0 = bat)
  - ball : stock COCO yolov8m.pt (class 32 = sports ball); downloaded automatically

Pass 1  detect bat + ball on every frame
Pass 2  bat: pick the detection that continues the track, fill gaps by interpolation, smooth
        ball: link detections into a trajectory
        impact: ball is near the bat AND its path changes sharply (direction or speed)
Pass 3  write the annotated video + a still of the impact frame

Bat box colours: green = detected, yellow = interpolated/held (bat not detected on that frame).

Usage:
  python bat_track.py input.mp4
  python bat_track.py input.mp4 -o out.mp4 --bat-model runs/bat/weights/best.pt
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COCO_BALL = 32


# ---------------------------------------------------------------- pass 1
def plausible_bat(d, w, h, ignore_bottom, min_aspect, max_area):
    """Reject boxes that cannot be a bat: scoreboard strip, too square, or too large."""
    x1, y1, x2, y2, _ = d
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    if (y1 + y2) / 2 > h * (1 - ignore_bottom):  # centre inside the bottom (scoreboard) strip
        return False
    if max(bw, bh) / min(bw, bh) < min_aspect:  # a bat is long and thin
        return False
    return bw * bh <= max_area * w * h


def detect(video, bat_model, ball_model, imgsz, bat_conf, ball_conf,
           ignore_bottom=0.12, min_aspect=1.5, max_area=0.10, bat_class=None):
    bm, lm = YOLO(bat_model), YOLO(ball_model)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(3)), int(cap.get(4))
    bats, balls, i = [], [], 0  # per frame: list of (x1,y1,x2,y2,conf) | list of (cx,cy,conf)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = bm.predict(frame, imgsz=imgsz, conf=bat_conf, classes=bat_class, verbose=False)[0]
        dets = [(*b.xyxy[0].tolist(), float(b.conf)) for b in r.boxes]
        bats.append([d for d in dets if plausible_bat(d, fw, fh, ignore_bottom, min_aspect, max_area)])
        r = lm.predict(frame, imgsz=imgsz, conf=ball_conf, classes=[COCO_BALL], verbose=False)[0]
        fb = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            fb.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf)))
        balls.append(fb)
        i += 1
        if i % 25 == 0 or i == n:
            print(f"  [detect] {i}/{n} frames ({100 * i // max(n, 1)}%)", flush=True)
    cap.release()
    return bats, balls


# ---------------------------------------------------------------- pass 2: bat
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_bat(bats):
    """One bat per frame: prefer the detection that continues the previous box, else the most confident."""
    chosen, prev = [], None
    for dets in bats:
        if not dets:
            chosen.append(None)
            continue
        best = max(dets, key=lambda d: d[4] + (0.5 * _iou(d, prev) if prev is not None else 0))
        chosen.append(np.array(best[:4]))
        prev = best
    return chosen


def fill_and_smooth(chosen, max_gap=15, alpha=0.6):
    """Interpolate gaps up to max_gap frames, EMA-smooth detected boxes.
    Returns (boxes, detected_flags). boxes[i] is None only where the bat is lost for > max_gap frames."""
    n = len(chosen)
    boxes = [None if c is None else c.copy() for c in chosen]
    detected = [c is not None for c in chosen]
    idx = [i for i, c in enumerate(chosen) if c is not None]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                boxes[k] = (1 - t) * chosen[a] + t * chosen[b]
    # smooth detected boxes only (keeps interpolated ones consistent with their neighbours)
    last = None
    for i in range(n):
        if boxes[i] is None:
            last = None
        elif detected[i]:
            last = boxes[i] if last is None else alpha * boxes[i] + (1 - alpha) * last
            boxes[i] = last.copy()
        else:
            last = boxes[i]
    return boxes, detected


# ---------------------------------------------------------------- pass 2: ball + impact
def track_ball(balls, max_jump):
    track, prev, vel, miss = [], None, np.zeros(2), 0
    for fb in balls:
        pick = None
        if fb:
            if prev is None:
                pick = max(fb, key=lambda d: d[2])
            else:
                pred = prev + vel
                cand = min(fb, key=lambda d: math.hypot(d[0] - pred[0], d[1] - pred[1]))
                if math.hypot(cand[0] - pred[0], cand[1] - pred[1]) <= max_jump * (1 + miss):
                    pick = cand
        if pick is not None:
            p = np.array(pick[:2])
            if prev is not None:
                vel = (p - prev) / (1 + miss)
            prev, miss = p, 0
            track.append(p)
        else:
            miss += 1
            if miss > 10:
                prev, vel = None, np.zeros(2)
            track.append(None)
    return track


def _dist_to_box(p, box, margin=0):
    dx = max(box[0] - margin - p[0], 0, p[0] - (box[2] + margin))
    dy = max(box[1] - margin - p[1], 0, p[1] - (box[3] + margin))
    return math.hypot(dx, dy)


def find_impact(bat_boxes, track, diag, near_frac=0.06, min_angle=35, min_ratio=2.5):
    """Return (frame, 'sharp'|'closest') or (None, None).
    sharp   : ball within near_frac*diag of the bat box and its velocity direction/speed changes sharply
    closest : fallback, the frame where the ball is nearest to the bat box (lower confidence)"""
    margin = near_frac * diag
    idx = [i for i, p in enumerate(track) if p is not None]
    best, best_score = None, 0.0
    for k in range(1, len(idx) - 1):
        i0, i1, i2 = idx[k - 1], idx[k], idx[k + 1]
        if i2 - i0 > 8 or bat_boxes[i1] is None:
            continue
        v_in = (track[i1] - track[i0]) / (i1 - i0)
        v_out = (track[i2] - track[i1]) / (i2 - i1)
        s_in, s_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if max(s_in, s_out) < 2:
            continue
        cos = np.dot(v_in, v_out) / (s_in * s_out + 1e-9)
        angle = math.degrees(math.acos(np.clip(cos, -1, 1)))
        ratio = max(s_in, s_out) / (min(s_in, s_out) + 1e-9)
        d = _dist_to_box(track[i1], bat_boxes[i1])
        if d > margin or not (angle >= min_angle or ratio >= min_ratio):
            continue
        score = angle / 180 + min(ratio, 10) / 10 + (1 - d / margin)
        if score > best_score:
            best, best_score = i1, score
    if best is not None:
        return best, "sharp"
    cands = [(i, _dist_to_box(track[i], bat_boxes[i])) for i in idx if bat_boxes[i] is not None]
    if cands:
        i, d = min(cands, key=lambda c: c[1])
        if d <= 2 * margin:
            return i, "closest"
    return None, None


# ---------------------------------------------------------------- pass 3
def render(video, out_path, bat_boxes, detected, track, impact, kind, freeze, hold, still_path):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    n = len(bat_boxes)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = bat_boxes[i]
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            col = (0, 255, 0) if detected[i] else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, "bat" if detected[i] else "bat (interp)", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if track[i] is not None:
            trail.append(tuple(int(v) for v in track[i]))
        trail = trail[-25:]
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, (0, 165, 255), 2)
        if trail and track[i] is not None:
            cv2.circle(frame, trail[-1], 5, (0, 165, 255), -1)

        cv2.putText(frame, f"frame {i}/{n - 1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if impact is not None and 0 <= i - impact < hold:
            label = "IMPACT" if kind == "sharp" else "IMPACT (closest approach)"
            cv2.putText(frame, f"{label} @ frame {impact}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        if i == impact:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            if track[i] is not None:
                cv2.circle(frame, tuple(int(v) for v in track[i]), 18, (0, 0, 255), 3)
            cv2.imwrite(str(still_path), frame)
            for _ in range(freeze):
                wr.write(frame)
        wr.write(frame)
        i += 1
        if i % 50 == 0 or i == n:
            print(f"  [render] {i}/{n} frames ({100 * i // max(n, 1)}%)", flush=True)
    cap.release()
    wr.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--output")
    ap.add_argument("--bat-model", default="runs/bat/weights/best.pt")
    ap.add_argument("--ball-model", default="yolov8m.pt", help="COCO model used for the ball")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--bat-conf", type=float, default=0.4, help="min confidence for a bat detection")
    ap.add_argument("--bat-class", type=int, nargs="+", default=None,
                    help="class id(s) of the bat in the bat model (old model: 1). Default: all classes")
    ap.add_argument("--ignore-bottom", type=float, default=0.12,
                    help="ignore bat boxes centred in the bottom fraction of the frame (scoreboard)")
    ap.add_argument("--min-aspect", type=float, default=1.5, help="min long/short side ratio of a bat box")
    ap.add_argument("--max-area", type=float, default=0.10, help="max bat box area as fraction of the frame")
    ap.add_argument("--ball-conf", type=float, default=0.10)
    ap.add_argument("--max-gap", type=int, default=15, help="interpolate bat over gaps up to this many frames")
    ap.add_argument("--freeze", type=int, default=12, help="extra frames the impact frame is held for")
    ap.add_argument("--hold", type=int, default=30, help="frames the IMPACT label stays on screen")
    a = ap.parse_args()

    video = Path(a.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if not Path(a.bat_model).exists():
        raise SystemExit(f"Bat model not found: {a.bat_model} (use --bat-model)")
    out = Path(a.output) if a.output else video.with_name(video.stem + "_tracked.mp4")
    still = out.with_name(out.stem + "_impact.jpg")

    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    diag = math.hypot(w, h)

    print("Pass 1/3: detecting bat and ball")
    bats, balls = detect(video, a.bat_model, a.ball_model, a.imgsz, a.bat_conf, a.ball_conf,
                         a.ignore_bottom, a.min_aspect, a.max_area, a.bat_class)

    print("Pass 2/3: tracking bat, ball and finding impact")
    bat_boxes, detected = fill_and_smooth(pick_bat(bats), max_gap=a.max_gap)
    track = track_ball(balls, max_jump=0.08 * diag)
    impact, kind = find_impact(bat_boxes, track, diag)

    print("Pass 3/3: rendering")
    render(video, out, bat_boxes, detected, track, impact, kind, a.freeze, a.hold, still)

    n = len(bat_boxes)
    print(f"Bat detected in {sum(detected)}/{n} frames; tracked (incl. interpolated) in "
          f"{sum(b is not None for b in bat_boxes)}/{n}")
    print(f"Ball tracked in {sum(t is not None for t in track)}/{n} frames")
    if impact is None:
        print("Impact: not found (no ball track near the bat). Try lowering --ball-conf.")
    else:
        note = "" if kind == "sharp" else "  [low confidence: closest approach, no sharp ball deflection seen]"
        print(f"Impact: frame {impact} ({impact / (cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30):.2f}s){note}")
        print(f"Impact still: {still}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
