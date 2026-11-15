# Cricket bat detection and impact tracking

A single-class YOLO bat detector, with Gemini-assisted labelling, plus a script that tracks the bat
through a video and marks the frame where the bat hits the ball.

## Setup
```
pip install -r requirements.txt
```
Gemini scripts read the key from the environment (never put it in code):
```
$env:GEMINI_API_KEY="your_key"
```

## Training pipeline (`bat_training/`)
1. `1_extract_frames.py`   sample frames from videos
2. `2_prelabel_gemini.py`  pre-label frames with Gemini (review the labels by hand afterwards)
3. `3_make_dataset.py`     build the YOLO dataset, split by source video
4. `4_train.py`            train (use a GPU, e.g. Colab)
5. `5_evaluate.py`         evaluate on held-out videos

## Track the bat and find the impact
```
python bat_training/bat_track.py input.mp4 --bat-model bat_training/runs/bat/weights/best.pt
```
Writes `<video>_tracked.mp4` and `<video>_tracked_impact.jpg`. The ball is detected with stock COCO `yolov8m.pt`.

## Other
`cricket_gemini.py` runs Gemini directly on video chunks to detect the bat.

`best.pt` was trained on a small, partly reviewed dataset; treat it as a baseline.
