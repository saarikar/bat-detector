# Cricket Bat Detection and Impact Tracking

Detect a cricket bat in broadcast footage, track it through a video, and mark the frame where the bat
meets the ball. The project has three parts:

1. **A training pipeline** that turns raw videos into a single-class YOLO bat detector, using Gemini to
   pre-label frames (you review the labels by hand before training).
2. **Tracking scripts** that run a trained bat model over a video, draw the bat box on every frame, follow
   the ball, and annotate the impact frame.
3. **A Gemini-only detector** (`cricket_gemini.py`) that sends video chunks straight to the Gemini API.

> **Status: baseline.** The included `best.pt` was trained on a small dataset (about 70 training frames,
> labelled by Gemini and not fully reviewed). It misses bats and can fire on the scoreboard. Impact timing
> depends on ball detection, which is currently the weakest part. See [Known limitations](#known-limitations).

---

## Repository layout

```
bat-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── cricket_gemini.py              # Gemini API: detect the bat directly on video chunks
└── bat_training/
    ├── 1_extract_frames.py        # step 1: sample frames from videos
    ├── 2_prelabel_gemini.py       # step 2: pre-label frames with Gemini
    ├── 3_make_dataset.py          # step 3: build the YOLO dataset (split by source video)
    ├── 4_train.py                 # step 4: train the detector
    ├── 5_evaluate.py              # step 5: evaluate on held-out videos
    ├── bat_track.py               # core: bat tracking + impact detection + rendering
    ├── bat_track_new.py           # wrapper preset for the new single-class model
    ├── bat_track_old.py           # wrapper preset for the old two-class model (weights not included)
    └── runs/bat/weights/best.pt   # new model weights (class 0 = bat)
```

Not in the repo (see `.gitignore`): videos (`*.mp4`), the `data/` folder (frames, labels, YOLO dataset),
and `yolov8m.pt` (downloaded automatically by ultralytics).

---

## Setup

Requires Python 3.10+.

```powershell
pip install -r requirements.txt
```

Dependencies: `ultralytics` (pulls in PyTorch), `opencv-python`, `numpy`, `pyyaml`, `google-genai`, `pydantic`.

**API keys.** The Gemini scripts read the key from the environment. Never put a key in code or commit it.

```powershell
$env:GEMINI_API_KEY="your_key"      # PowerShell, current window only
```

`GOOGLE_API_KEY` is also accepted. If both are set, `GEMINI_API_KEY` wins.

---

## 1. Training pipeline

Run from the `bat_training/` folder. Put your source videos somewhere convenient (the examples use `..\`).

### Step 1: extract frames
```powershell
python 1_extract_frames.py ..\0.mp4 ..\1.mp4 ..\2.mp4 --every 5 --out data/images_all
```
- Looks at every Nth frame (`--every`) and skips near-duplicates (`--min-diff`, default 4.0).
- Frames are named `<video>__<frame>.jpg`. Step 3 uses the video name to split train and validation.
- Frames wider than `--max-width` (default 1920) are downscaled.

### Step 2: pre-label with Gemini (optional, then review)
```powershell
$env:GEMINI_API_KEY="your_key"
python 2_prelabel_gemini.py --images data/images_all --out data/labels_all
```
- Writes one YOLO `.txt` per frame to `data/labels_all` (class 0 = bat). Frames with **no bat get an empty
  file**, which is what teaches the model to ignore crowds, scoreboards and grass.
- Prints one progress line per frame: `[37/202] 3__000020.jpg: 1 bat(s) | 12 frames with a bat so far`.
- Resumable: frames that already have a `.txt` are skipped, so re-running continues where it stopped.
  Frames that fail all retries are left unlabelled and retried on the next run.
- **Free-tier quota.** Free keys allow only about 20 requests per day per model, so 200 frames will not
  finish in one day. Options: wait a day, enable billing, or use another model with `--model <name>`.
  Quota is counted per Google Cloud project, so a new key in the same project does not reset it.
- **You must review these labels** in CVAT, Roboflow or labelImg before training. Gemini boxes are a
  starting point, not ground truth.

### Step 3: build the dataset
```powershell
python 3_make_dataset.py --val-videos 5 --images data/images_all --labels data/labels_all
```
- Splits by **source video**, not by frame. Neighbouring frames are near-duplicates, and a random frame
  split leaks them into validation and inflates the score. (An earlier model reported mAP50 0.995 and then
  failed on new footage.)
- Videos named in `--val-videos` go entirely to validation. Unlabelled frames are skipped.
- Writes `data/yolo/` and `data/yolo/data.yaml`, and rebuilds the folder each time it runs.
- Pick a validation video that has both bat and bat-free frames, otherwise the false-alarm check is meaningless.

### Step 4: train (needs a GPU)
```powershell
python 4_train.py --data data/yolo/data.yaml --model yolov8m.pt --epochs 100 --device 0
```
- Single class (`bat`), started from COCO-pretrained weights, `imgsz 1280` (bats are thin and small).
- Blur, scale and colour augmentation; early stopping after 25 epochs without improvement.
- Weights are saved to `runs/bat/weights/best.pt`.
- 1280 px on CPU is impractically slow. Use Google Colab or another CUDA machine.

**Training on Colab**
1. Zip the dataset: `Compress-Archive -Path data\yolo -DestinationPath yolo.zip -Force`
2. In Colab choose Runtime, Change runtime type, **T4 GPU**. Check with `import torch; print(torch.cuda.is_available())`;
   it must print `True`, or training fails with `Invalid CUDA 'device=0'`.
3. `!pip install ultralytics`, then upload `yolo.zip` and `4_train.py` and run `!unzip -q yolo.zip`.
4. `data.yaml` contains your Windows path. Rewrite it for Colab:
   ```python
   import yaml
   cfg = {"path": "/content/yolo", "train": "images/train", "val": "images/val", "names": {0: "bat"}}
   yaml.safe_dump(cfg, open("/content/yolo/data.yaml", "w"))
   ```
5. `!python 4_train.py --data /content/yolo/data.yaml --device 0` (add `--batch 4` if you run out of memory).
6. Download only the weights: `files.download("runs/bat/weights/best.pt")`, then place it at
   `bat_training/runs/bat/weights/best.pt`. Colab wipes its files when the session ends.

### Step 5: evaluate
```powershell
python 5_evaluate.py --weights runs/bat/weights/best.pt --data data/yolo/data.yaml
```
Reports mAP, precision and recall on the validation split, plus two practical numbers:
- **Recall frames**: share of bat-containing validation frames where a bat was found (target 90% or more).
- **False-alarm rate**: share of bat-free validation frames where a bat box fired (target 5% or less).

Also look at the prediction images in `runs/bat/` by eye. Frame-level "found" can hide wrong boxes.

---

## 2. Track the bat and find the impact

Three entry points, all running the same logic in `bat_track.py`:

| Script | Model | Bat class | imgsz | Bat conf |
|---|---|---|---|---|
| `bat_track_new.py` | `runs/bat/weights/best.pt` | 0 (`bat`) | 1280 | 0.4 |
| `bat_track_old.py` | `../best.pt` (old model, not in repo) | 1 (`cricket-bats`) | 640 | 0.25 |
| `bat_track.py` | any (`--bat-model`) | all unless `--bat-class` | 1280 | 0.4 |

The old model has two classes, `0 = Batsman` (the person) and `1 = cricket-bats`. It must be run with
`--bat-class 1`, which `bat_track_old.py` does for you.

```powershell
cd bat_training
python bat_track_new.py ..\2.mp4 -o ..\2_new.mp4
python bat_track_old.py ..\2.mp4 -o ..\2_old.mp4
```

Any option below overrides the wrapper's default. Do **not** type the square brackets from usage text,
they only mean "optional".

### What it does
1. **Detect.** Your bat model finds bats on every frame. The stock COCO `yolov8m.pt` (class 32, sports ball)
   finds the ball, because the bat models have no ball class. It downloads automatically.
2. **Filter the bat boxes.** Boxes are dropped if their centre is in the bottom strip (scoreboard), if they
   are too square (a bat is long and thin), or if they are too large.
3. **Track the bat.** One box per frame, preferring the detection that continues the previous box. Gaps of up
   to `--max-gap` frames are filled by interpolation (motion blur during the swing), and boxes are smoothed.
4. **Track the ball.** Detections are linked into a trajectory (nearest to the predicted position).
5. **Find the impact.** The frame where the ball is near the bat box and its direction or speed changes
   sharply. If no sharp change is found it falls back to the frame of closest approach and labels it
   `IMPACT (closest approach)`, which is lower confidence.
6. **Render.** Writes the annotated video and a still of the impact frame.

### Output
- `<video>_tracked.mp4` (or `-o`): annotated video.
- `<video>_tracked_impact.jpg`: still of the impact frame.
- Green box = bat detected on that frame. Yellow box labelled `bat (interp)` = filled in by interpolation.
  No box = bat lost for longer than `--max-gap`.
- Orange trail = tracked ball. On the impact frame: red border, a circle on the ball, and the frame is held
  for `--freeze` extra frames; the `IMPACT @ frame N` label stays for `--hold` frames.
- The terminal prints bat and ball coverage and the impact frame with its timestamp.

### Options

| Option | Default | Meaning |
|---|---|---|
| `-o`, `--output` | `<video>_tracked.mp4` | output video path |
| `--bat-model` | per wrapper | bat detector weights |
| `--bat-class` | all | class id(s) to treat as the bat |
| `--ball-model` | `yolov8m.pt` | COCO model used for the ball |
| `--imgsz` | 1280 (640 old) | inference size |
| `--bat-conf` | 0.4 (0.25 old) | minimum bat confidence |
| `--ball-conf` | 0.10 | minimum ball confidence (try 0.05 if the ball is missed) |
| `--ignore-bottom` | 0.12 | ignore bat boxes centred in the bottom fraction of the frame |
| `--min-aspect` | 1.5 | minimum long/short side ratio of a bat box |
| `--max-area` | 0.10 | maximum box area as a fraction of the frame |
| `--max-gap` | 15 | interpolate the bat over gaps up to this many frames |
| `--freeze` | 12 | extra frames to hold the impact frame |
| `--hold` | 30 | frames the impact label stays on screen |

To see raw, unfiltered model output (including wrong boxes):
```powershell
python bat_track_new.py ..\2.mp4 --bat-conf 0.1 --min-aspect 1.0 --max-area 0.3 --ignore-bottom 0
```

Detection runs two models per frame, so it is slow on CPU. Test on a short clip first. For a quicker, less
accurate run, use `--imgsz 960` or `--imgsz 640`.

---

## 3. Gemini-only detection (`cricket_gemini.py`)

Sends overlapping chunks of video frames to Gemini and draws the bat boxes it returns.

```powershell
$env:GEMINI_API_KEY="your_key"
python cricket_gemini.py crick_vid_1.mp4 -o out.mp4 --stride 1 --model <model-name>
```
Options: `-o/--output`, `--model`, `--stride`, `--chunk` (frames per request, default 24), `--overlap`
(frames shared between chunks, default 4). One request is sent per chunk, so free-tier daily limits apply.
To see which models your key can use:
```powershell
python -c "from google import genai; import os; c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); [print(m.name) for m in c.models.list() if 'generateContent' in (m.supported_actions or [])]"
```

---

## Known limitations

- **Small, unreviewed training set.** The new model sees few bats and often misses the real one. In one test
  it boxed the scoreboard and ignored a clearly visible bat. The filters hide the scoreboard case but cannot
  find a missed bat.
- **Ball detection is the weak link for impact timing.** COCO's generic ball class misses small, fast
  cricket balls and sometimes locks onto the batsman's foot or arm. When the ball track is wrong, the impact
  frame is meaningless. A trained ball detector is the main fix.
- **Very short clips** (for example 11 frames) do not give the tracker enough frames to interpolate or to
  measure a change in the ball's path.
- Test on footage the model was **not** trained on, otherwise results look better than they are.

## Improving the model

1. Label the remaining frames and review every box by hand.
2. Add more clips, especially ones with scoreboards, crowds and fielders, and include bat-free frames as
   empty-label examples.
3. Label the ball too (second class or separate model) and use it in place of the COCO ball detector.
4. Hold out a validation video that has both bat and bat-free frames, then retrain and re-run step 5.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `unrecognized arguments: [-o out.mp4] ...` | the square brackets in usage text were typed literally; omit them |
| `Invalid CUDA 'device=0'` | the Colab runtime has no GPU; switch the runtime to T4 and redo the upload steps |
| `429 RESOURCE_EXHAUSTED` | free-tier daily quota used up; wait, enable billing, or use another model or project |
| `503 UNAVAILABLE` | temporary Gemini overload; the script retries automatically |
| `Empty train or val split` | step 3 found no labels for the chosen `--val-videos`; check names and that labels exist |
| `Impact: not found` | no ball track near the bat; try `--ball-conf 0.05`, or the clip is too short |
| `pip.exe` blocked | use `python -m pip install ...` |

## Notes

- Do not commit API keys, videos or the dataset. If a key is ever pasted into a chat, a log or a commit,
  revoke it at https://aistudio.google.com/apikey and create a new one.
- Ultralytics and Gemini model names and free-tier limits change often; check the current documentation.
