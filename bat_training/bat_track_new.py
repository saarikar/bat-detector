"""Bat tracking + impact with the NEW model (single class: 0 = bat, trained at 1280).

Usage: python bat_track_new.py ..\2.mp4 [-o out.mp4] [any bat_track.py option to override a default]
"""
import sys
from pathlib import Path

import bat_track

HERE = Path(__file__).resolve().parent
DEFAULTS = [
    "--bat-model", str(HERE / "runs" / "bat" / "weights" / "best.pt"),
    "--bat-class", "0",
    "--imgsz", "1280",
    "--bat-conf", "0.4",
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]  # later flags win, so user options override
    bat_track.main()
