"""Bat tracking + impact with the OLD model (classes: 0 = Batsman, 1 = cricket-bats; the bat is class 1).

Expects the old weights at ..\best.pt (the best.pt in downloads, next to the bat_training folder).
Override with --bat-model.

Usage: python bat_track_old.py ..\2.mp4 [-o out.mp4] [any bat_track.py option to override a default]
"""
import sys
from pathlib import Path

import bat_track

HERE = Path(__file__).resolve().parent
DEFAULTS = [
    "--bat-model", str(HERE.parent / "best.pt"),
    "--bat-class", "1",
    "--imgsz", "640",
    "--bat-conf", "0.25",
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]  # later flags win, so user options override
    bat_track.main()
