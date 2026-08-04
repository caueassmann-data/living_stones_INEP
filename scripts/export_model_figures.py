"""Export static PNG figures for model results (Slack / reports)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_viz import export_level_figures
from src.utils import EDUCATION_LEVELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export model-result PNG figures")
    parser.add_argument(
        "--level",
        choices=["fundamental", "medio", "both"],
        default="both",
    )
    args = parser.parse_args(argv)
    levels = list(EDUCATION_LEVELS) if args.level == "both" else [args.level]
    for level in levels:
        paths = export_level_figures(level)
        print(f"[{level}]")
        for p in paths:
            print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
