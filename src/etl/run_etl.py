"""Brazil school-risk ETL entrypoint (Fundamental + Médio)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.build_school_risk_marts import build_school_risk_br, build_school_risk_br_level
from src.etl.config import discover_brasil_years, ensure_dirs
from src.etl.utils import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Brazil school-risk marts (Fundamental + Médio)"
    )
    parser.add_argument(
        "--brasil-years",
        type=int,
        nargs="*",
        default=None,
        help="Census / rendimento years (default: auto intersection of staged rendimento + Censo)",
    )
    parser.add_argument(
        "--level",
        choices=["fundamental", "medio", "both"],
        default="both",
        help="Which education-level mart(s) to build",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    years = args.brasil_years if args.brasil_years else discover_brasil_years(min_year=2016)
    print("Years:", years)

    t0 = time.time()
    report: dict = {"years": years, "steps": {}}

    print("== School risk BR (official INEP abandonment) ==")
    t = time.time()
    if args.level == "both":
        outs = build_school_risk_br(years=years)
    else:
        outs = {args.level: build_school_risk_br_level(args.level, years=years)}
    report["steps"]["school_risk_br"] = {
        "seconds": round(time.time() - t, 2),
        "outputs": {k: str(v) for k, v in outs.items()},
    }
    for k, v in outs.items():
        print(f"  {k}: {v}")

    report["total_seconds"] = round(time.time() - t0, 2)
    write_json(report, PROJECT_ROOT / "latam_education_data" / "dq" / "etl_run_report.json")
    print(f"Done in {report['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
