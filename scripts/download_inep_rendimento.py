"""Download INEP Taxas de Rendimento Escolar (school-level) for 2016→latest available."""

from __future__ import annotations

import ssl
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latam_education_data" / "02_national" / "brasil" / "taxas_rendimento"
OUT.mkdir(parents=True, exist_ok=True)

CTX = ssl._create_unverified_context()
UA = {"User-Agent": "Mozilla/5.0 (compatible; LivingStoneFoundation/1.0)"}

# Probe through next calendar year in case INEP already published a late release.
YEAR_START = 2016
YEAR_END = datetime.now().year + 1


def year_urls(year: int) -> list[str]:
    base = "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais"
    return [
        f"{base}/{year}/tx_rend_escolas_{year}.zip",
        f"{base}/{year}/TX_REND_ESCOLAS_{year}.zip",
        f"{base}/{year}/TX_RENDIMENTO_ESCOLAS_{year}.zip",
        f"{base}/{year}/taxas_rendimento/TX_REND_ESCOLAS_{year}.zip",
        f"{base}/{year}/tx_rendimento_escolas_{year}.zip",
    ]


def fetch(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180, context=CTX) as resp:
            data = resp.read()
        if len(data) < 1000:
            print(f"TOO_SMALL {len(data)} {url}")
            return False
        dest.write_bytes(data)
        print(f"OK {len(data):,} bytes -> {dest.name}")
        return True
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc} :: {url}")
        return False


def download_year(year: int) -> Path | None:
    dest = OUT / f"tx_rend_escolas_{year}.zip"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"SKIP exists {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    for url in year_urls(year):
        if fetch(url, dest):
            extract_dir = OUT / f"tx_rend_escolas_{year}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(extract_dir)
            print(f"  extracted -> {extract_dir}")
            return dest
    if dest.exists() and dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
    return None


def main() -> None:
    ok: list[int] = []
    missing: list[int] = []
    for year in range(YEAR_START, YEAR_END + 1):
        print(f"\n=== {year} ===")
        path = download_year(year)
        if path is None:
            missing.append(year)
        else:
            # Ensure extracted even when zip was skipped as already present
            extract_dir = OUT / f"tx_rend_escolas_{year}"
            xlsx_hits = list(extract_dir.rglob("*.xlsx")) + list(extract_dir.rglob("*.xls"))
            if not xlsx_hits and path.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
                print(f"  extracted -> {extract_dir}")
            ok.append(year)

    print("\n========== SUMMARY ==========")
    print("available years:", ok)
    print("missing years:  ", missing)
    summary = OUT / "download_summary.txt"
    summary.write_text(
        f"available={ok}\nmissing={missing}\n",
        encoding="utf-8",
    )
    print("wrote", summary)


if __name__ == "__main__":
    main()
