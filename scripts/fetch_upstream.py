#!/usr/bin/env python3
"""Fetch the exact public upstream revision; no mutable branch is used."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from floworaclebench.potentials import UPSTREAM_COMMIT

destination = ROOT / "vendor" / "tnwf-paper"
if destination.is_dir() and (destination / "src/tnwf/mps_v/model.py").is_file():
    print(f"Source already present: {destination}")
else:
    subprocess.run(["git", "clone", "https://github.com/sygaldry-tech/tnwf-paper", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    print(f"Fetched upstream {UPSTREAM_COMMIT}")
