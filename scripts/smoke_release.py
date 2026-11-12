#!/usr/bin/env python3
"""Verify release hashes and test an extracted checkout with the supplied runtime.

This checks source/data isolation, not a fresh network dependency installation.
The output directory must not exist, so an earlier checkout cannot mask omissions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def verify_and_extract(archive: Path, destination: Path) -> tuple[Path, dict]:
    """Verify membership and hashes before writing or running extracted files."""
    with zipfile.ZipFile(archive) as source:
        roots = set()
        files = {}
        for member in source.infolist():
            name = member.filename
            path = PurePosixPath(name)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (path.is_absolute() or len(path.parts) < 2 or ".." in path.parts
                    or "\\" in name or name != path.as_posix() or mode not in (0, stat.S_IFREG)):
                raise ValueError(f"Unexpected archive member: {name}")
            roots.add(path.parts[0])
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in files:
                raise ValueError(f"Duplicate archive member: {name}")
            files[relative] = member
        if len(roots) != 1 or not roots <= {"qflow-compiler", "FlowOracleBench"}:
            raise ValueError(f"Unexpected archive root: {roots}")
        if len({name.casefold() for name in files}) != len(files):
            raise ValueError("Archive contains paths that collide on case-insensitive filesystems")
        root = roots.pop()
        manifest = json.loads(source.read(files["release_manifest.json"]))
        paths = {item["path"] for item in manifest["files"]}
        if len(paths) != len(manifest["files"]) or set(files) != paths | {"release_manifest.json"}:
            raise ValueError("Archive and manifest members differ")
        for item in manifest["files"]:
            data = source.read(files[item["path"]])
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Release digest or size differs: {item['path']}")
        destination.mkdir(parents=True, exist_ok=False)
        source.extractall(destination)
    return destination / root, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--record-dir", type=Path, default=ROOT / "dist" / "smoke")
    args = parser.parse_args()
    archive = args.archive.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new extraction directory: {destination}")
    checkout, manifest = verify_and_extract(archive, destination)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(checkout / "src"), str(checkout / "vendor/tnwf-paper/src")])
    env["MPLBACKEND"] = "Agg"
    python = str(args.python.absolute())
    source_check = (
        "from pathlib import Path; import floworaclebench; "
        "p=Path(floworaclebench.__file__).resolve(); "
        "assert p.is_relative_to(Path.cwd()/'src'), str(p); print(p)"
    )
    check_flags = ["--check-only"] if manifest.get("version") == "1" else []
    commands = [
        [python, "-c", source_check],
        [python, "-m", "pytest", "-q", "--junitxml=results/pytest-archive-smoke.xml"],
        [python, "scripts/verify_saved_oracle.py", *check_flags],
        [python, "scripts/validate_results.py", *check_flags],
    ]
    if (checkout / "scripts/verify_v03_artifacts.py").is_file():
        commands.append([python, "scripts/verify_v03_artifacts.py"])
    if (checkout / "scripts/check_release_claims.py").is_file():
        commands.append([python, "scripts/check_release_claims.py"])
    record_dir = args.record_dir.resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, command in enumerate(commands):
        start = time.perf_counter()
        process = subprocess.run(command, cwd=checkout, env=env, text=True, capture_output=True)
        output = process.stdout + process.stderr
        log_path = record_dir / f"archive_smoke_{index}.log"
        log_path.write_text(output)
        print(output, flush=True)
        records.append({"argv": command, "cwd": str(checkout), "exit_code": process.returncode,
                        "runtime_seconds": time.perf_counter() - start,
                        "log": log_path.name})
        process.check_returncode()
    tests = list(ET.parse(checkout / "results/pytest-archive-smoke.xml").getroot().iter("testcase"))
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_under_test": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "manifest_files_verified": len(manifest["files"]),
        "test_count": len(tests),
        "source_import_isolated": True,
        "commands": records,
        "scope": "Freshly extracted source and saved data using the supplied installed runtime dependencies.",
        "tested_file_sha256": {item["path"]: item["sha256"] for item in manifest["files"]},
    }
    (record_dir / "archive_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "manifest_files_verified", "test_count", "source_import_isolated"]}, indent=2))


if __name__ == "__main__":
    main()
