#!/usr/bin/env python3
"""Create the source/data release archive with per-file SHA-256 provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
ARCHIVE_ROOT = "qflow-compiler"
MANIFEST = "release_manifest.json"


def release_paths(root):
    """Use the Git index, or the exact manifest membership in a release ZIP."""
    if (root / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        names = [name for name in tracked.decode().split("\0") if name]
    else:
        manifest = json.loads((root / MANIFEST).read_text())
        names = [item["path"] for item in manifest["files"]]
    paths = []
    for name in sorted(set(names) - {MANIFEST}):
        parts = Path(name).parts
        if parts and parts[0].casefold() == "report":
            raise ValueError(f"Private report path is excluded from release archives: {name}")
        path = root / name
        if not path.is_file() or path.resolve() != path or not path.is_relative_to(root):
            raise ValueError(f"Release path must be a regular file under the project root: {name}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"QFlow-Compiler-v{VERSION}.zip")
    args = parser.parse_args()
    files = release_paths(ROOT)
    manifest_path = ROOT / MANIFEST
    if args.output.resolve() in {*files, manifest_path}:
        parser.error("--output must be separate from all release inputs")
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {"project": "QFlow Compiler", "version": VERSION, "archive_root": ARCHIVE_ROOT,
        "upstream_commit": "11358d3dca52193f93f037046229797296a2063e",
        "note": "This manifest excludes its own digest. Experimental T counts are model estimates; see docs/resource_model.md.",
                "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in payloads.items()]}
    payloads[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode()
    manifest_path.write_bytes(payloads[MANIFEST])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}")
            # ZIP extraction does not require executable scripts; the docs invoke Bash/Python.
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"archive": str(args.output.resolve()), "files": len(payloads),
                      "bytes": args.output.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
