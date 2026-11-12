#!/usr/bin/env python3
"""Record exact installed runtime dependencies (including transitive pins).

The generated lock targets the measured Linux/Python 3.12 CPU environment.
It intentionally does not freeze unrelated system-site packages.
"""
import importlib.metadata as metadata
from pathlib import Path
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]
pending=["numpy","scipy","matplotlib","qiskit","torch","scikit-learn","threadpoolctl","pytest","setuptools","pip"]
pins={};environment=default_environment()|{"extra":""}
while pending:
    name=canonicalize_name(pending.pop())
    if name in pins:continue
    distribution=metadata.distribution(name)
    pins[name]=distribution.version
    for dependency in distribution.requires or []:
        requirement=Requirement(dependency)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            pending.append(requirement.name)
header=("# Exact measured Linux/Python 3.12 CPU environment.\n"
        "# Regenerate only when deliberately changing the environment.\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n\n")
(ROOT/"requirements.lock.txt").write_text(header+"\n".join(f"{name}=={pins[name]}" for name in sorted(pins))+"\n")
print(f"Pinned {len(pins)} packages")

