#!/usr/bin/env python3
"""Compare v0.3.1 results with the preserved v0.3.0 release."""
import argparse
from contextlib import contextmanager
import io
import subprocess
import tarfile
import tempfile
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tomllib
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

@contextmanager
def baseline_source(path):
    if path is not None:
        yield path.resolve(), "supplied v0.3.0 checkout"
        return
    try:
        archive = subprocess.check_output(["git", "archive", "v0.3.0"], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("Pass --source /path/to/extracted-v0.3.0 for the historical comparison.") from exc
    with tempfile.TemporaryDirectory(prefix="floworacle-v030-") as temp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(temp, filter="data")
        yield Path(temp), "git tag v0.3.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Extracted v0.3.0 release; defaults to the local Git tag")
    args = parser.parse_args()
    with baseline_source(args.source) as (source, reference):
        verify(source, reference)


def verify(source, reference):
    old=source/'results/v03'; new=ROOT/'results/v03'
    comparisons={}
    for name in ['oracles.csv','oracles_pareto.csv','oracles_qubit_caps.csv','preparation_midpoints.csv']:
        a=list(csv.DictReader((old/name).open())); b=list(csv.DictReader((new/name).open()))
        assert len(a)==len(b)
        keys=set(a[0])&set(b[0])
        diff=sorted({k for x,y in zip(a,b) for k in keys if x[k]!=y[k] and 'seconds' not in k})
        assert set(diff)<= {'per_rotation_error'}, (name,diff)
        comparisons[name]={'rows':len(b),'changed_existing_nontiming_fields':diff,'added_fields':sorted(set(b[0])-set(a[0]))}
    original=json.loads((source/'release_manifest.json').read_text())
    for entry in original['files']:
        assert hashlib.sha256((source/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    oldprep=json.loads((old/'preparation.json').read_text()); prep=json.loads((new/'preparation.json').read_text())
    for a,b in zip(oldprep['cases'],prep['cases'],strict=True):
        for x,y in zip(a['methods'],b['methods'],strict=True):
            for account,budget in [('per_A_and_inverse_resource_accounting','preparation_synthesis_budget'),('repeated_amplification_accounting','repeated_synthesis_budget')]:
                xr=x[account]; yr=y[account]
                for key in ['t_count_estimate','logical_qubits','native_gates','native_gate_counts']:
                    assert xr[key]==yr[key],(account,key)
                allocation=Fraction(yr['allocated_rotation_synthesis_budget_exact'])
                assert allocation==Fraction(str(prep['config'][budget]))
                assert Fraction(yr['used_rotation_synthesis_allowance_exact'])<=allocation
            assert x['native_state_and_tv_bound_exact']==y['native_state_and_tv_bound_exact']
    tests=ElementTree.parse(new/'pytest.xml')
    counts={key:sum(int(s.attrib.get(key,0)) for s in tests.iter('testsuite')) for key in ['tests','failures','errors','skipped']}
    assert counts["tests"] >= 150 and not any(counts[k] for k in ("failures", "errors", "skipped")), counts
    result={'status':'passed','tests':counts,'original_source_manifest_entries_unchanged':len(original['files']),'comparisons':comparisons,'preparation_resources_and_native_bounds_unchanged':True,'exact_aggregate_synthesis_allocations_verified':True,'budget_convention':'exact decimal string, matching conditional certificate','replayed_v02_in_this_audit':False,'baseline':reference,'release':VERSION}
    (new/'audit_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
