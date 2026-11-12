#!/usr/bin/env python3
"""Re-score saved qualified circuits; no new circuits or physical-runtime claims."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def analyze():
    with (ROOT/'results/v03/oracles.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['experiment']=='matched' and r['meets_target']=='True']
    groups = defaultdict(list)
    for r in rows:
        groups[r['case'],r['tolerance']].append(r)
    scenarios=[]
    for slope in (3,4):
        for offset in (0,10,20):
            decisions=[]
            def cost(r):
                rotations=int(r['arbitrary_rotations']); delta=float(r['per_rotation_error'])
                original=math.ceil(float(r['rotation_model_slope'])*math.log2(1/delta)+float(r['rotation_model_offset']))
                return int(r['t_count_estimate'])+rotations*(math.ceil(slope*math.log2(1/delta)+offset)-original)
            for (case,tolerance), candidates in sorted(groups.items()):
                winner=min(candidates,key=lambda r:(cost(r),int(r['logical_qubits']),r['candidate']))
                decisions.append(dict(case=case,tolerance=float(tolerance),winner=winner['method'],candidate=winner['candidate'],projected_t=cost(winner)))
            scenarios.append(dict(slope=slope,offset=offset,winner_counts=dict(Counter(d['winner'] for d in decisions)),decisions=decisions))
    return dict(scope='Fixed already-qualified candidate menu; re-score T projections only. Does not test actual synthesis, stronger lookup, physical costs, or unseen candidates.',scenarios=scenarios)

if __name__=='__main__':
    out=ROOT/'results/v03/audit_sensitivity.json'
    result=analyze(); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:s[k] for k in ('slope','offset','winner_counts')} for s in result['scenarios']],indent=2))
