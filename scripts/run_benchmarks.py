#!/usr/bin/env python3
"""Run reproducible circuit experiments, retaining raw JSON and CSV data."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from floworaclebench.benchmark import run_oracle_benchmarks,run_scaling,dump_json,environment_record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/benchmark.json")
    parser.add_argument("--stage",choices=["oracles","scaling","evolution","all"],default="all")
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args();config=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    dump_json(args.output/"config.json",config)
    dump_json(args.output/"environment.json",environment_record())
    start=time.perf_counter()
    if args.stage in ("oracles","all"):run_oracle_benchmarks(config["oracle"],args.output)
    if args.stage in ("scaling","all"):run_scaling(config["scaling"],args.output)
    if args.stage in ("evolution","all"):
        from floworaclebench.flow_benchmark import run_evolution_benchmarks
        run_evolution_benchmarks(config["evolution"],args.output)
    dump_json(args.output/f"timing_{args.stage}.json",{"stage":args.stage,"elapsed_seconds":time.perf_counter()-start})


if __name__=="__main__":main()
