"""Experiment automation. All reported circuit costs come from built circuits."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from .baselines import build_walsh_oracle, build_qrom_oracle, phase_error, select_phase_bits
from .potentials import load_potential, contract_cores
from .reversible import build_tt_oracle, quantize_cores, contract_cores as integer_contract
from .resources import estimate_resources
from .simulation import simulate_paths, validate_oracle

ROOT=Path(__file__).resolve().parents[2]


def dump_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def write_rows(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def quantized_values(cores, fraction_bits):
    ints=quantize_cores(cores,fraction_bits)
    values=np.asarray(integer_contract(ints),dtype=float).reshape(-1)
    return values*math.ldexp(1.,-len(cores)*fraction_bits)


def select_precision(cores,beta,budget,max_bits=24):
    """Exhaustive finite-grid operator check; never substitutes an L2 metric."""
    exact=contract_cores(cores).reshape(-1)
    for bits in range(max_bits+1):
        values=quantized_values(cores,bits)
        error=phase_error(beta*values,beta*exact)
        if error <= budget:
            return bits,error
    raise RuntimeError(f"No precision through {max_bits} bits meets error {budget}")


def _assert_valid(validation,tolerance=3e-10):
    for key in ("address_preserved","ancillas_clean","inverse_ancillas_clean"):
        if not validation[key]: raise AssertionError(f"Oracle failed: {validation}")
    for key in ("basis_phase_error","superposition_error","inverse_error"):
        if validation[key] > tolerance: raise AssertionError(f"Oracle failed: {validation}")


def audited_row(oracle,phases,context,synthesis_budget,validate=True,max_inputs=64):
    start=time.perf_counter()
    if validate:
        count=2**oracle.n_address
        addresses=None if count<=max_inputs else np.unique(np.linspace(0,count-1,max_inputs,dtype=np.int64))
        validation=validate_oracle(oracle.circuit,oracle.n_address,phases,addresses)
        _assert_valid(validation)
    else: validation={"exhaustive":False,"validated_inputs":0}
    row={**context,**oracle.metadata,**estimate_resources(oracle.circuit,oracle.n_address,synthesis_budget),
         **validation,"validation_seconds":time.perf_counter()-start}
    row["t_count_estimate_slope4"]=estimate_resources(oracle.circuit,oracle.n_address,synthesis_budget,slope=4.)["t_count_estimate"]
    return row


def baseline_portfolio(values,beta,tolerance,context,max_qubits=4096,validate=True):
    from .selectswap import build_selectswap_oracle
    budget=tolerance/2
    phase_bits=select_phase_bits(values,beta,budget)
    candidates=[]; rows=[]
    walsh=build_walsh_oracle(values,beta,approximation_budget=budget)
    candidates.append(walsh)
    rows.append(audited_row(walsh,walsh.phases,context,budget,validate))
    qrom=build_qrom_oracle(values,beta,phase_bits)
    candidates.append(qrom)
    rows.append(audited_row(qrom,qrom.phases,context,budget,validate))
    n=qrom.n_address
    for block_bits in range(1,n+1):
        qubits=n+(2**block_bits)*phase_bits+max(0,n-block_bits-1)
        if qubits>max_qubits: break
        oracle=build_selectswap_oracle(values,beta,phase_bits,block_bits)
        candidates.append(oracle)
        rows.append(audited_row(oracle,oracle.phases,context,budget,validate))
    for row in rows:
        if row["operator_error"] > budget+1e-11:
            raise AssertionError("Baseline failed matched accuracy")
    best=min(range(len(rows)),key=lambda i:rows[i]["t_count_estimate"])
    return candidates[best],rows[best],rows


def run_oracle_benchmarks(config,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]; matched=[]; portfolio=[]
    beta=float(config["beta"]); t=float(config["time"])
    for entry in config["learned"]:
        potential=load_potential(ROOT/entry["checkpoint"])
        for n in entry["grids"]:
            start=time.perf_counter(); cores=potential.cores(t,n)
            values=contract_cores(cores).reshape(-1)
            preprocessing=time.perf_counter()-start
            base={"dataset":entry["name"],"kind":"learned","dimension":potential.dimension,
                  "grid":n,"rank":max(c.shape[2] for c in cores),"time":t,"beta":beta,
                  "preprocess_seconds":preprocessing,"table_points":len(values),
                  "checkpoint":entry["checkpoint"]}
            print(f"oracles {base['dataset']} n={n}",flush=True)
            for bits in config["precision_sweep"]:
                oracle=build_tt_oracle(cores,beta,int(bits))
                phases=beta*oracle.quantized_values
                row=audited_row(oracle,phases,{**base,"experiment":"precision"},1e-4)
                row["operator_error"]=phase_error(phases,beta*values)
                row["potential_sup_error"]=float(np.max(np.abs(oracle.quantized_values-values)))
                row["operator_error_bound"]=abs(beta)*oracle.metadata["core_quantization_bound"]
                rows.append(row)
            for tolerance in config["tolerances"]:
                start=time.perf_counter()
                bits,error=select_precision(cores,beta,tolerance/2)
                precision_search=time.perf_counter()-start
                context={**base,"experiment":"matched","tolerance":tolerance,
                         "approximation_budget":tolerance/2,"precision_search_seconds":precision_search}
                tt=build_tt_oracle(cores,beta,bits)
                ttrow=audited_row(tt,beta*tt.quantized_values,context,tolerance/2)
                ttrow["operator_error"]=error
                ttrow["potential_sup_error"]=float(np.max(np.abs(tt.quantized_values-values)))
                ttrow["operator_error_bound"]=abs(beta)*tt.metadata["core_quantization_bound"]
                _,best,allrows=baseline_portfolio(values,beta,tolerance,context,config["baseline_max_qubits"])
                ttrow["best_baseline_method"]=best["method"]
                ttrow["best_baseline_block_bits"]=best.get("block_bits",0)
                ttrow["t_ratio_vs_best_baseline"]=ttrow["t_count_estimate"]/max(1,best["t_count_estimate"])
                matched.extend([ttrow,best]);portfolio.extend(allrows)
                print(f"  eps={tolerance:g}: TT f={bits}, T~{ttrow['t_count_estimate']}, best {best['method']} T~{best['t_count_estimate']}",flush=True)
    write_rows(output/"precision.csv",rows)
    write_rows(output/"matched.csv",matched)
    write_rows(output/"baseline_portfolio.csv",portfolio)
    dump_json(output/"oracle_results.json",{"precision":rows,"matched":matched,"baseline_portfolio":portfolio})
    return rows,matched,portfolio


def synthetic_cores(d,n,rank,seed=91):
    """Controlled, non-trained dense random TT cores, max grid potential 1.

    All bonds use requested rank; for this benchmark rank is a representation
    parameter and is not claimed to equal every unfolding's numerical rank.
    Balanced global rescaling avoids concentrating normalisation in one core.
    Dense normalisation is a charged classical benchmark-only operation.
    """
    rng=np.random.default_rng(seed+1000*d+100*n+rank)
    ranks=[1]+[rank]*(d-1)+[1]
    cores=[rng.uniform(-1,1,(ranks[j],n,ranks[j+1]))/np.sqrt(max(1,ranks[j])) for j in range(d)]
    values=contract_cores(cores)
    scale=float(np.max(np.abs(values)))**(-1/d)
    return [core*scale for core in cores]


def run_scaling(config,output):
    rows=[];portfolio=[]
    beta=config["beta"];tol=config["tolerance"]
    for case in config["cases"]:
        d,n,r=case
        start=time.perf_counter();cores=synthetic_cores(d,n,r,config["seed"])
        values=contract_cores(cores).reshape(-1);preprocess=time.perf_counter()-start
        context={"experiment":"scaling","dataset":f"synthetic_d{d}_n{n}_r{r}","kind":"synthetic",
                 "dimension":d,"grid":n,"rank":r,"beta":beta,"tolerance":tol,
                 "approximation_budget":tol/2,"table_points":len(values),"preprocess_seconds":preprocess,
                 "seed":config["seed"]}
        print(f"scaling d={d},n={n},r={r}",flush=True)
        bits,error=select_precision(cores,beta,tol/2)
        tt=build_tt_oracle(cores,beta,bits)
        row=audited_row(tt,beta*tt.quantized_values,context,tol/2,max_inputs=16)
        row["operator_error"]=error
        row["operator_error_bound"]=abs(beta)*tt.metadata["core_quantization_bound"]
        _,best,allrows=baseline_portfolio(values,beta,tol,context,config["baseline_max_qubits"],validate=len(values)<=1024)
        row["best_baseline_method"]=best["method"]
        row["best_baseline_block_bits"]=best.get("block_bits",0)
        row["t_ratio_vs_best_baseline"]=row["t_count_estimate"]/max(1,best["t_count_estimate"])
        # Report the qubit-resource tradeoff, not only the unlimited-qubit T winner.
        for cap in (128,256,512):
            eligible=[x for x in allrows if x["logical_qubits"]<=cap]
            constrained=min(eligible,key=lambda x:x["t_count_estimate"])
            row[f"best_baseline_t_at_{cap}_qubits"]=constrained["t_count_estimate"]
        rows.extend([row,best]);portfolio.extend(allrows)
        print(f"  TT f={bits}: {row['logical_qubits']} qubits, T~{row['t_count_estimate']}; {best['method']} T~{best['t_count_estimate']}",flush=True)
        # Checkpoint each expensive case, without presenting incomplete output.
        write_rows(Path(output)/"scaling.csv",rows)
        write_rows(Path(output)/"scaling_portfolio.csv",portfolio)
    dump_json(Path(output)/"scaling_results.json",{"scaling":rows,"baseline_portfolio":portfolio})
    return rows


def environment_record():
    import scipy,qiskit,torch,matplotlib,pytest
    return {"python":platform.python_version(),"platform":platform.platform(),
            "machine":platform.machine(),"numpy":np.__version__,"scipy":scipy.__version__,
            "qiskit":qiskit.__version__,"torch":torch.__version__,"matplotlib":matplotlib.__version__,
            "pytest":pytest.__version__,"execution":"CPU classical circuit and numerical simulation",
            "date":"2026-10-29","upstream_commit":"11358d3dca52193f93f037046229797296a2063e"}

