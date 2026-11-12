"""Complete learned finite-grid flow with executed, clean compiled oracles.

Each phase block is simulated gate-by-gate on *every* clean address input.
Its resulting diagonal restriction is reused in the eight-factor formula.
This is exact blockwise simulation of the compiled circuit: clean ancillas
make the restriction valid on any superposition. Dense TT evaluation never
substitutes for execution of a compiled phase block.
"""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from qiskit.quantum_info import Operator,Statevector
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from .baselines import build_walsh_oracle,phase_error,select_phase_bits
from .benchmark import ROOT,dump_json,write_rows,select_precision
from .evolution import (kinetic_matrix,gaussian_state,dense_reference,build_kinetic_circuit,
                        build_gaussian_preparation,apply_product_formula,state_metrics)
from .potentials import load_potential
from .reversible import build_tt_oracle
from .resources import estimate_resources
from .selectswap import build_selectswap_oracle
from .simulation import simulate_paths,validate_oracle


class SerialResourceCounter:
    """Actual block counts under an explicit serial block schedule.

    Workspace is reused across calls. Depth is the sum of measured block
    depths: a valid schedule with block barriers, not globally optimized.
    T-depth remains the stated synthesis/model macro schedule.
    """
    def __init__(self,n_address):
        self.n_address=n_address;self.qubits=n_address;self.ops=Counter()
        self.total={k:0 for k in ("gate_count","native_depth","cnot_decomposed","cx","ccx",
                    "single_qubit_gates","arbitrary_rotations","t_count_estimate","t_depth_macro_estimate")}

    def add(self,circuit,budget,multiplicity=1):
        result=estimate_resources(circuit,self.n_address,budget)
        self.qubits=max(self.qubits,result["logical_qubits"])
        self.ops.update({k:multiplicity*v for k,v in result["native_gate_counts"].items()})
        for key in self.total:self.total[key]+=multiplicity*result[key]

    def result(self):
        return {**self.total,"logical_qubits":self.qubits,"address_qubits":self.n_address,
                "ancilla_qubits":self.qubits-self.n_address,"native_gate_counts":dict(self.ops),
                "depth_schedule":"sum of measured block depths; explicit serial block boundaries",
                "t_depth_interpretation":"estimated serial macro schedule, not optimized Clifford+T depth"}


def balanced_coefficients(potential,n,steps,final_time):
    dt=final_time/steps
    count=int(potential.config["N_t"])
    knots=np.unique(np.r_[np.linspace(0,1,count)[np.linspace(0,1,count)<=final_time],final_time])
    a=max(float(np.ptp(potential.values(float(t),n)))/2 for t in knots)
    k=potential.dimension*np.pi**2*n**2/(4*potential.length**2)
    if a==0:return math.sqrt(dt/2),math.sqrt(dt/2),a,k
    return math.sqrt(dt*a/(2*k)),math.sqrt(dt*k/(2*a)),a,k


def _execute_phase(oracle,expected_phases):
    result=simulate_paths(oracle.circuit,oracle.n_address)
    addresses=np.arange(2**oracle.n_address,dtype=np.int64)
    expected_bits=((addresses[None,:]>>np.arange(oracle.n_address)[:,None])&1).astype(bool)
    if result.bits[oracle.n_address:].any() or not np.array_equal(result.bits[:oracle.n_address],expected_bits):
        raise AssertionError("End-to-end oracle did not clean its workspace/address")
    circuit_error=float(np.max(np.abs(result.amplitudes-np.exp(1j*expected_phases))))
    if circuit_error>5e-10:raise AssertionError(f"Gate execution disagrees with intended phase: {circuit_error}")
    return result.amplitudes,circuit_error


def run_evolution_benchmarks(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    potential=load_potential(ROOT/config["checkpoint"])
    methods=("walsh","selectswap","reversible_tt")
    rows=[];step_rows=[];references=[];states={}
    final_time=float(config["final_time"])
    # Small dense algebra should not pay dozens of BLAS-thread launches.
    with threadpool_limits(limits=1):
        for n in config["grids"]:
            reference,meta=dense_reference(potential,n,final_time)
            refined,refined_meta=dense_reference(potential,n,final_time,rtol=1e-11,atol=1e-13)
            meta["refinement_state_difference"]=float(np.linalg.norm(reference-refined))
            meta["refined_solver"]=refined_meta;references.append(meta)
            states[f"n{n}_reference"]=reference
            kinetic=kinetic_matrix(n,potential.dimension,potential.length)
            n_address=potential.dimension*(n.bit_length()-1)
            preparation=build_gaussian_preparation(n,potential.dimension,potential.length,potential.sigma0)
            initial=Statevector.from_instruction(preparation).data
            initial_error=float(np.linalg.norm(initial-gaussian_state(n,potential.dimension,potential.length,potential.sigma0)))
            if initial_error>1e-12:raise AssertionError("Gaussian preparation mismatch")
            for steps in config["steps"]:
                start=perf_counter()
                alpha,beta,a,k=balanced_coefficients(potential,n,steps,final_time)
                dt=final_time/steps
                eta=float(config["circuit_approximation_budget"])/(4*steps)
                rotation_budget=float(config["synthesis_budget"])/(8*steps+1)
                k_circuit=build_kinetic_circuit(n,potential.dimension,potential.length,alpha)
                k_executed=Operator(k_circuit).data
                k_exact=expm(1j*alpha*kinetic)
                k_error=float(np.linalg.norm(k_executed-k_exact,ord=2))
                if k_error>1e-12:raise AssertionError("Compiled kinetic step mismatch")
                counters={m:SerialResourceCounter(n_address) for m in methods}
                for counter in counters.values():
                    counter.add(preparation,rotation_budget)
                    counter.add(k_circuit,rotation_budget,4*steps)
                actual={m:initial.copy() for m in methods}
                exact_formula=initial.copy();midpoint_expm=initial.copy()
                errors={m:0. for m in methods};phase_circuit_error={m:0. for m in methods}
                compile_time={m:0. for m in methods};execution_time={m:0. for m in methods}
                search_time=0.;max_bits=0;min_bits=100;block_hist=Counter()
                saved_phases={m:[] for m in methods}
                print(f"full flow n={n}, steps={steps}, T={final_time}, eta={eta:.3g}",flush=True)
                for step in range(steps):
                    t=(step+.5)*dt
                    cores=potential.cores(t,n);values=potential.values(t,n)
                    phase_exact=np.exp(1j*beta*values)
                    exact_formula=apply_product_formula(exact_formula,phase_exact,k_exact)
                    commutator=kinetic*values[None,:]-values[:,None]*kinetic
                    midpoint_expm=expm(dt*commutator)@midpoint_expm
                    search_start=perf_counter();bits,_=select_precision(cores,beta,eta)
                    phase_bits=select_phase_bits(values,beta,eta)
                    search_time+=perf_counter()-search_start
                    max_bits=max(max_bits,bits);min_bits=min(min_bits,bits)
                    walsh=build_walsh_oracle(values,beta,eta)
                    tt=build_tt_oracle(cores,beta,bits)
                    # Construct and count each small-grid lookup tradeoff;
                    # selection depends only on oracle target/error, never state.
                    lookup_candidates=[]
                    lookup_compile=0.
                    for block in range(n_address+1):
                        candidate=build_selectswap_oracle(values,beta,phase_bits,block)
                        lookup_compile+=candidate.metadata["compile_seconds"]
                        cost=estimate_resources(candidate.circuit,n_address,rotation_budget)["t_count_estimate"]
                        lookup_candidates.append((cost,candidate))
                    lookup=min(lookup_candidates,key=lambda item:item[0])[1]
                    block_hist[str(lookup.metadata["block_bits"])]+=1
                    oracles={"walsh":walsh,"selectswap":lookup,"reversible_tt":tt}
                    for method,oracle in oracles.items():
                        expected=beta*tt.quantized_values if method=="reversible_tt" else oracle.phases
                        run_start=perf_counter();executed,gate_error=_execute_phase(oracle,expected)
                        execution_time[method]+=perf_counter()-run_start
                        # Algebraic inverse is checked by gate simulation at
                        # three representative times; all calls are clean.
                        if step in {0,steps//2,steps-1}:
                            check=validate_oracle(oracle.circuit,n_address,expected)
                            if check["inverse_error"]>5e-10 or not check["inverse_ancillas_clean"]:
                                raise AssertionError("Flow phase inverse failed")
                        phase_error_here=float(np.max(np.abs(executed-phase_exact)))
                        if phase_error_here>eta+5e-10:
                            raise AssertionError("Flow oracle exceeded allocated operator error")
                        errors[method]+=4*phase_error_here
                        phase_circuit_error[method]=max(phase_circuit_error[method],gate_error)
                        actual[method]=apply_product_formula(actual[method],executed,k_executed)
                        counters[method].add(oracle.circuit,rotation_budget,4)
                        compile_time[method]+=(lookup_compile if method=="selectswap" else oracle.metadata["compile_seconds"])
                        saved_phases[method].append(executed)
                        step_rows.append({"grid":n,"steps":steps,"step":step,"time":t,"method":method,
                            "alpha":alpha,"beta":beta,"operator_error":phase_error_here,
                            "gate_phase_error":gate_error,"ancillas_clean":True,"validated_inputs":n**potential.dimension,
                            "core_fraction_bits":bits if method=="reversible_tt" else None,
                            "phase_bits":phase_bits if method=="selectswap" else None,
                            "block_bits":lookup.metadata["block_bits"] if method=="selectswap" else None,
                            "compile_seconds":oracle.metadata["compile_seconds"],
                            **estimate_resources(oracle.circuit,n_address,rotation_budget)})
                    if (step+1)%64==0:print(f"  executed {step+1}/{steps} time slices",flush=True)
                total_elapsed=perf_counter()-start
                common={"checkpoint":config["checkpoint"],"grid":n,"dimension":potential.dimension,"rank":potential.config["D"],
                        "steps":steps,"final_time":final_time,"alpha":alpha,"beta":beta,
                        "potential_half_range":a,"kinetic_half_range":k,
                        "coefficient_rule":"spectral half-range balance; alpha*beta=dt/2",
                        "initial_preparation_error":initial_error,"kinetic_operator_error":k_error,
                        "time_freezing_error":float(np.linalg.norm(midpoint_expm-reference)),
                        "product_formula_error":float(np.linalg.norm(exact_formula-midpoint_expm)),
                        "reference_refinement_error":meta["refinement_state_difference"],
                        "core_fraction_bits_min":min_bits,"core_fraction_bits_max":max_bits,
                        "precision_search_seconds":search_time,"total_case_seconds":total_elapsed,
                        "rotation_synthesis_executed":False,"rotation_synthesis_budget":config["synthesis_budget"]}
                rows.append({**common,"method":"exact_formula",**state_metrics(exact_formula,reference),
                             "logical_qubits":None,"t_count_estimate":None,"costed":False,
                             "circuit_vs_formula_error":0.,"phase_error_accumulation_bound":0.})
                for method in methods:
                    row={**common,"method":method,"costed":True,**state_metrics(actual[method],reference),**counters[method].result(),
                         "circuit_vs_formula_error":float(np.linalg.norm(actual[method]-exact_formula)),
                         "phase_error_accumulation_bound":errors[method],
                         "accumulation_bound_including_kinetic_and_preparation":errors[method]+4*steps*k_error+initial_error,
                         "max_executed_gate_phase_error":phase_circuit_error[method],
                         "compile_seconds":compile_time[method],"gate_execution_seconds":execution_time[method],
                         "potential_calls":4*steps,"kinetic_calls":4*steps,"ancillas_clean_all_steps":True,
                         "lookup_block_histogram":dict(block_hist) if method=="selectswap" else {}}
                    if row["circuit_vs_formula_error"]>row["accumulation_bound_including_kinetic_and_preparation"]+1e-9:
                        raise AssertionError("Unitary error accumulation bound failed")
                    rows.append(row)
                    states[f"n{n}_r{steps}_{method}"]=actual[method]
                    states[f"n{n}_r{steps}_{method}_executed_phases"]=np.stack(saved_phases[method])
                    print(f"  {method}: fidelity={row['fidelity']:.9f}, TV={row['tv_distance']:.5f}, T~{row['t_count_estimate']}",flush=True)
                states[f"n{n}_r{steps}_exact_formula"]=exact_formula
                states[f"n{n}_r{steps}_midpoint_expm"]=midpoint_expm
                write_rows(output/"evolution.csv",rows)
                write_rows(output/"evolution_steps.csv",step_rows)
                dump_json(output/"evolution_results.json",{"results":rows,"reference":references})
                np.savez_compressed(output/"evolution_states.npz",**states)
    return rows

