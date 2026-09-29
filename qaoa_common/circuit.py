"""
QAOA circuit factory and ZNE circuit folding.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_qaoa_circuit(Q: np.ndarray, p: int = 1) -> QuantumCircuit:
    """
    Build a parameterised QAOA circuit for a QUBO problem.

    Gate angle conventions (Qiskit):
      RZ(θ)  = exp(-iθ/2 · Z)    → θ = γ * Q[i,i]
      RZZ(θ) = exp(-iθ/2 · Z⊗Z) → θ = γ * Q[i,j] / 2  (factor ½ from QUBO expansion)
      RX(θ)  = exp(-iθ/2 · X)   → θ = 2β

    Qiskit is little-endian: rightmost bit in bitstring = qubit 0.

    Args:
        Q: (n x n) symmetric QUBO matrix.
        p: Number of QAOA layers.

    Returns:
        Parameterised QuantumCircuit with parameters named γ[0..p-1] and β[0..p-1].
    """
    n = Q.shape[0]
    gamma = ParameterVector('γ', p)
    beta  = ParameterVector('β', p)

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for layer in range(p):
        # Cost layer
        for i in range(n):
            if abs(Q[i, i]) > 1e-10:
                qc.rz(gamma[layer] * Q[i, i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-10:
                    qc.rzz(gamma[layer] * Q[i, j] / 2, i, j)

        qc.barrier()

        # Mixer layer
        for i in range(n):
            qc.rx(2 * beta[layer], i)

        qc.barrier()

    qc.measure_all()
    return qc


def get_params(qc: QuantumCircuit) -> dict:
    """
    Return a name→Parameter dict to avoid alphabetical-sort index ambiguity.
    (β[0] < γ[0] in Unicode, so qc.parameters[0] is β, not γ.)
    """
    return {p.name: p for p in qc.parameters}


def bind_params(qc: QuantumCircuit, gamma_vals: list[float], beta_vals: list[float]) -> QuantumCircuit:
    """
    Bind γ and β parameter vectors and return a concrete circuit.

    Args:
        qc:          Parameterised QAOA circuit from build_qaoa_circuit.
        gamma_vals:  List of γ values, one per layer.
        beta_vals:   List of β values, one per layer.

    Returns:
        Bound (concrete) QuantumCircuit.
    """
    p_by_name = get_params(qc)
    assignments = {}
    for layer, (g, b) in enumerate(zip(gamma_vals, beta_vals)):
        assignments[p_by_name[f'γ[{layer}]']] = g
        assignments[p_by_name[f'β[{layer}]']] = b
    return qc.assign_parameters(assignments)


def fold_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Apply gate folding for Zero Noise Extrapolation (ZNE).

    Each gate G is replaced by G (G† G)^m where scale_factor = 2m + 1.
    Only odd scale factors are supported (1, 3, 5, ...).

    Args:
        circuit:      Bound (no free parameters) QuantumCircuit.
        scale_factor: Odd integer ≥ 1.

    Returns:
        Folded QuantumCircuit with increased effective noise.
    """
    if scale_factor % 2 == 0 or scale_factor < 1:
        raise ValueError(f"scale_factor must be a positive odd integer, got {scale_factor}")
    if scale_factor == 1:
        return circuit.copy()

    m = (scale_factor - 1) // 2
    # Remove measurements for folding, re-add at the end
    circ_no_meas = circuit.remove_final_measurements(inplace=False)
    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for instruction in circ_no_meas.data:
        folded.append(instruction)
        if instruction.operation.name not in ('barrier',):
            for _ in range(m):
                folded.append(instruction.operation.inverse(), instruction.qubits)
                folded.append(instruction)

    folded.measure_all()
    return folded
