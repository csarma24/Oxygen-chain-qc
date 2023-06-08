"""
Custom gate decompositions into the CNOT / RZ / RY gate set.
"""

import pennylane as qml
from pennylane import numpy as np


def custom_hadamard(wire):
    # Use RZ/RY instead of RZ RX RZ in the native decomp
    return [qml.PhaseShift(np.pi, wires=wire), qml.RY(np.pi / 2, wires=wire)]


def custom_ch(wires):
    return [
        qml.RY(-np.pi / 4, wires=wires[1]),
        custom_hadamard(wires[0]),
        qml.CNOT(wires=[wires[1], wires[0]]),
        custom_hadamard(wires[0]),
        qml.RY(+np.pi / 4, wires=wires[1]),
    ]


def custom_toffoli(wires):
    return [
        qml.PhaseShift(-np.pi / 4, wires=wires[0]),
        qml.PhaseShift(-np.pi / 4, wires=wires[1]),
        custom_hadamard(wires[2]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.PhaseShift(np.pi / 4, wires=wires[0]),
        qml.CNOT(wires=[wires[1], wires[2]]),
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.PhaseShift(np.pi / 4, wires=wires[2]),
        qml.PhaseShift(-np.pi / 4, wires=wires[0]),
        qml.CNOT(wires=[wires[1], wires[2]]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.PhaseShift(-np.pi / 4, wires=wires[2]),
        qml.PhaseShift(np.pi / 4, wires=wires[0]),
        qml.CNOT(wires=[wires[1], wires[0]]),
        custom_hadamard(wires[2]),
    ]


def custom_ctrl_toffoli(control, wires):
    # Source: Fig. 4 of this work: https://arxiv.org/pdf/1904.06920.pdf
    return [
        qml.PhaseShift(np.pi/8, wires=control),
        qml.PhaseShift(-np.pi/8, wires=wires[0]),
        qml.PhaseShift(np.pi/4, wires=wires[1]),
        custom_hadamard(wire=wires[2]),
        qml.CNOT(wires=[wires[2], wires[1]]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.CNOT(wires=[wires[2], control]),
        qml.PhaseShift(-np.pi/8, wires=control),
        qml.PhaseShift(-np.pi/8, wires=wires[0]),
        qml.PhaseShift(-np.pi/4, wires=wires[1]),
        qml.PhaseShift(np.pi/8, wires=wires[2]),
        qml.CNOT(wires=[wires[2], wires[1]]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.CNOT(wires=[wires[2], control]),
        custom_hadamard(wire=wires[1]),
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.CNOT(wires=[wires[1], control]),
        qml.PhaseShift(np.pi/4, wires=control),
        qml.PhaseShift(np.pi/4, wires=wires[0]),
        qml.PhaseShift(-np.pi/4, wires=wires[1]),
        qml.CNOT(wires=[control, wires[0]]),
        qml.CNOT(wires=[wires[1], control]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        qml.PhaseShift(-np.pi/8, wires=wires[0]),
        qml.PhaseShift(-np.pi/4, wires=wires[1]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        custom_hadamard(wire=wires[1]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.CNOT(wires=[wires[1], wires[2]]),
        qml.PhaseShift(np.pi/8, wires=wires[0]),
        qml.PhaseShift(-np.pi/4, wires=wires[1]),
        qml.PhaseShift(np.pi/4, wires=wires[2]),
        qml.CNOT(wires=[wires[1], wires[2]]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        custom_hadamard(wire=wires[1]),
        custom_hadamard(wire=wires[2]),
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.CNOT(wires=[wires[1], control]),
        qml.PhaseShift(-np.pi/4, wires=control),
        qml.PhaseShift(np.pi/4, wires=wires[0]),
        qml.PhaseShift(np.pi/4, wires=wires[1]),
        qml.CNOT(wires=[control, wires[0]]),
        qml.CNOT(wires=[wires[1], control]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        qml.PhaseShift(np.pi/4, wires=wires[0]),
        qml.PhaseShift(-np.pi/4, wires=wires[1]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        custom_hadamard(wire=wires[1])
    ]


def custom_initial_double(phi, wires):
    # This operation applies a double excitation to a single basis state
    # It should be used only at the beginning of a circuit in the
    # initial cascade after the Hartree-Fock state is constructed.
    return [
        qml.SingleExcitation.compute_decomposition(phi, wires=[wires[1], wires[2]]),
        qml.CNOT(wires=[wires[2], wires[0]]),
        qml.CNOT(wires=[wires[2], wires[3]])
    ]


def ctrl_double(phi, control, target_wires):
    return [
        qml.ctrl(qml.SingleExcitation, control=control)(phi, wires=[target_wires[1], target_wires[2]]),
        qml.CNOT(wires=[target_wires[3], target_wires[0]]),
        qml.Toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.CNOT(wires=[target_wires[3], target_wires[0]])
    ]


def ctrl_double_decomp(phi, control, target_wires):
    return [
        qml.CNOT(wires=[target_wires[1], target_wires[2]]),
        qml.RY(phi/4, wires=target_wires[1]),
        qml.CNOT(wires=[target_wires[2], target_wires[1]]),
        qml.RY(-phi/4, wires=target_wires[1]),
        qml.CNOT(wires=[control, target_wires[1]]),
        qml.RY(phi/4, wires=target_wires[1]),
        qml.CNOT(wires=[target_wires[2], target_wires[1]]),
        qml.RY(-phi/4, wires=target_wires[1]),
        qml.CNOT(wires=[control, target_wires[1]]),
        qml.CNOT(wires=[target_wires[1], target_wires[2]]),
        qml.CNOT(wires=[target_wires[3], target_wires[0]]),
        custom_toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.CNOT(wires=[target_wires[3], target_wires[0]])
    ]


def ctrl_ctrl_double(phi, control, target_wires):
    return [
        qml.Toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.CRY(phi/4, wires=[control, target_wires[2]]),
        qml.Toffoli(wires=[control, target_wires[3], target_wires[2]]),
        qml.CRY(-phi/4, wires=[control, target_wires[2]]),
        qml.Toffoli(wires=[control, target_wires[0], target_wires[2]]),
        qml.CRY(phi/4, wires=[control, target_wires[2]]),
        qml.Toffoli(wires=[control, target_wires[3], target_wires[2]]),
        qml.CRY(-phi/4, wires=[control, target_wires[2]]),
        qml.Toffoli(wires=[control, target_wires[0], target_wires[2]]),
        qml.Toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.Toffoli(wires=[control, target_wires[4], target_wires[1]]),
        qml.ctrl(qml.Toffoli, control=control)(wires=[target_wires[3], target_wires[0], target_wires[4]]),
        qml.Toffoli(wires=[control, target_wires[4], target_wires[1]])
    ]


def ctrl_ctrl_double_decomp(phi, control, target_wires):
    return [
        custom_toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.RY(phi/8, wires=target_wires[2]),
        qml.CNOT(wires=[control, target_wires[2]]),
        qml.RY(-phi/8, wires=target_wires[2]),
        qml.PauliX(wires=target_wires[3]),
        custom_toffoli(wires=[control, target_wires[3], target_wires[2]]),
        qml.RY(-phi/8, wires=target_wires[2]),
        qml.CNOT(wires=[control, target_wires[2]]),
        qml.RY(phi/8, wires=target_wires[2]),
        qml.PauliX(wires=target_wires[0]),
        custom_toffoli(wires=[control, target_wires[0], target_wires[2]]),
        qml.RY(phi/8, wires=target_wires[2]),
        qml.CNOT(wires=[control, target_wires[2]]),
        qml.RY(-phi/8, wires=target_wires[2]),
        custom_toffoli(wires=[control, target_wires[3], target_wires[2]]),
        qml.PauliX(wires=target_wires[3]),
        qml.RY(-phi/8, wires=target_wires[2]),
        qml.CNOT(wires=[control, target_wires[2]]),
        qml.RY(phi/8, wires=target_wires[2]),
        custom_toffoli(wires=[control, target_wires[0], target_wires[2]]),
        qml.PauliX(wires=target_wires[0]),
        custom_toffoli(wires=[control, target_wires[2], target_wires[3]]),
        qml.CNOT(wires=[target_wires[4], target_wires[1]]),
        custom_ctrl_toffoli(control, wires=[target_wires[0], target_wires[3], target_wires[4]]),
        qml.CNOT(wires=[target_wires[4], target_wires[1]])
    ]
