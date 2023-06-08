"""
This file contains all the ansatze with some custom optimizations at the
beginning of each circuit.
"""

import pennylane as qml

from custom_decompositions import *


def ansatz_18O(params, simplify_interaction=False):
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)

    qml.RY(params[0], wires=3)
    
    for param_idx, init_wire in enumerate(range(3, 10, 2)): 
        qml.RY(-params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        qml.CNOT(wires=[init_wire, init_wire-1])
        qml.RY(params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        
    qml.CNOT(wires=[11, 10])
        
    for qubit_idx in range(2, 12):
        qml.CNOT(wires=[qubit_idx, qubit_idx-2])

    
def ansatz_20O(params, simplify_interaction=False):
    for wire in range(4):
        qml.PauliX(wires=wire)

    qml.RY(params[0], wires=5)
    
    for param_idx, init_wire in enumerate(range(5, 10, 2)): 
        qml.RY(-params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        qml.CNOT(wires=[init_wire, init_wire-1])
        qml.RY(params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        
    qml.CNOT(wires=[11, 10])
        
    for qubit_idx in range(4, 12):
        qml.CNOT(wires=[qubit_idx, qubit_idx-2])
        
    ctrl_double_decomp(params[4], 10, [0, 1, 2, 3])
    ctrl_double_decomp(params[5], 10, [2, 3, 4, 5])
    ctrl_double_decomp(params[6], 10, [4, 5, 6, 7])
    ctrl_double_decomp(params[7], 10, [6, 7, 8, 9])
    
    ctrl_double_decomp(params[8], 8, [11, 10, 7, 6])
    ctrl_double_decomp(params[9], 8, [7, 6, 5, 4])
    ctrl_double_decomp(params[10], 8, [5, 4, 3, 2])

    ctrl_double_decomp(params[11], 2, [9, 8, 7, 6])
    ctrl_double_decomp(params[12], 2, [7, 6, 5, 4])
    
    ctrl_double_decomp(params[13], 4, [2, 3, 6, 7])


def ansatz_22O(params, simplify_interaction=False):
    for wire in range(6):
        qml.PauliX(wires=wire)

    # First four excitations
    qml.RY(params[0], wires=7)
    
    for param_idx, init_wire in enumerate(range(7, 10, 2)): 
        qml.RY(-params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        qml.CNOT(wires=[init_wire, init_wire-1])
        qml.RY(params[param_idx+1]/2, wires=init_wire+2)
        qml.CNOT(wires=[init_wire, init_wire+2])
        
    qml.CNOT(wires=[11, 10])
    
    for qubit_idx in range(6, 12):
        qml.CNOT(wires=[qubit_idx, qubit_idx-2])

    ctrl_double_decomp(params[3], 10, [2, 3, 4, 5])
    ctrl_double_decomp(params[4], 10, [4, 5, 6, 7])
    ctrl_double_decomp(params[5], 10, [6, 7, 8, 9])
    ctrl_double_decomp(params[6], 8, [11, 10, 7, 6])
    ctrl_double_decomp(params[7], 8, [7, 6, 5, 4])
    
    ctrl_ctrl_double_decomp(params[8], 0, [4, 9, 8, 7, 6])
    ctrl_ctrl_double_decomp(params[9], 4, [6, 0, 1, 2, 3])
    ctrl_ctrl_double_decomp(params[10], 2, [4, 6, 7, 8, 9])
    ctrl_ctrl_double_decomp(params[11], 2, [4, 8, 9, 10, 11])
    ctrl_ctrl_double_decomp(params[12], 2, [10, 4, 5, 6, 7])
    ctrl_ctrl_double_decomp(params[13], 6, [10, 2, 3, 4, 5])
    ctrl_ctrl_double_decomp(params[14], 6, [10, 4, 5, 8, 9])
    ctrl_ctrl_double_decomp(params[15], 6, [8, 11, 10, 5, 4])
    ctrl_ctrl_double_decomp(params[16], 4, [8, 6, 7, 10, 11])
    ctrl_ctrl_double_decomp(params[17], 8, [10, 5, 4, 3, 2])
    ctrl_ctrl_double_decomp(params[18], 2, [8, 11, 10, 7, 6])
    

def ansatz_24O(params, simplify_interaction=False):
    for wire in range(8, 12):
        qml.PauliX(wires=wire)

    qml.RY(params[0], wires=6)

    for param_idx, init_wire in enumerate(range(6, 0, -2)): 
        qml.RY(-params[param_idx+1]/2, wires=init_wire-2)
        qml.CNOT(wires=[init_wire, init_wire-2])
        qml.CNOT(wires=[init_wire, init_wire+1])
        qml.RY(params[param_idx+1]/2, wires=init_wire-2)
        qml.CNOT(wires=[init_wire, init_wire-2])

    qml.CNOT(wires=[0, 1])

    for qubit_idx in range(7, -1, -1):
        qml.CNOT(wires=[qubit_idx, qubit_idx+2])

    ctrl_double_decomp(params[4], 1, [11, 10, 9, 8])
    ctrl_double_decomp(params[5], 1, [9, 8, 7, 6])
    ctrl_double_decomp(params[6], 1, [7, 6, 5, 4])
    ctrl_double_decomp(params[7], 1, [5, 4, 3, 2])

    ctrl_double_decomp(params[8], 3, [0, 1, 4, 5])
    ctrl_double_decomp(params[9], 3, [4, 5, 6, 7])
    ctrl_double_decomp(params[10], 3, [6, 7, 8, 9])

    ctrl_double_decomp(params[11], 9, [2, 3, 4, 5])
    ctrl_double_decomp(params[12], 9, [4, 5, 6, 7])

    ctrl_double_decomp(params[13], 7, [9, 8, 5, 4])
    
    for wire in range(12):
        qml.PauliX(wires=wire)


def ansatz_26O(params, simplify_interaction=False):
    qml.PauliX(wires=10)
    qml.PauliX(wires=11)

    qml.RY(params[0], wires=8)

    for param_idx, init_wire in enumerate(range(8, 1, -2)): 
        qml.RY(-params[param_idx+1]/2, wires=init_wire-2)
        qml.CNOT(wires=[init_wire, init_wire-2])
        qml.RY(params[param_idx+1]/2, wires=init_wire-2)
        qml.CNOT(wires=[init_wire, init_wire+1])
        qml.CNOT(wires=[init_wire, init_wire-2])

    qml.CNOT(wires=[0, 1])

    for qubit_idx in range(9, -1, -1):
        qml.CNOT(wires=[qubit_idx, qubit_idx+2])

    # Exchanges the 0s and 1s
    for wire in range(12):
        qml.PauliX(wires=wire)
