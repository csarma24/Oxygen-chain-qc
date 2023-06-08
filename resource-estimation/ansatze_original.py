"""
This file contains all the ansatze expressed in terms of native
PennyLane excitation gates, with no optimization or gate decomposition.

The simplify_interaction keyword argument can be used to construct the
regular ansatz or the reduced ansatz.
"""

import pennylane as qml

from custom_decompositions import *

def ansatz_18O(params, simplify_interaction=False):
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)

    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[2, 3, 4, 5])
    qml.DoubleExcitation(params[2], wires=[4, 5, 6, 7])
    qml.DoubleExcitation(params[3], wires=[6, 7, 8, 9])
    qml.DoubleExcitation(params[4], wires=[8, 9, 10, 11])


def ansatz_20O(params, simplify_interaction=False):
    for wire in range(4):
        qml.PauliX(wires=wire)

    # DoubleExcitation: |0, 1, 2, 3> --> |0, 1, 4, 5>; |0, 1, 6, 7>, |0, 1, 8, 9>, |0, 1, 10, 11>
    qml.DoubleExcitation(params[0], wires=[2, 3, 4, 5])
    qml.DoubleExcitation(params[1], wires=[4, 5, 6, 7])
    qml.DoubleExcitation(params[2], wires=[6, 7, 8, 9])
    qml.DoubleExcitation(params[3], wires=[8, 9, 10, 11])

    # Controlled-double: |2, 3, 4, 5>; |2, 3, 6, 7>; |2, 3, 8, 9>; |2, 3, 10, 11>
    #                   |4, 5, 6, 7>; |4, 5, 8, 9>; |4, 5, 10, 11>
    qml.ctrl(qml.DoubleExcitation, control=4)(params[4], wires=[0, 1, 2, 3])
    qml.ctrl(qml.DoubleExcitation, control=6)(params[5], wires=[0, 1, 2, 3])
    qml.ctrl(qml.DoubleExcitation, control=8)(params[6], wires=[0, 1, 2, 3])
    qml.ctrl(qml.DoubleExcitation, control=10)(params[7], wires=[0, 1, 2, 3])

    qml.ctrl(qml.DoubleExcitation, control=6)(params[8], wires=[2, 3, 4, 5])
    qml.ctrl(qml.DoubleExcitation, control=8)(params[9], wires=[2, 3, 4, 5])
    qml.ctrl(qml.DoubleExcitation, control=10)(params[10], wires=[2, 3, 4, 5])

    qml.ctrl(qml.DoubleExcitation, control=8)(params[11], wires=[4, 5, 6, 7])
    qml.ctrl(qml.DoubleExcitation, control=10)(params[12], wires=[4, 5, 6, 7])

    qml.ctrl(qml.DoubleExcitation, control=8)(params[13], wires=[6, 7, 10, 11])


def ansatz_22O(params, simplify_interaction=False):
    for wire in range(6):
        qml.PauliX(wires=wire)

    qml.DoubleExcitation(params[0], wires=[4, 5, 6, 7])
    qml.DoubleExcitation(params[1], wires=[6, 7, 8, 9])
    qml.DoubleExcitation(params[2], wires=[8, 9, 10, 11])

    qml.ctrl(qml.DoubleExcitation, control=10)(params[3], [2, 3, 4, 5])
    qml.ctrl(qml.DoubleExcitation, control=10)(params[4], [4, 5, 6, 7])
    qml.ctrl(qml.DoubleExcitation, control=10)(params[5], [6, 7, 8, 9])
    qml.ctrl(qml.DoubleExcitation, control=8)(params[6], wires=[11, 10, 7, 6])
    qml.ctrl(qml.DoubleExcitation, control=8)(params[7], wires=[7, 6, 5, 4])

    qml.ctrl(qml.DoubleExcitation, control=[0, 4])(params[8], wires=[9, 8, 7, 6])
    qml.ctrl(qml.DoubleExcitation, control=[4, 6])(params[9], wires=[0, 1, 2, 3])
    qml.ctrl(qml.DoubleExcitation, control=[2, 4])(params[10], wires=[6, 7, 8, 9])
    qml.ctrl(qml.DoubleExcitation, control=[2, 4])(params[11], wires=[8, 9, 10, 11])
    qml.ctrl(qml.DoubleExcitation, control=[2, 10])(params[12], wires=[4, 5, 6, 7])

    qml.ctrl(qml.DoubleExcitation, control=[6, 10])(params[13], wires=[2, 3, 4, 5])
    qml.ctrl(qml.DoubleExcitation, control=[6, 10])(params[14], wires=[4, 5, 8, 9])
    qml.ctrl(qml.DoubleExcitation, control=[6, 8])(params[15], wires=[11, 10, 5, 4])
    qml.ctrl(qml.DoubleExcitation, control=[4, 8])(params[16], wires=[6, 7, 10, 11])
    qml.ctrl(qml.DoubleExcitation, control=[8, 10])(params[17], wires=[5, 4, 3, 2])
    qml.ctrl(qml.DoubleExcitation, control=[2, 8])(params[18], wires=[11, 10, 7, 6])


def ansatz_24O(params, simplify_interaction=False):
    for wire in range(8, 12):
        qml.PauliX(wires=wire)

    qml.DoubleExcitation(params[0], wires=[9, 8, 7, 6])
    qml.DoubleExcitation(params[1], wires=[7, 6, 5, 4])
    qml.DoubleExcitation(params[2], wires=[5, 4, 3, 2])
    qml.DoubleExcitation(params[3], wires=[3, 2, 1, 0])

    qml.ctrl(qml.DoubleExcitation, control=7)(params[4], [11, 10, 9, 8])
    qml.ctrl(qml.DoubleExcitation, control=5)(params[5], [11, 10, 9, 8])
    qml.ctrl(qml.DoubleExcitation, control=3)(params[6], [11, 10, 9, 8])
    qml.ctrl(qml.DoubleExcitation, control=1)(params[7], [11, 10, 9, 8])

    if not simplify_interaction:
        qml.ctrl(qml.DoubleExcitation, control=5)(params[8], [9, 8, 7, 6])
        qml.ctrl(qml.DoubleExcitation, control=3)(params[9], [9, 8, 7, 6])
        qml.ctrl(qml.DoubleExcitation, control=1)(params[10], [9, 8, 7, 6])
        qml.ctrl(qml.DoubleExcitation, control=3)(params[11], [7, 6, 5, 4])
        qml.ctrl(qml.DoubleExcitation, control=1)(params[12], [7, 6, 5, 4])
        qml.ctrl(qml.DoubleExcitation, control=3)(params[13], [5, 4, 1, 0])

    for wire in range(12):
        qml.PauliX(wires=wire)


def ansatz_26O(params, simplify_interaction=False):
    # Construct the "inverse" ansatz by exchanging 0s and 1s. This is even
    # better because the first 10 qubits stay in their ground states as long
    # as possible and as such are less susceptible to decoherence.
    qml.PauliX(wires=10)
    qml.PauliX(wires=11)

    # This construction requires no controls; it is the same as that of 18O
    # but "upside down"
    qml.DoubleExcitation(params[0], wires=[11, 10, 9, 8])
    qml.DoubleExcitation(params[1], wires=[9, 8, 7, 6])
    qml.DoubleExcitation(params[2], wires=[7, 6, 5, 4])
    qml.DoubleExcitation(params[3], wires=[5, 4, 3, 2])
    qml.DoubleExcitation(params[4], wires=[3, 2, 1, 0])

    for wire in range(12):
        qml.PauliX(wires=wire)
