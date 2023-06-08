import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np

"""
Results direct from Aria
"""
job_results = [
    {
        "histogram": {
            "5": 0.001,
            "6": 0.003,
            "7": 0.003,
            "10": 0.009,
            "11": 0.006,
            "12": 0.004,
            "13": 0.006,
            "14": 0.257,
            "15": 0.298,
            "22": 0.004,
            "23": 0.005,
            "26": 0.002,
            "27": 0.007,
            "28": 0.009,
            "29": 0.011,
            "30": 0.153,
            "31": 0.222,
        }
    },
    {
        "histogram": {
            "3": 0.001,
            "5": 0.011,
            "6": 0.001,
            "7": 0.004,
            "8": 0.001,
            "9": 0.013,
            "11": 0.006,
            "12": 0.011,
            "13": 0.286,
            "14": 0.011,
            "15": 0.25,
            "21": 0.003,
            "23": 0.002,
            "25": 0.008,
            "27": 0.006,
            "28": 0.01,
            "29": 0.239,
            "30": 0.008,
            "31": 0.129,
        }
    },
    {
        "histogram": {
            "1": 0.001,
            "3": 0.001,
            "7": 0.002,
            "9": 0.006,
            "10": 0.027,
            "11": 0.294,
            "13": 0.006,
            "14": 0.011,
            "15": 0.229,
            "18": 0.001,
            "19": 0.002,
            "23": 0.004,
            "24": 0.001,
            "25": 0.005,
            "26": 0.006,
            "27": 0.148,
            "29": 0.008,
            "30": 0.007,
            "31": 0.241,
        }
    },
    {
        "histogram": {
            "3": 0.001,
            "7": 0.004,
            "11": 0.011,
            "13": 0.008,
            "14": 0.024,
            "15": 0.527,
            "21": 0.001,
            "22": 0.001,
            "23": 0.011,
            "27": 0.012,
            "29": 0.022,
            "30": 0.017,
            "31": 0.361,
        }
    },
    {
        "histogram": {
            "3": 0.011,
            "4": 0.002,
            "5": 0.009,
            "6": 0.011,
            "7": 0.27,
            "11": 0.003,
            "13": 0.005,
            "14": 0.013,
            "15": 0.288,
            "18": 0.001,
            "19": 0.011,
            "21": 0.011,
            "22": 0.006,
            "23": 0.135,
            "27": 0.005,
            "29": 0.014,
            "30": 0.003,
            "31": 0.202,
        }
    },
    {
        "histogram": {
            "3": 0.002,
            "7": 0.006,
            "9": 0.002,
            "11": 0.015,
            "12": 0.002,
            "13": 0.013,
            "14": 0.017,
            "15": 0.03,
            "23": 0.01,
            "27": 0.013,
            "29": 0.027,
            "30": 0.02,
            "31": 0.843,
        }
    },
    {
        "histogram": {
            "0": 0.039,
            "1": 0.052,
            "2": 0.041,
            "3": 0.056,
            "4": 0.02,
            "5": 0.028,
            "6": 0.022,
            "7": 0.026,
            "8": 0.036,
            "9": 0.037,
            "10": 0.038,
            "11": 0.034,
            "12": 0.015,
            "13": 0.015,
            "14": 0.025,
            "15": 0.026,
            "16": 0.03,
            "17": 0.036,
            "18": 0.015,
            "19": 0.015,
            "20": 0.044,
            "21": 0.048,
            "22": 0.028,
            "23": 0.024,
            "24": 0.026,
            "25": 0.033,
            "26": 0.015,
            "27": 0.028,
            "28": 0.047,
            "29": 0.053,
            "30": 0.022,
            "31": 0.026,
        }
    },
    {
        "histogram": {
            "0": 0.005,
            "1": 0.003,
            "2": 0.007,
            "3": 0.004,
            "4": 0.001,
            "5": 0.005,
            "6": 0.004,
            "7": 0.004,
            "8": 0.004,
            "9": 0.005,
            "10": 0.009,
            "11": 0.005,
            "12": 0.006,
            "13": 0.004,
            "14": 0.007,
            "15": 0.001,
            "16": 0.053,
            "17": 0.071,
            "18": 0.032,
            "19": 0.045,
            "20": 0.072,
            "21": 0.079,
            "22": 0.044,
            "23": 0.056,
            "24": 0.047,
            "25": 0.074,
            "26": 0.035,
            "27": 0.04,
            "28": 0.078,
            "29": 0.087,
            "30": 0.058,
            "31": 0.055,
        }
    },
]

"""
Hamiltonian for 26O
"""
coeffs = np.array(
    [
        -10.943495,
        6.590075,
        6.59009,
        6.59008,
        6.1535,
        -3.991875,
        3.991875,
        -0.56607,
        -0.400535,
        -0.334915,
        -0.841875,
        0.34829,
        -0.665395,
        -0.33492,
        -0.545725,
        0.644445,
        -0.334915,
        -0.39765,
        0.792515,
        -0.282125,
        0.282125,
        0.225,
        0.34782,
        0.34782,
        -0.182285,
        -0.182285,
        -0.419445,
        -0.419445,
        -0.667545,
        -0.667545,
        0.24693,
        0.24693,
        0.44714,
        0.44714,
        0.419445,
        0.419445,
        0.415175,
        0.415175,
        -0.4993,
        -0.4993,
        -0.419445,
        -0.419445,
        -0.28899,
        -0.28899,
        0.625485,
        0.625485,
        -0.22309,
        -0.22309,
        0.22309,
        0.22309,
        0.27825,
        0.27825,
    ]
)

ops = [
    Identity(wires=[0]),
    PauliZ(wires=[0]),
    PauliZ(wires=[1]),
    PauliZ(wires=[2]),
    PauliZ(wires=[3]),
    PauliZ(wires=[4]),
    PauliZ(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliZ(wires=[3])
    @ PauliZ(wires=[4]),
    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
    PauliZ(wires=[0]) @ PauliZ(wires=[4]),
    PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]),
    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
    PauliZ(wires=[1]) @ PauliZ(wires=[4]),
    PauliZ(wires=[0]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]),
    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
    PauliZ(wires=[2]) @ PauliZ(wires=[4]),
    PauliZ(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[3]) @ PauliZ(wires=[4]),
    PauliZ(wires=[3]) @ PauliZ(wires=[4]),
    PauliZ(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[4]),
    PauliZ(wires=[0]) @ PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]),
    PauliY(wires=[0]) @ PauliY(wires=[1]),
    PauliX(wires=[0]) @ PauliX(wires=[1]),
    PauliY(wires=[0]) @ PauliY(wires=[2]),
    PauliX(wires=[0]) @ PauliX(wires=[2]),
    PauliY(wires=[0]) @ PauliY(wires=[3]),
    PauliX(wires=[0]) @ PauliX(wires=[3]),
    PauliY(wires=[0]) @ PauliY(wires=[4]),
    PauliX(wires=[0]) @ PauliX(wires=[4]),
    PauliX(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliZ(wires=[3])
    @ PauliZ(wires=[4]),
    PauliX(wires=[0]),
    PauliY(wires=[1]) @ PauliY(wires=[2]),
    PauliX(wires=[1]) @ PauliX(wires=[2]),
    PauliY(wires=[1]) @ PauliY(wires=[3]),
    PauliX(wires=[1]) @ PauliX(wires=[3]),
    PauliY(wires=[1]) @ PauliY(wires=[4]),
    PauliX(wires=[1]) @ PauliX(wires=[4]),
    PauliZ(wires=[0])
    @ PauliX(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliZ(wires=[3])
    @ PauliZ(wires=[4]),
    PauliX(wires=[1]),
    PauliY(wires=[2]) @ PauliY(wires=[3]),
    PauliX(wires=[2]) @ PauliX(wires=[3]),
    PauliY(wires=[2]) @ PauliY(wires=[4]),
    PauliX(wires=[2]) @ PauliX(wires=[4]),
    PauliZ(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliX(wires=[2])
    @ PauliZ(wires=[3])
    @ PauliZ(wires=[4]),
    PauliX(wires=[2]),
    PauliY(wires=[3]) @ PauliY(wires=[4]),
    PauliX(wires=[3]) @ PauliX(wires=[4]),
    PauliZ(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliX(wires=[3])
    @ PauliZ(wires=[4]),
    PauliX(wires=[3]),
    PauliZ(wires=[0])
    @ PauliZ(wires=[1])
    @ PauliZ(wires=[2])
    @ PauliZ(wires=[3])
    @ PauliX(wires=[4]),
    PauliX(wires=[4]),
]

H_tapered = qml.Hamiltonian(coeffs, ops)
