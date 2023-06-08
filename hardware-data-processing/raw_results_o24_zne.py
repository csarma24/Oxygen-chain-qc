import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np

"""
Results direct from Aria
"""
job_results = [
    {
        "histogram": {
            "0": 0.002,
            "1": 0.012,
            "2": 0.015,
            "3": 0.016,
            "4": 0.023,
            "5": 0.022,
            "6": 0.022,
            "7": 0.024,
            "8": 0.03,
            "9": 0.021,
            "10": 0.039,
            "11": 0.065,
            "12": 0.082,
            "13": 0.071,
            "14": 0.143,
            "15": 0.166,
            "16": 0.003,
            "17": 0.001,
            "18": 0.012,
            "19": 0.009,
            "20": 0.012,
            "21": 0.015,
            "22": 0.02,
            "23": 0.011,
            "24": 0.005,
            "25": 0.015,
            "26": 0.012,
            "27": 0.026,
            "28": 0.022,
            "29": 0.023,
            "30": 0.031,
            "31": 0.03,
        }
    },
    {
        "histogram": {
            "0": 0.015,
            "1": 0.013,
            "2": 0.008,
            "3": 0.009,
            "4": 0.012,
            "5": 0.039,
            "6": 0.018,
            "7": 0.046,
            "8": 0.066,
            "9": 0.034,
            "10": 0.039,
            "11": 0.025,
            "12": 0.056,
            "13": 0.131,
            "14": 0.05,
            "15": 0.21,
            "16": 0.009,
            "17": 0.005,
            "18": 0.003,
            "19": 0.005,
            "20": 0.003,
            "21": 0.012,
            "22": 0.004,
            "23": 0.032,
            "24": 0.012,
            "25": 0.018,
            "26": 0.014,
            "27": 0.017,
            "28": 0.009,
            "29": 0.028,
            "30": 0.016,
            "31": 0.042,
        }
    },
    {
        "histogram": {
            "0": 0.012,
            "1": 0.037,
            "2": 0.022,
            "3": 0.041,
            "4": 0.012,
            "5": 0.03,
            "6": 0.018,
            "7": 0.032,
            "8": 0.032,
            "9": 0.058,
            "10": 0.077,
            "11": 0.17,
            "12": 0.021,
            "13": 0.065,
            "14": 0.055,
            "15": 0.115,
            "16": 0.008,
            "17": 0.012,
            "18": 0.005,
            "19": 0.013,
            "20": 0.008,
            "21": 0.013,
            "22": 0.009,
            "23": 0.014,
            "24": 0.005,
            "25": 0.015,
            "26": 0.01,
            "27": 0.03,
            "28": 0.008,
            "29": 0.022,
            "30": 0.01,
            "31": 0.021,
        }
    },
    {
        "histogram": {
            "0": 0.006,
            "1": 0.018,
            "2": 0.017,
            "3": 0.012,
            "4": 0.017,
            "5": 0.031,
            "6": 0.015,
            "7": 0.043,
            "8": 0.03,
            "9": 0.024,
            "10": 0.085,
            "11": 0.042,
            "12": 0.048,
            "13": 0.081,
            "14": 0.057,
            "15": 0.237,
            "16": 0.008,
            "17": 0.011,
            "18": 0.008,
            "19": 0.014,
            "20": 0.01,
            "21": 0.012,
            "22": 0.007,
            "23": 0.018,
            "24": 0.011,
            "25": 0.011,
            "26": 0.011,
            "27": 0.02,
            "28": 0.016,
            "29": 0.042,
            "30": 0.013,
            "31": 0.025,
        }
    },
    {
        "histogram": {
            "0": 0.01,
            "1": 0.024,
            "2": 0.041,
            "3": 0.021,
            "4": 0.013,
            "5": 0.063,
            "6": 0.045,
            "7": 0.205,
            "8": 0.014,
            "9": 0.023,
            "10": 0.039,
            "11": 0.016,
            "12": 0.029,
            "13": 0.069,
            "14": 0.034,
            "15": 0.12,
            "16": 0.004,
            "17": 0.015,
            "18": 0.012,
            "19": 0.018,
            "20": 0.013,
            "21": 0.026,
            "22": 0.013,
            "23": 0.031,
            "24": 0.006,
            "25": 0.006,
            "26": 0.009,
            "27": 0.013,
            "28": 0.009,
            "29": 0.021,
            "30": 0.007,
            "31": 0.031,
        }
    },
    {
        "histogram": {
            "0": 0.008,
            "1": 0.011,
            "2": 0.01,
            "3": 0.006,
            "4": 0.015,
            "5": 0.036,
            "6": 0.013,
            "7": 0.046,
            "8": 0.016,
            "9": 0.013,
            "10": 0.03,
            "11": 0.025,
            "12": 0.026,
            "13": 0.072,
            "14": 0.038,
            "15": 0.167,
            "16": 0.008,
            "17": 0.006,
            "18": 0.01,
            "19": 0.012,
            "20": 0.015,
            "21": 0.036,
            "22": 0.014,
            "23": 0.036,
            "24": 0.02,
            "25": 0.021,
            "26": 0.046,
            "27": 0.031,
            "28": 0.011,
            "29": 0.048,
            "30": 0.037,
            "31": 0.117,
        }
    },
    {
        "histogram": {
            "0": 0.039,
            "1": 0.016,
            "2": 0.028,
            "3": 0.027,
            "4": 0.022,
            "5": 0.029,
            "6": 0.02,
            "7": 0.023,
            "8": 0.047,
            "9": 0.019,
            "10": 0.063,
            "11": 0.042,
            "12": 0.035,
            "13": 0.026,
            "14": 0.038,
            "15": 0.023,
            "16": 0.058,
            "17": 0.026,
            "18": 0.02,
            "19": 0.025,
            "20": 0.036,
            "21": 0.044,
            "22": 0.026,
            "23": 0.019,
            "24": 0.034,
            "25": 0.018,
            "26": 0.048,
            "27": 0.029,
            "28": 0.03,
            "29": 0.026,
            "30": 0.033,
            "31": 0.031,
        }
    },
    {
        "histogram": {
            "0": 0.022,
            "1": 0.036,
            "2": 0.027,
            "3": 0.055,
            "4": 0.015,
            "5": 0.022,
            "6": 0.037,
            "7": 0.051,
            "8": 0.025,
            "9": 0.07,
            "10": 0.021,
            "11": 0.045,
            "12": 0.023,
            "13": 0.03,
            "14": 0.038,
            "15": 0.024,
            "16": 0.024,
            "17": 0.047,
            "18": 0.022,
            "19": 0.034,
            "20": 0.02,
            "21": 0.039,
            "22": 0.042,
            "23": 0.04,
            "24": 0.02,
            "25": 0.039,
            "26": 0.018,
            "27": 0.022,
            "28": 0.02,
            "29": 0.022,
            "30": 0.026,
            "31": 0.024,
        }
    },
]

"""
Hamiltonian for 24O
"""
coeffs = np.array(
    [
        -10.943495,
        6.590075,
        6.59009,
        6.59008,
        6.1535,
        -3.991875,
        -3.991875,
        -0.56607,
        -0.400535,
        -0.334915,
        -0.841875,
        -0.34829,
        -0.665395,
        -0.33492,
        -0.545725,
        -0.644445,
        -0.334915,
        -0.39765,
        -0.792515,
        -0.282125,
        -0.282125,
        -0.225,
        0.34782,
        0.34782,
        -0.182285,
        -0.182285,
        -0.419445,
        -0.419445,
        -0.667545,
        -0.667545,
        -0.24693,
        0.24693,
        0.44714,
        0.44714,
        0.419445,
        0.419445,
        0.415175,
        0.415175,
        0.4993,
        -0.4993,
        -0.419445,
        -0.419445,
        -0.28899,
        -0.28899,
        -0.625485,
        0.625485,
        -0.22309,
        -0.22309,
        -0.22309,
        0.22309,
        -0.27825,
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
