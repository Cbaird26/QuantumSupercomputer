import pennylane as qml
from pennylane import numpy as np

def quantum_neural_network(weights):
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    return circuit(weights)
