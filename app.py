import streamlit as st
import qiskit
import pennylane as qml
import quantum_algorithms
from pennylane import numpy as np

st.title('Quantum Supercomputer')
st.write('This is a basic interface for QuantumBridge.')

# Example quantum circuit
def create_quantum_circuit():
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# Display the circuit
circuit = create_quantum_circuit()
st.write(circuit.draw())

if st.button('Run Quantum Circuit'):
    backend = qiskit.Aer.get_backend('statevector_simulator')
    result = qiskit.execute(circuit, backend).result()
    st.write(result.get_statevector())

st.write('Quantum Neural Network')
weights = np.random.rand(2)
result = quantum_algorithms.quantum_neural_network(weights)
st.write('Result:', result)
