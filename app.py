import streamlit as st
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_ibm_runtime import QiskitRuntimeService
import pennylane as qml
import quantum_algorithms
from pennylane import numpy as np

st.title('Quantum Supercomputer')
st.write('This is a basic interface for QuantumBridge.')

# Load IBMQ account using QiskitRuntimeService
api_token = st.secrets["IBMQ_TOKEN"]
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token=api_token
)

# Example quantum circuit
def create_quantum_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# Display the circuit
circuit = create_quantum_circuit()
st.write(circuit.draw())

if st.button('Run Quantum Circuit'):
    backend = service.least_busy(['ibmq_qasm_simulator', 'ibmq_belem', 'ibmq_lima', 'ibmq_manila'])
    transpiled_circuit = transpile(circuit, backend)
    qobj = assemble(transpiled_circuit)
    job = backend.run(qobj)
    result = job.result()
    st.write(result.get_statevector())

st.write('Quantum Neural Network')
weights = np.random.rand(2)
result = quantum_algorithms.quantum_neural_network(weights)
st.write('Result:', result)
