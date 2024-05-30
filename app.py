import streamlit as st
from qiskit import QuantumCircuit, transpile, assemble, IBMQ
from qiskit.providers.ibmq import least_busy
import pennylane as qml
import quantum_algorithms
from pennylane import numpy as np

st.title('Quantum Supercomputer')
st.write('This is a basic interface for QuantumBridge.')

# Load IBMQ account
api_token = st.secrets["526944faf27aeedd952653ff44c8f585f40beb9812cfaca7d4d415ebd4ead190e32ee32ef158ce92c18247d669d2d335bce731f934973d3206b12cf74d91ed2d"]
IBMQ.enable_account(api_token)
provider = IBMQ.get_provider(hub='ibm-q')

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
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 2 and
                                           not x.configuration().simulator and x.status().operational==True))
    transpiled_circuit = transpile(circuit, backend)
    qobj = assemble(transpiled_circuit)
    job = backend.run(qobj)
    result = job.result()
    st.write(result.get_statevector())

st.write('Quantum Neural Network')
weights = np.random.rand(2)
result = quantum_algorithms.quantum_neural_network(weights)
st.write('Result:', result)
