import streamlit as st
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import pennylane as qml
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

# Set up the Streamlit app
st.title("Quantum and AI Integration with Streamlit")

st.sidebar.title("Select Functionality")
functionality = st.sidebar.selectbox("Choose a functionality:", ["Quantum Computing", "Machine Learning", "Data Visualization"])

# Quantum Computing Section
if functionality == "Quantum Computing":
    st.header("Quantum Computing with Qiskit and Pennylane")

    st.subheader("Qiskit Example")
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    st.write(qc.draw(output="mpl"))

    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend)
    qobj = assemble(t_qc)
    result = backend.run(qobj).result()
    counts = result.get_counts()
    st.write(plot_histogram(counts))

    st.subheader("Pennylane Example")
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = circuit()
    st.write(f"Probabilities: {probs}")

# Machine Learning Section
elif functionality == "Machine Learning":
    st.header("Machine Learning with TensorFlow and PyTorch")

    st.subheader("TensorFlow Example")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    st.write("TensorFlow model summary:")
    model.summary(print_fn=lambda x: st.text(x))

    st.subheader("PyTorch Example")
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    torch_model = SimpleNN()
    st.write("PyTorch model structure:")
    st.text(str(torch_model))

# Data Visualization Section
elif functionality == "Data Visualization":
    st.header("Data Visualization with Matplotlib and Pandas")

    st.subheader("Matplotlib Example")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

    st.subheader("Pandas Example")
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100)
    })
    st.write(df.describe())

    st.line_chart(df)

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
