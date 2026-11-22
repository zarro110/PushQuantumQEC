import os
import matplotlib.pyplot as plt
from iqm.qubit_selector.qubit_selector import *
from iqm.qubit_selector.qiskit_utils import get_circuit, CircuitType
from iqm.qiskit_iqm import IQMProvider

# Input your Resonance token
#token = "WPQmSui9oOgOHbYdAK6ifESf8GUmvFyAAyyEm/wanpEBof/7E7twA5LISX7dO1ep" # Replace with your actual token
#os.environ["IQM_TOKEN"] = token

#api_token = os.environ.get("IQM_TOKEN")
#print(api_token)
iqm_server_url = "https://resonance.iqm.tech"  # Replace with your IQM server URL
quantum_computer = "emerald"  # Replace with your quantum computer name if needed
provider = IQMProvider(iqm_server_url, quantum_computer = quantum_computer)
backend = provider.get_backend()

calibration_data = CalibrationDataManager().get_calibration_fidelities(backend)


def plot_calibration_data(key, data):
    two_qubit_keys = [CalibrationType.CZ.value, CalibrationType.CLIFFORD.value]
    coherence_keys = [CalibrationType.T1.value, CalibrationType.T2.value]
    every_second = key in two_qubit_keys
    coherence = key in coherence_keys
    xlabel = 'Pairs' if key in two_qubit_keys else 'Qubits'
    ylabel = 'Error' if key not in coherence_keys else 'Coherence in (us)'
    pairs = list(data.items())
    if every_second:
        pairs = pairs[::2]
    labels = [pair[0] for pair in pairs]

    if coherence:
        values = [pair[1] for pair in pairs]
    else:
        values = [1-pair[1] for pair in pairs]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{key} metrics')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

print(calibration_data)
print(backend.coupling_map)
plot_calibration_data(*calibration_data['CZ'])