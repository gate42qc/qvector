from functools import lru_cache
from qiskit import execute, BasicAer
from qiskit import Aer, IBMQ
from qiskit.providers.aer import noise, QasmSimulator
from qiskit.providers.aer.noise import errors, NoiseModel


def get_fidelity(counts, n_qubits):
    return 1 - counts.get("0" * n_qubits, 0) / sum(counts.values())


def add_measure_instructions(qc, n_qubits):
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))


@lru_cache(maxsize=128)
def get_quantum_error(dim):
    error_param = 0.1  # damping parameter

    # Construct the error
    i_error = errors.amplitude_damping_error(error_param)
    for i in range(dim - 1):
        i_error = i_error.tensor(errors.amplitude_damping_error(error_param))

    return i_error

    # return errors.depolarizing_error(0.1, dim)


class DefaultQcRunner:
    def __init__(self, n, r):
        self.n = n
        self.r = r
        self.n_qubits = n + r

    def get_counts(self, qc):
        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(qc, backend)
        return job.result().get_counts(0)

    def get_fidelity(self, qc):
        add_measure_instructions(qc, self.n_qubits)
        return get_fidelity(self.get_counts(qc), self.n_qubits)


class DummyQcRunner(DefaultQcRunner):
    def get_fidelity(self, qc):
        return 1


class NoisyQcRunner(DefaultQcRunner):
    def __init__(self, n, r):
        super().__init__(n, r)

        # Choose a real device to simulate
        provider = IBMQ.get_provider(group='open')
        device = provider.get_backend('ibmq_16_melbourne')
        properties = device.properties()
        self.coupling_map = device.configuration().coupling_map

        # Generate an Aer noise model for device
        self.noise_model = noise.device.basic_device_noise_model(properties)
        self.basis_gates = self.noise_model.basis_gates

        self.backend = Aer.get_backend('qasm_simulator')

    def get_counts(self, qc):
        job_sim = execute(qc, self.backend, shots=1024,
                          coupling_map=self.coupling_map,
                          noise_model=self.noise_model,
                          basis_gates=self.basis_gates)

        sim_result = job_sim.result()
        return sim_result.get_counts()


class CustomNoiseModelQcRunner(DefaultQcRunner):
    def get_counts(self, qc):
        # Construct the error
        i_error = get_quantum_error(self.n)

        # Build the noise model by adding the error to the "iswap" gate
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(i_error, 'noisy_I')
        noise_model.add_basis_gates(['unitary'])

        # Execute on the simulator with noise
        job = execute(qc, QasmSimulator(), shots=1024,
                      basis_gates=noise_model.basis_gates,
                      noise_model=noise_model)
        result = job.result()
        return result.get_counts(qc)
