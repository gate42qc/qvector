from qiskit import IBMQ
import numpy as np
from qiskit.quantum_info import random_unitary

from optimizer import LBFGSOptimizer
from qc_runner import CustomNoiseModelQcRunner
from utils import get_classical_register, get_quantum_register, get_new_qc, get_I, pairwise, bold

IBMQ.load_account()
NUMBER_OF_CYCLES_FOR_AVERAGE_FIDELITY_CALCULATION = 10
VERBOSE = False


def get_variational_parametrized_qc(qubits, params):
    register = get_quantum_register(len(qubits))
    qc = get_new_qc(register)
    
    i = 0
    parameters_are_exhausted = False
    while not parameters_are_exhausted:
        for qubit in qubits:
            # rotate qubit with an angle from parameters
            theta = params[i] * 2 * np.pi
            qc.rz(theta, qubit)
            i += 1
        
        for qubit in qubits:
            # rotate qubit with an angle from parameters
            theta = params[i] * 2 * np.pi
            qc.rx(theta, qubit)
            i += 1
        
        for qubit_pair in pairwise(qubits):
            # do control-Z rotations for neighbour qubits using angles from the params array
            theta = params[i] * 2 * np.pi
            qc.crz(theta, *list(qubit_pair))
            i += 1

        parameters_are_exhausted = i < len(params)
          
    return qc


def get_encode_qc(qubits, params):
    # on n qubits. k logical and n-k syndrome qubits
    return get_variational_parametrized_qc(qubits, params)


def get_recovery_qc(qubits, params):
    # on n + r qubits. k logical and n-k syndrome qubits, r is the number of refresh qubits
    # my guess is that this is the same as above?
    return get_variational_parametrized_qc(qubits, params)


def get_sample_state_qc(k):
    # k is the number of available logical qubits
    return random_unitary(2 ** k).to_instruction()


def estimate_average_fidelity(p, q, k, n, r, qc_runner, number_of_iterations=1):
    if VERBOSE:
        print("Running optimization with params p and q:")
        print("p =", p)
        print("q =", q)
    
    fidelities = []
    # We calculate the fidelity for L different S samples
    for i in range(NUMBER_OF_CYCLES_FOR_AVERAGE_FIDELITY_CALCULATION):
        qc = get_new_qc(get_quantum_register(n + r), get_classical_register(n + r))
        
        # sample the state S
        init_qc = get_sample_state_qc(k)
        
        encode_qc = get_encode_qc(range(n), p)
        recovery_qc = get_recovery_qc(range(n + r), q)
        
        noisy_I = get_I(n)
        
        init_reg = range(k)
        encode_reg = range(n)
        recover_reg = range(n + r)
        noise_reg = range(n)
        
        # init the state S
        # apply noisy encoding(q)
        # apply recovery(p)
        # apply (noisy encoding)^(dagger) = decoding(q)
        # apply (S)^(dagger)
        qc.append(init_qc, init_reg)
        qc.append(encode_qc, encode_reg)
        for i in range(number_of_iterations):
            qc.unitary(noisy_I, noise_reg, label='noisy_I')
            qc.append(recovery_qc, recover_reg)
        qc.append(encode_qc.inverse(), encode_reg)
        qc.append(init_qc.inverse(), init_reg)
        
        # run the circuit and get results
        # the refresh qubits should work without noise?
        fidelity = qc_runner.get_fidelity(qc)
        fidelities.append(fidelity)
    
    # estimate the average fidelity for q and p
    average_fidelity = sum(fidelities)/len(fidelities)
    if VERBOSE:
        print(bold(f"For current p and q average fidelity was: {average_fidelity}"))
        print("=========================================")
        print()
    
    return average_fidelity


def get_params_length(k, n, r):
    total_qubits = n + r
    params_per_cycle = int(total_qubits * 3) * 2  # x2 for p and q
    return params_per_cycle


def optimize(k, n, r, qc_runner, optimizer=LBFGSOptimizer(), initial_params=None, number_of_cycles=2):
    # k is the number of logical qubits, n-k is the number of syndrom qubits and r is the number of refresh qubits
    def init_params():
        params_per_cycle = get_params_length(k, n, r)
        return np.random.rand(number_of_cycles * params_per_cycle)
    
    def fn(x, args):
        p, q = x[:len(x)//2], x[len(x)//2:]
        return estimate_average_fidelity(p, q, *args)
    
    print("Optimization started")

    if initial_params is None:
        initial_params = init_params()
    
    return optimizer.optimize(fn, initial_params, [k, n, r, qc_runner])


if __name__ == "__main__":
    k, n, r = 1, 5, 3
    # k, n, r = 1, 3, 1
    cycles = 1
    params_len = get_params_length(k, n, r) * cycles

    f = estimate_average_fidelity(np.random.rand(params_len//2), np.random.rand(params_len//2), k, n, r,
                                  qc_runner=CustomNoiseModelQcRunner(n, r))
    print("Fidelity is: ", f)
    res = optimize(k, n, r, qc_runner=CustomNoiseModelQcRunner(n, r), optimizer=LBFGSOptimizer())
    print("Optimization results: ", res)

