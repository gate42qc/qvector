import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator


def bold(s):
    return '\033[1m' + s + '\033[0m'


def pairwise(iterable):
    from itertools import tee
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_new_qc(*registers):
    return QuantumCircuit(*registers)


def get_quantum_register(length: int):
    return QuantumRegister(length)


def get_classical_register(length: int):
    return ClassicalRegister(length)


def get_I(dim: int):
    return Operator(np.eye(2 ** dim))
