# QVECTOR algorithm implementation

QVECTOR: an algorithm for device-tailored quantum error correction
Implemented based on this paper: [Johnson, Peter D., et al. "QVECTOR: an algorithm for device-tailored quantum error correction." arXiv preprint arXiv:1711.02249 (2017)](https://arxiv.org/abs/1711.02249).

# Usage

Using `optimize` function in `qvector.py` one can find optimal parameters for error correction for specific system.
 
After optimal parameters are found one can use the `get_encode_qc` functions to encode the state of the logical qubits to physical qubits
and `get_recovery_qc` to recover from errors.
