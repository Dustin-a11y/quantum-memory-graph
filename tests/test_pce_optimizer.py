import numpy as np

from quantum_memory_graph.pce_optimizer import pce_encode_adjacency


def test_pce_encode_adjacency_does_not_double_count_shared_qubits():
    encoding = {
        0: ("ZZZ", (0, 1, 2), 1.0),
        1: ("IZZ", (1, 2), 1.0),
    }
    adjacency = np.zeros((2, 2))
    adjacency[0, 1] = 1.0

    qubit_adjacency = pce_encode_adjacency(adjacency, encoding, n_qubits=3)

    expected = np.zeros((3, 3))
    for qi in encoding[0][1]:
        for qj in encoding[1][1]:
            if qi != qj:
                expected[qi, qj] = 0.5
    np.testing.assert_allclose(qubit_adjacency, expected)
