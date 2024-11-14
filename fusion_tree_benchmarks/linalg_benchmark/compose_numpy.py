"""To be used in the `-m` argument of benchmark.py."""
import numpy as np
from su2_test_cases import generate_su2_tensors


def setup_benchmark(n_sites: int = 3, seed: int = 1, n_legs: int = 2, **kwargs):
    a, b = generate_su2_tensors(n_tensors=2, n_sites=n_sites, n_legs=n_legs, seed=seed,
                                symmetry='pure_numpy', backend=None)
    axes = (list(range(n_legs, 2 * n_legs)), list(reversed(range(n_legs))))
    return a, b, axes


def benchmark(data):
    a, b, axes = data
    np.tensordot(a, b, axes)
