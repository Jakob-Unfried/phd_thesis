"""To be used in the `-m` argument of benchmark.py."""
import numpy as np
from su2_test_cases import generate_su2_tensors


def setup_benchmark(n_sites: int = 3, seed: int = 1, n_legs: int = 2, **kwargs):
    a, = generate_su2_tensors(n_tensors=1, n_sites=n_sites, n_legs=n_legs, seed=seed,
                              symmetry='pure_numpy', backend=None)
    return a, n_legs


def benchmark(data):
    a, n_legs = data
    N = np.prod(a.shape[:n_legs])
    a = np.reshape(a, (N, -1))
    np.linalg.svd(a)
