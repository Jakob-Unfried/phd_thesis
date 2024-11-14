"""To be used in the `-m` argument of benchmark.py."""
from su2_test_cases import tp, generate_su2_tensors


def setup_benchmark(n_sites: int = 3, seed: int = 1, n_legs: int = 2, symmetry: str = 'SU(2)',
                    symmetry_backend: str = 'fusion_tree', block_backend: str = 'numpy',
                    **kwargs):
    backend = tp.get_backend(symmetry_backend, block_backend)
    a, b = generate_su2_tensors(n_tensors=2, n_sites=n_sites, n_legs=n_legs, seed=seed,
                                symmetry=symmetry, backend=backend)
    return a, b


def benchmark(data):
    a, b = data
    tp.compose(a, b)
