import os
import sys
import numpy as np

repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(repo_root, 'tenpy_v2_repo'))
import tenpy as tp
# Make sure we are using the correct tenpy
assert tp.version.version.startswith('2.')
phd_repo_root = os.path.dirname(os.path.dirname(__file__))
assert tp.__file__.startswith(phd_repo_root)


def generate_su2_tensors(n_tensors: int, n_sites: int, n_legs: int, seed: int, symmetry: str,
                         backend: tp.backends.TensorBackend | None):
    """Generate an SU(2) symmetric test case.

    Parameters
    ----------
    n_sites : int
        Determines the sectors on each leg.
        The sectors are the same as for the the tensor product of `n_sites` spin-1/2 sites.
    n_legs : int
        The number of legs. The generated tensors have ``2 * n_legs`` legs each, divided evenly
        between domain and codomain.
    seed : int
        A seed for the RNG
    symmetry : {'SU(2)', 'U(1)', 'None'}
        The symmetry for the leg to be returned. 

    Returns
    -------
    a, b
        If backend is None, dense numpy data for the test case tensors.
        If backend is given, tenpy tensors.
    """

    # generate SU(2) symmetric data, even if we dont enforce the symmetry
    spin_half_su2 = tp.ElementarySpace(tp.SU2Symmetry(), np.array([[1]]))
    leg_su2 = tp.ProductSpace([spin_half_su2] * n_sites).as_ElementarySpace()
    leg_su2._basis_perm = None
    co_domain_su2 = [leg_su2] * n_legs
    rng = np.random.default_rng(seed)
    
    def random_complex(size):
        return (2 * rng.random(size) - 1) + 1.j * (2 * rng.random(size) - 1)

    tensors = [
        tp.SymmetricTensor.from_block_func(random_complex, co_domain_su2, co_domain_su2)
        for _ in range(n_tensors)
    ]

    # parse symmetry
    if symmetry == 'SU(2)':
        return tensors

    tensors_np = [t.to_numpy() for t in tensors]

    if backend is None:
        return tensors_np
    
    if symmetry == 'U(1)':
        # can not build like leg_su2; the perms induced by forming the product space is different.
        sectors = []
        mults = []
        for jj, m in zip(leg_su2.sectors, leg_su2.multiplicities):
            jj = int(jj[0])
            dim = jj + 1  # d_j = 2j + 1
            sym = tp.u1_symmetry
            new_sectors = np.arange(-jj, jj + 2, 2)  # U(1) charges 2 * Sz
            assert len(new_sectors) == dim
            sectors.extend([s] for s in new_sectors)
            mults.extend([m] * dim)
        leg = tp.ElementarySpace.from_sectors(
            sym, np.array(sectors, int), np.array(mults)
        )
    elif symmetry in [None, 'none', 'None']:
        sym = tp.no_symmetry
        leg = tp.ElementarySpace.from_trivial_sector(leg_su2.dim, sym)
    else:
        raise ValueError(symmetry)

    co_domain = tp.ProductSpace([leg] * n_legs, backend=backend)
    tensors = [
        tp.SymmetricTensor.from_dense_block(t_np, co_domain, co_domain, backend=backend)
        for t_np in tensors_np
    ]
    return tensors
