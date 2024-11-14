import numpy as np
import tenpy as tp


def main():
    backend = tp.get_backend('fusion_tree', 'numpy')
    L = 50
    eng = TEBD.from_heisenberg_dynamics(backend, L=L, dt=0.01, chi_max=200)
    eng.sweep()
    eng.half_chain_entropy()
    h_bond = np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]], complex)
    h_bond = np.transpose(np.reshape(h_bond, [2, 2, 2, 2]), [0, 1, 3, 2])  # [p0, p1, p1*, p0*]
    p = tp.ElementarySpace(tp.SU2Symmetry(), [[1]])
    h_bond = tp.SymmetricTensor.from_dense_block(h_bond, [p, p], [p, p], backend)
    eng.two_site_expvals([h_bond] * (L - 1))


class TEBD:

    def __init__(self, backend: tp.backends.TensorBackend, B_list: list[tp.SymmetricTensor],
                 S_list: list[tp.DiagonalTensor], U_list: list[tp.SymmetricTensor],
                 chi_max: int):
        self.backend = backend
        self.B_list = B_list
        self.U_list = U_list  # U_list[i] acting on sites [i] and [i+1], legs [p0, p1, p0*, p1*]
        self.L = L = len(B_list)
        assert len(S_list) == L + 1
        self.S_list = S_list  #  S_list[i] is between B_list[i-1] and B_list[i]
        self.svd_options = dict(chi_max=chi_max, svd_min=1e-14)

    def update_bond(self, i, U_bond):
        """Update the bond between sites ``i`` and ``i + 1``."""
        C = tp.tdot(self.B_list[i], self.B_list[i+1], 'vR', 'vL',
                    relabel1=dict(p='p0'), relabel2=dict(p='p1'))
        C = tp.tdot(C, U_bond, ['p0', 'p1'], ['p0*', 'p1*'])
        theta = tp.scale_axis(C, self.S_list[i], 'vL')
        U, S, Vh, err, renormalize = tp.truncated_svd(
            theta, ['vR', 'vL'], normalize_to=1, options=self.svd_options
        )
        B_R = Vh.relabel({'p1': 'p'})
        B_L = tp.tdot(C, tp.dagger(Vh), ['p1', 'vR'], ['p1', 'vR'], relabel1={'p0': 'p'}) / renormalize
        self.S_list[i + 1] = S
        self.B_list[i] = B_L
        self.B_list[i + 1] = B_R
        return err

    def sweep(self):
        err = 0
        for i, U_bond in enumerate(self.U_list):
            err += self.update_bond(i, U_bond)
        return err

    @classmethod
    def from_heisenberg_dynamics(cls, backend: tp.backends.TensorBackend, L: int, dt: float,
                                 chi_max: int):
        # build singlet covering MPS
        sym = tp.SU2Symmetry()
        p = tp.ElementarySpace(sym, [[1]])
        v0 = tp.ElementarySpace(sym, [[0]])

        X_tensor = sym.fusion_tensor(np.array([1]), np.array([1]), np.array([0]))[0, :, :, :]
        B_even = tp.SymmetricTensor.from_dense_block(
            np.transpose(X_tensor, [2, 1, 0]),  # [p, vR, vL] -> [vL, vR, p]
            [v0], [p, p], backend=backend, labels=['vL', 'vR', 'p']
        )
        B_odd = tp.SymmetricTensor.from_eye([p], backend, labels=['vL', 'p'])
        B_odd = tp.add_trivial_leg(B_odd, domain_pos=1, label='vR')
        for B in [B_even, B_odd]:
            assert B.domain_labels == ['p', 'vR']
            assert B.codomain_labels == ['vL']
        assert L % 2 == 0
        B_list = [B_even, B_odd] * (L // 2)
        # S_even: between B_odd and B_even, virtual bond has spin 0 -> number 1
        S_even = tp.DiagonalTensor.from_eye(v0, backend=backend, labels=['vL', 'vR'])
        # S_even: between B_even and B_odd, virtual bond has spin 1/2 -> normalized identity 
        S_odd = tp.DiagonalTensor.from_eye(p, backend=backend, labels=['vL', 'vR']) / np.sqrt(2)
        S_list = [S_even, S_odd] * (L // 2) + [S_even]

        # h_bond = -3 |sing><sing| + 1 \sum_i |trip_i><trip_i|
        # u_bond = e^{-i h_bond dt} = e^{+3idt} |sing><sing| + e^{-i dt} \sum_i |trip_i><trip_i|
        P_sing = np.array([[0, 0, 0, 0], [0, .5, -.5, 0], [0, -.5, .5, 0], [0, 0, 0, 0]], complex)
        P_trip = np.array([[1, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0], [0, 0, 0, 1]], complex)
        u_bond = np.exp(+3.j * dt) * P_sing + np.exp(-1.j * dt) * P_trip
        # (p0, p1, p0*, p1*) -> (p0, p1, p1*, p0*)
        u_bond = np.transpose(np.reshape(u_bond, (2, 2, 2, 2)), [0, 1, 3, 2])
        u_bond = tp.SymmetricTensor.from_dense_block(
            block=u_bond, codomain=[p, p], domain=[p, p], backend=backend,
            labels=['p0', 'p1', 'p1*', 'p0*']
        )
        U_list = [u_bond] * L

        return cls(backend=backend, B_list=B_list, S_list=S_list, U_list=U_list, chi_max=chi_max)


if __name__ == '__main__':
    main()
