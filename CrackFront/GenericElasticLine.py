import timeit

import numpy as np
import scipy.sparse.linalg
import pytest

class ElasticLine():
    def __init__(self, L, Lk, pinning_field):
        r"""
        Generic equation

        .. math::

            0 = \frac{1}{c} \delta_a \Pi(a, w, z) = q_\kappa (a(z) - w)  + \sum_n |q| \tilde a_n e^{iq_n z}  +  V^{(1)}(z, a(z)) / c

        For a crack front
            - :math: `c` is the mean work of adhesion
            - :math:`V^{(1)}(z, a(z))` is the deviation of the local work of adhesion from the mean.
            - :math:`q_\kappa=\frac{2\pi}{L_\kappa}` describes how fast the energy release rate changes with position
            for a straight crack:  :math:`q_\kappa=\partial_a G_0 / G_0`. Note that we used that :math:`G_0 \simeq c`.

        The pixel size is 0

        Parameters:
        -----------
        L: integer
            length of the line in pixels.
        Lk: float
            structural length
            :math:`q_k = 2 \pi L_k` is the prefactor of the quadratic driving potential
        pinning_field: callable

            parameters:
                - ``position`` : array of length `L`
                        index of the position along the crack front
                - ``derivative``: order of derivative (0 or 1)

            returns:
                - array of length `L` gradient or curvature of the pinning potential

        """

        self.L = self.npx = n = L

        self.z = np.arange(L) # position along the line (in the straight configuration)
        self.qk = 2 * np.pi / Lk
        self.pinning_field = pinning_field
        self._elastic_hessian = None
        self.q = 2 * np.pi * np.fft.rfftfreq(n, L / n)

    @property
    def elastic_hessian(self):
        """
        Computes and returns the elastic_hessian.

        the elastic jacobian is computed only at the first call and afterwards cached.
        """
        if self._elastic_hessian is None:
            npx = self.npx
            elastic_jac = np.zeros((npx, npx))
            v = np.fft.irfft(self.q, n=npx)
            for i in range(npx):
                for j in range(npx):
                    elastic_jac[i, j] = v[i - j]
            self._elastic_hessian = elastic_jac
        return self._elastic_hessian

    def elastic_hessp(self, a):
        r"""

        computes the elastic hessian product with one crack shape

        .. math ::
            \sum_n |q_n| \tilde a_n e^{i q_n z}
        """
        return np.fft.irfft(self.q * np.fft.rfft(a), n=self.npx)

    def elastic_potential(self, a, a_forcing):
        return 0.5 * self.qk * np.sum((a - a_forcing) ** 2) + 0.5 * np.sum(self.elastic_hessp(a) * a)

    def gradient(self, a, a_forcing):
        return self.elastic_hessp(a) + self.qk * (a - a_forcing) + self.pinning_field(a)

    def hessian_product(self, p, a):
        return self.qk * p + self.elastic_hessp(p) + self.pinning_field(a, der="1") * p

    def hessian_operator_cached(self, a):
        _pinning_curvature = self.pinning_field(a, der="1")
        def hessian_product(p):
            return self.qk * p + self.elastic_hessp(p) + _pinning_curvature * p
        return scipy.sparse.linalg.LinearOperator((self.L, self.L), matvec=hessian_product)

    def hessian(self, a):
        return np.diag(self.qk + self.pinning_field(a, der="1")) + self.elastic_hessian

    def eigenvalues(self, a, k=1):
        return scipy.sparse.linalg.eigsh(self.hessian_operator_cached(a), k=1, which="SA")
        #return scipy.sparse.linalg.eigsh(self.hessian(a), k=k, which="SA")

    def dump(self, ncFrame, a_forcing, a, dump_fields=True):
        """
        Writes the results of the current solution into the ncFrame
        """

        ncFrame.driving_position = a_forcing
        if dump_fields:
            ncFrame.position = a
        ncFrame.position_mean = mean_a = np.mean(a)
        ncFrame.position_rms = np.sqrt(np.mean((a - mean_a) ** 2))

        ncFrame.driving_force = - self.qk * (np.mean(a) - a_forcing)

        ncFrame.position_min = np.min(a)
        ncFrame.position_max = np.max(a)

        ncFrame.elastic_potential = self.elastic_potential(a, a_forcing)
