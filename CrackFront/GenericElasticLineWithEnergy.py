

import numpy as np
import pytest
from muFFT import FFT
import scipy.sparse.linalg

# TODO: some of this is redundant and could be inherited
class ElasticLinePotential():
    def __init__(self, L, Lk, pinning_potential):

        self.L = self.npx = n = L

        self.z = np.arange(L) # position along the line (in the straight configuration)
        self.qk = 2 * np.pi / Lk
        self.pinning_potential = pinning_potential
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

    def potential(self, a, a_forcing):
        return 0.5 * self.qk * np.sum((a - a_forcing) ** 2) + 0.5 * np.sum(self.elastic_hessp(a) * a) + np.sum(
            self.pinning_potential(a))
        # TODO: this can be optimized by either computing the sum in Fourier space or by computing gradient and energy together.

    def gradient(self, a, a_forcing):
        return self.elastic_hessp(a) + self.qk * (a - a_forcing) + self.pinning_potential(a, der="1")

    def hessian_product(self, p, a):
        return self.qk * p + self.elastic_hessp(p) + self.pinning_potential(a, der="2") * p

    def hessian(self, a):
        return np.diag(self.qk + self.pinning_potential(a, der="2")) + self.elastic_hessian

    def eigenvalues(self, a, k=1):
        return scipy.sparse.linalg.eigsh(self.hessian(a), k=k, which="SA")

    def dump(self, ncFrame, a_forcing, a, dump_fields=True):
        """
        Writes the results of the current solution into the ncFrame

        this assumes the trust-region-newton-cg has been used.
        """

        ncFrame.driving_position = a_forcing
        if dump_fields:
            ncFrame.position = a
        ncFrame.position_mean = mean_a = np.mean(a)
        ncFrame.position_rms = np.sqrt(np.mean((a - mean_a) ** 2))

        ncFrame.driving_force = - self.qk * (np.mean(a) - a_forcing)

        ncFrame.position_min = np.min(a)
        ncFrame.position_max = np.max(a)

        ncFrame.total_energy = self.potential(a, a_forcing)


class ElasticLinePotentialPreconditionned(ElasticLinePotential):
    def __init__(self, L, Lk, pinning_potential):
        super().__init__(L, Lk, pinning_potential)
        # self._elastic_hessian = None
        n = L

        self.fftengine = FFT((n,), #fft="fftw",
                     allow_temporary_buffer=False,
                     allow_destroy_input=True
                     )

        self.real_buffer = self.fftengine.register_halfcomplex_field("hc-real-space", 1)
        self.fourier_buffer = self.fftengine.register_halfcomplex_field("hc-fourier-space", 1)

        #self.
        self.q_hc = 2 * np.pi * np.fft.fftfreq(
                n,
                L / # physical size
                n)
        self.hc_coeffs = np.ones(n)
        if (n % 2 == 0):
            self.hc_coeffs[0] = 1
            self.hc_coeffs[1:n // 2] = 2
            self.hc_coeffs[n // 2] = 1
            self.hc_coeffs[n // 2 + 1:] = 2
        else:
            self.hc_coeffs[0] = 1
            self.hc_coeffs[1:] = 2

        self.halfcomplex_stiffness_diagonal = self.qk + abs(self.q_hc)
        self.preconditioner = np.sqrt(self.halfcomplex_stiffness_diagonal)
        # The new variable b_hc = preconditioner * a_hc

    def real_to_halfcomplex(self, a):
        self.real_buffer.array()[...] = a
        self.fftengine.hcfft(self.real_buffer, self.fourier_buffer)
        return self.fourier_buffer.array()[...].copy() * self.fftengine.normalisation

    def halfcomplex_to_real(self, a_hc):
        self.fourier_buffer.array()[...] = a_hc
        self.fftengine.ihcfft(self.fourier_buffer, self.real_buffer)
        return self.real_buffer.array()[...].copy()

    def halfcomplex_potential(self, a_hc, a_forcing, a_r=None):
        if a_r is None:
            a_r = self.halfcomplex_to_real(a_hc)

        a_hc[0] -= a_forcing

        elastic_potential = 0.5 * self.L * np.sum(self.halfcomplex_stiffness_diagonal * self.hc_coeffs * a_hc**2)
        adhesion_potential = np.sum(self.pinning_potential(a_r, der="0"))
        return elastic_potential + adhesion_potential

    def preconditioned_potential(self, b_hc, a_forcing, a_r=None):
        if a_r is None:
            a_hc = b_hc / self.preconditioner
            a_r = self.halfcomplex_to_real(a_hc)

        b_hc[0] -= self.preconditioner[0] * a_forcing

        elastic_potential = 0.5 * self.L * np.sum(self.hc_coeffs * b_hc**2)
        adhesion_potential = np.sum(self.pinning_potential(a_r, der="0"))
        return elastic_potential + adhesion_potential

    def halfcomplex_gradient(self, a_hc, a_forcing, a_r=None):
        if a_r is None:
            a_r = self.halfcomplex_to_real(a_hc)
        a_hc[0] -= a_forcing
        elastic_gradient = self.L * self.halfcomplex_stiffness_diagonal * self.hc_coeffs * a_hc
        adhesive_gradient = self.L * self.hc_coeffs * self.real_to_halfcomplex(self.pinning_potential(a_r, der="1"))
        return elastic_gradient + adhesive_gradient

    def preconditioned_gradient(self, b_hc, a_forcing, a_r=None):
        if a_r is None:
            a_hc = b_hc / self.preconditioner
            a_r = self.halfcomplex_to_real(a_hc)

        b_hc[0] -= self.preconditioner[0] * a_forcing

        elastic_potential = 0.5 * self.L * np.sum(self.hc_coeffs * b_hc**2)
        adhesion_potential = np.sum(self.pinning_potential(a_r, der="0"))
        a_hc[0] -= a_forcing
        elastic_gradient = self.L * self.hc_coeffs * b_hc
        adhesive_gradient = self.L * self.hc_coeffs / self.preconditioner * self.real_to_halfcomplex(self.pinning_potential(a_r, der="1"))
        return elastic_gradient + adhesive_gradient

import pytest

@pytest.fixture(params=[4,5])
def simple_elastic_line_preconditionned(request):
    npx = request.param
    phases = np.random.uniform(size=npx) * 2 * np.pi
    q_potential = 2 * np.pi / np.random.randint(1, 4, size=npx)
    amplitudes = np.random.normal(5,size=npx)
    def potential(a, der="0"):
        if der == "0":
            return amplitudes * np.cos(a * q_potential + phases)
        elif der == "1":
            return - amplitudes * q_potential * np.sin(a * q_potential + phases)
        elif der == "2":
            return - amplitudes * q_potential ** 2 * np.cos(a * q_potential + phases)
    return ElasticLinePotentialPreconditionned(npx, npx / 2, potential)

@pytest.fixture(params=[4,5, 100])
def purely_elastic_line_preconditionned(request):
    npx = request.param
    def potential(a, der="0"):
        if der == "0":
            return np.zeros(npx)
        elif der == "1":
            return np.zeros(npx)
        elif der == "2":
            return np.zeros(npx)
    return ElasticLinePotentialPreconditionned(npx, npx / 2, potential)

def test_r_hc_basic(simple_elastic_line_preconditionned):
    line = simple_elastic_line_preconditionned
    a_r_ori= np.ones(line.L)
    a_hc = line.real_to_halfcomplex(a_r_ori)
    assert a_hc[0] == 1
    #np.testing.assert_almost_equal(a_r_roundtrip, a_r_ori)


def test_r_hc_roundtrip(simple_elastic_line_preconditionned):
    line = simple_elastic_line_preconditionned
    a_r_ori= np.random.normal(size=line.L)
    a_r_roundtrip = line.halfcomplex_to_real(line.real_to_halfcomplex(a_r_ori))

    np.testing.assert_almost_equal(a_r_roundtrip, a_r_ori)

def test_hc_energy(simple_elastic_line_preconditionned):
    line = simple_elastic_line_preconditionned
    a = np.random.normal(size=line.L)
    a_forcing=np.random.normal()
    a_hc = line.real_to_halfcomplex(a)
    np.testing.assert_allclose(line.halfcomplex_potential(a_hc, a_forcing), line.potential(a, a_forcing))

def test_preconditioned_energy(simple_elastic_line_preconditionned):
    line = simple_elastic_line_preconditionned
    a = np.random.normal(size=line.L)
    a_forcing=np.random.normal()
    b_hc = line.real_to_halfcomplex(a) * line.preconditioner
    np.testing.assert_allclose(line.preconditioned_potential(b_hc, a_forcing), line.potential(a, a_forcing))


def test_hc_elastic_energy(purely_elastic_line_preconditionned):
    line =  purely_elastic_line_preconditionned
    a = np.random.normal(size=line.L)
    a_forcing=np.random.normal()
    a_hc = line.real_to_halfcomplex(a)
    np.testing.assert_allclose(line.halfcomplex_potential(a_hc, a_forcing), line.potential(a, a_forcing))


def test_halfcomplex_potential_gradient_consistency(simple_elastic_line_preconditionned):
    cf = simple_elastic_line_preconditionned
    npx = cf.L

    a = np.random.normal(size=npx)
    a_test = np.random.normal(size=npx)
    a_test /= np.linalg.norm(a_test)

    actual_gradient = np.dot(cf.halfcomplex_gradient(a, 0), a_test)

    epsilons = 10.**np.arange(-10, 10)
    fd_gradient = np.zeros_like(epsilons)
    #fd_gradient_potential = np.zeros_like(epsilons)

    print(actual_gradient)

    for i, eps in enumerate(epsilons):
        # This is the first order finite differences approximation
        # the error is supposed to scale as epsilon
        fd_gradient[i] = (cf.halfcomplex_potential(eps * a_test + a, 0) - cf.halfcomplex_potential(a, 0)) / eps
        #fd_gradient_potential[i] = (np.sum(potential(eps * a_test + a)) - np.sum(potential(a)) )/ eps

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.loglog(epsilons, abs((fd_gradient - actual_gradient)) #/ epsilons
                        , "+")
    #ax.loglog(epsilons, abs((fd_gradient_potential - np.dot(potential(a, der="1") , a_test ))) #/ epsilons
    #                , "x")
    #ax.set_ylim(10.**-2, 10.**2)
    ax.plot(epsilons, epsilons, "k")
    plt.show()


def test_preconditioned_potential_gradient_consistency(simple_elastic_line_preconditionned):
    cf = simple_elastic_line_preconditionned
    npx = cf.L

    a = np.random.normal(size=npx)
    a_test = np.random.normal(size=npx)
    a_test /= np.linalg.norm(a_test)

    actual_gradient = np.dot(cf.preconditioned_gradient(a, 0), a_test)

    epsilons = 10.**np.arange(-10, 10)
    fd_gradient = np.zeros_like(epsilons)
    #fd_gradient_potential = np.zeros_like(epsilons)

    print(actual_gradient)

    for i, eps in enumerate(epsilons):
        # This is the first order finite differences approximation
        # the error is supposed to scale as epsilon
        fd_gradient[i] = (cf.preconditioned_potential(eps * a_test + a, 0) - cf.preconditioned_potential(a, 0)) / eps
        #fd_gradient_potential[i] = (np.sum(potential(eps * a_test + a)) - np.sum(potential(a)) )/ eps

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.loglog(epsilons, abs((fd_gradient - actual_gradient)) #/ epsilons
                        , "+")
    #ax.loglog(epsilons, abs((fd_gradient_potential - np.dot(potential(a, der="1") , a_test ))) #/ epsilons
    #                , "x")
    #ax.set_ylim(10.**-2, 10.**2)
    ax.plot(epsilons, epsilons, "k")
    plt.show()


########## ELASTICLINEPOTENTIAL tests


def test_potential_gradient_consistency():
    npx = 4

    phases = np.random.uniform(size=npx) * 2 * np.pi

    q_potential = 2 * np.pi / np.random.randint(1, 4, size=npx)

    amplitudes = np.random.normal(5,size=npx)


    def potential(a, der="0"):
        if der == "0":
            return amplitudes * np.cos(a * q_potential + phases)
        elif der == "1":
            return - amplitudes * q_potential * np.sin(a * q_potential + phases)
        elif der == "2":
            return - amplitudes * q_potential ** 2 * np.cos(a * q_potential + phases)

    cf = ElasticLinePotential(npx, npx / 2, potential)
    a = np.random.normal(size=npx)
    a_test = np.random.normal(size=npx)
    a_test /= np.linalg.norm(a_test)

    actual_gradient = np.dot(cf.gradient(a, 0), a_test)

    epsilons = 10.**np.arange(-10, 10)
    fd_gradient = np.zeros_like(epsilons)
    fd_gradient_potential = np.zeros_like(epsilons)

    print(actual_gradient)

    for i, eps in enumerate(epsilons):
        # This is the first order finite differences approximation
        # the error is supposed to scale as epsilon
        fd_gradient[i] = (cf.potential(eps * a_test + a, 0) - cf.potential(a, 0)) / eps
        fd_gradient_potential[i] = (np.sum(potential(eps * a_test + a)) - np.sum(potential(a)) )/ eps
    print(fd_gradient_potential)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.loglog(epsilons, abs((fd_gradient - actual_gradient)) #/ epsilons
                        , "+")
    ax.loglog(epsilons, abs((fd_gradient_potential - np.dot(potential(a, der="1") , a_test ))) #/ epsilons
                    , "x")
    #ax.set_ylim(10.**-2, 10.**2)
    ax.plot(epsilons, epsilons, "k")
    plt.show()

def test_elastic_hessp_vs_brute_force_elastic_hessian():
    npx = 32
    cf = ElasticLinePotential(npx, 8, lambda x: None)
    a_test = np.random.normal(size=npx)
    np.testing.assert_allclose(cf.elastic_hessp(a_test),
                               cf.elastic_hessian @ a_test)


def test_hessp_vs_brute_force_hessian():
    npx = 32

    phases = np.random.uniform(size=npx) * 2 * np.pi

    q_potential = 2 * np.pi / 3

    def potential(a, der="0"):
        if der == "0":
            return np.cos(a * q_potential + phases)
        elif der == "1":
            return - q_potential * np.sin(a * q_potential + phases)
        elif der == "2":
            return - q_potential ** 2 * np.cos(a * q_potential + phases)

    cf = ElasticLinePotential(npx, 8, potential)
    a = np.random.normal(size=npx)
    a_test = np.random.normal(size=npx)
    np.testing.assert_allclose(cf.hessian_product(a_test, a),
                               cf.hessian(a) @ a_test)

def test_smallest_eigenvalue_no_disorder():
    npx = 32

    def potential(a, der="0"):
        return np.zeros_like(a)

    Lk = 8
    qk = 2 * np.pi / Lk
    cf = ElasticLinePotential(npx, Lk, potential)
    assert abs((cf.eigenvalues(np.zeros(npx))[0] - qk) / qk) < 1e-6