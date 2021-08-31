import pytest
import timeit
from CrackFront.GenericElasticLine import ElasticLine
import scipy.sparse
import numpy as np


def test_elastic_hessp_vs_brute_force_elastic_hessian():
    npx = 32
    cf = ElasticLine(npx, 8, lambda x: None)
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

    cf = ElasticLine(npx, 8, potential)
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
    cf = ElasticLine(npx, Lk, potential)
    assert abs((cf.eigenvalues(np.zeros(npx))[0] - qk) / qk) < 1e-6


@pytest.mark.skip("performance benchmark")
def test_eigvalues_speed():
    npx = 32
    phases = np.random.uniform(size=npx) * 2 * np.pi

    q_potential = 2 * np.pi / 3

    def pinning_field(a, der="0"):
        if der == "0":
            return np.cos(a * q_potential + phases)
        elif der == "1":
            return - q_potential * np.sin(a * q_potential + phases)

    Lk = npx / 4
    qk = 2 * np.pi / Lk
    cf = ElasticLine(npx, Lk, pinning_field)
    a = np.random.normal(size=npx)
    cf.hessian(a)
    # assert the three methods give similar values
    sc_full = scipy.sparse.linalg.eigsh(cf.hessian(a), k=1, which="SA")[0]
    sc_sparse = scipy.sparse.linalg.eigsh(cf.hessian_operator_cached(a), k=1, which="SA")[0]
    np_ = np.linalg.eigvalsh(cf.hessian(a))[0]

    assert abs(sc_full - np_) < 1e-6
    assert abs(sc_sparse - np_) < 1e-6

    ######### Performance

    setup = """
from CrackFront.GenericElasticLine import ElasticLine
import numpy as np
import scipy.sparse.linalg
npx = 256

phases = np.random.uniform(size=npx) * 2 * np.pi

q_potential = 2 * np.pi / 3

def pinningfield(a, der="0"):
    if der == "0":
        return np.cos(a * q_potential + phases)
    elif der == "1":
        return - q_potential * np.sin(a * q_potential + phases)

Lk = npx / 4
qk = 2 * np.pi / Lk
cf = ElasticLine(npx, Lk, pinningfield)
a = np.random.normal(size=npx)
cf.hessian(a)
    """

    print("scipy, full matrix")
    print(
        np.mean(timeit.Timer('scipy.sparse.linalg.eigsh(cf.hessian(a), k=1, which="SA")', setup=setup).repeat(5, 10)))
    print("numpy, full matrix")
    print(np.mean(timeit.Timer('np.linalg.eigvalsh(cf.hessian(a))[0]', setup=setup).repeat(5, 10)))
    print("scipy. sparse")
    print(np.mean(
        timeit.Timer('scipy.sparse.linalg.eigsh(cf.hessian_operator_cached(a), k=1, which="SA")', setup=setup).repeat(
            5, 10)))

    # for 32, numpy is much faster.
    # for 1000, scipy is already faster
    # for 500 they are similar, scipy a bit faster

    # At 500 and above, the sparse version is faster
    # Above 2000, the sparse version speeds up by two orders of magnitude already


def test_hessian_product():
    penetration = 0

    w = 10

    w_amplitude = 0.4

    n_rays = 8
    L = npx = 32

    # l_waviness = 0.1
    q_waviness = 2 * np.pi / 0.1

    z = np.arange(L)

    def pinning_field(a, der="0"):
        if der == "0":
            return w_amplitude * np.cos(a * q_waviness) * np.cos(z / L * n_rays) * w
        elif der == "1":
            return - w * w_amplitude * q_waviness * np.sin(a * q_waviness) * np.cos(z / L * n_rays)

    cf = ElasticLine(npx, Lk=npx / 3, pinning_field=pinning_field)

    a = 0.1 + np.random.normal(size=npx) / 10
    da = np.random.normal(size=npx) * np.mean(a) / 10
    a_forcing = 0.1
    grad = cf.gradient(a, a_forcing)
    if False:
        hs = np.array([10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            grad_d = cf.gradient(a + h * da, a_forcing=a_forcing)
            dgrad = grad_d - grad
            dgrad_from_hess = cf.hessian_product(h * da, a)
            rms_errors.append(np.sqrt(np.mean((dgrad_from_hess - dgrad) ** 2)))

        # Visualize the quadratic convergence of the taylor expansion
        # What to expect:
        # Taylor expansion: g(x + h ∆x) - g(x) = Hessian * h * ∆x + O(h^2)
        # We should see quadratic convergence as long as h^2 > g epsmach,
        # the precision with which we are able to determine ∆g.
        # What is the precision with which the hessian product is made ?
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hs, rms_errors / hs ** 2
                , "+-")
        print(rms_errors)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        plt.show()

    hs = np.array([1e-2, 1e-3, 1e-4])
    rms_errors = []
    for h in hs:
        grad_d = cf.gradient(a + h * da, a_forcing=a_forcing)
        dgrad = grad_d - grad
        dgrad_from_hess = cf.hessian_product(h * da, a)
        rms_errors.append(np.sqrt(np.mean((dgrad_from_hess - dgrad) ** 2)))
        rms_errors.append(
            np.sqrt(
                np.mean(
                    (dgrad_from_hess.reshape(-1) - dgrad.reshape(-1)) ** 2)))

    rms_errors = np.array(rms_errors)
    assert rms_errors[-1] / rms_errors[0] < 1.5 * (hs[-1] / hs[0]) ** 2
