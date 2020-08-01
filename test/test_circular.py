from Adhesion.ReferenceSolutions import JKR
from muFFT.NetCDF import NCStructuredGrid

from CrackFront.Optimization import trustregion_newton_cg
from CrackFront.Circular import (
    SphereCrackFrontPenetration,
    SphereCrackFrontPenetrationLin
    )
import numpy as np
import pytest


@pytest.mark.parametrize("cfclass", [SphereCrackFrontPenetration,
                                     SphereCrackFrontPenetrationLin])
def test_circular_front_vs_jkr(cfclass):
    """
    assert we recover the JKR solution for an uniform distribution of
    work adhesion
    """
    _plot = False
    n = 8
    w = 1 / np.pi
    Es = 3. / 4
    Kc = np.sqrt(2 * Es * w)
    cf = cfclass(n,
                 kc=lambda a, theta: np.ones_like(a) * Kc,
                 dkc=lambda a, theta: np.zeros_like(a), )

    # initial guess
    penetrations = np.concatenate((np.linspace(0.001, 1, endpoint=False),
                                   np.linspace(1, -0.9)))
    radii = []
    a = np.ones(cf.npx) * JKR.contact_radius(penetration=penetrations[0])

    if _plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _a = np.linspace(0, 4)
        ax.plot(JKR.penetration(_a), _a)
        ax.axvline(penetrations[-1])
        plt.pause(0.001)

    for penetration in penetrations:
        sol = trustregion_newton_cg(
            x0=a, gradient=lambda a: cf.gradient(a, penetration),
            hessian=lambda a: cf.hessian(a, penetration),
            trust_radius=0.1 * np.min(a),
            maxiter=3000,
            gtol=1e-11)
        assert sol.success
        contact_radius = np.mean(sol.x)
        radii.append(contact_radius)
        assert abs(np.max(sol.x) - contact_radius) < 1e-10
        assert abs(np.min(sol.x) - contact_radius) < 1e-10
        if _plot:
            ax.plot(penetration, contact_radius, "+")
            plt.pause(0.00011)
        assert abs(penetration - JKR.penetration(contact_radius, )) < 1e-10

        a = sol.x


@pytest.mark.parametrize("penetration", [-0.4, 1.])
@pytest.mark.parametrize("npx, n_rays", [(2, 1),
                                         (3, 1),
                                         (17, 8),
                                         (16, 8),  # smallest possible
                                         #  discretisation !
                                         (128, 1),
                                         (128, 8)])
def test_single_sinewave(penetration, n_rays, npx):
    r"""
    For a sinusoidal stress intensity factor fluctuation,
    the shape of the crack front can be solved by hand (using the fully
    linearized version of the equaiton)

    For the stress intensity factor distribution
    .. math ::

        K_c(\theta) = \bar K_c (1 + dK \cos(n \theta))

    The contact radius takes the form

    .. math ::

        a(\theta) = a_0 + da \cos(\theta)

    with

    .. math ::

        da = \frac{\bar K_c dK}{
        \frac{\partial K^0}{\partial da}(a_0, \Delta) +
        \frac{|n| K^0(a_0, \Delta)}{2 a_0}
        }

    and :math:`a_0` the solution of

    .. math ::

        \bar K_c = K^0(a_0, \Delta)

    """
    w = 1 / np.pi
    Es = 3. / 4
    mean_Kc = np.sqrt(2 * Es * w)

    dK = 0.4

    def kc(radius, angle):
        return (1 + dK * np.cos(angle * n_rays)) * mean_Kc

    def dkc(radius, angle):
        return np.zeros_like(radius)

    cf = SphereCrackFrontPenetrationLin(npx,
                                        kc=kc,
                                        dkc=dkc)
    # initial guess: 
    a = np.ones(npx) * JKR.contact_radius(penetration=penetration)
    sol = trustregion_newton_cg(
        x0=a, gradient=lambda a: cf.gradient(a, penetration),
        hessian=lambda a: cf.hessian(a, penetration),
        trust_radius=0.25 * np.min(a),
        maxiter=3000,
        gtol=1e-11)
    assert sol.success
    assert (abs(cf.gradient(sol.x, penetration)) < 1e-11).all()  #
    radii_cf = sol.x

    # Reference
    a0 = JKR.contact_radius(penetration=penetration)
    radii_lin_by_hand = dK * mean_Kc / (
            JKR.stress_intensity_factor(a0, penetration, der="1_a")
            + JKR.stress_intensity_factor(a0, penetration) * n_rays / (2 * a0)
    ) * np.cos(n_rays * cf.angles) + a0

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(radii_lin_by_hand, "o", label="by hand")
        ax.plot(radii_cf, "+", label="general model")
        plt.show()
    np.testing.assert_allclose(radii_cf, radii_lin_by_hand)


def test_elastic_hessp_vs_brute_force_elastic_hessian():
    npx = 32
    cf = SphereCrackFrontPenetration(npx, lambda x: None, lambda x: None)
    a_test = np.random.normal(size=npx)
    np.testing.assert_allclose(cf.elastic_hessp(a_test),
                               cf.elastic_jacobian @ a_test)


@pytest.mark.parametrize("penetration", [-0.4, 1.])
def test_converges_to_linear(penetration):
    r"""
    asserts the less linearized model converges to the linearized one as the
    amplitude of sif fluctuations decrease.
    """

    n_rays = 1
    npx = 16

    w = 1 / np.pi
    Es = 3. / 4
    mean_Kc = np.sqrt(2 * Es * w)

    dKs = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.001, 0.0005]
    errors = []
    for dK in dKs:

        def kc(radius, angle):
            return (1 + dK * np.cos(angle * n_rays)) * mean_Kc

        def dkc(radius, angle):
            return np.zeros_like(radius)

        cf_lin = SphereCrackFrontPenetrationLin(
            npx,
            kc=kc,
            dkc=dkc)

        cf = SphereCrackFrontPenetration(
            npx,
            kc=kc,
            dkc=dkc, )

        a = np.ones(npx) * JKR.contact_radius(penetration=penetration)
        sol = trustregion_newton_cg(
            x0=a, gradient=lambda a: cf_lin.gradient(a, penetration),
            hessian=lambda a: cf_lin.hessian(a, penetration),
            trust_radius=0.25 * np.min(a),
            maxiter=3000,
            gtol=1e-11)
        assert sol.success
        assert (abs(cf_lin.gradient(sol.x, penetration)) < 1e-11).all()  #
        radii_cf_lin = sol.x

        sol = trustregion_newton_cg(
            x0=a, gradient=lambda a: cf.gradient(a, penetration),
            hessian=lambda a: cf.hessian(a, penetration),
            trust_radius=0.25 * np.min(a),
            maxiter=3000,
            gtol=1e-11)
        assert sol.success
        assert (abs(cf.gradient(sol.x, penetration)) < 1e-11).all()  #
        radii_cf = sol.x

        mean_error = np.mean(radii_cf - radii_cf_lin)
        fluct_error = radii_cf - radii_cf_lin - mean_error
        errors.append(np.sqrt(np.sum((radii_cf - radii_cf_lin) ** 2)))
    errors = np.array(errors)
    dKs = np.array(dKs)
    rel_errors = errors / dKs  # since the amplitude of the radius
    # flucutation is proportional to dK
    # assert it gets better for smaller dK
    assert ((rel_errors[1:] - rel_errors[:-1]) < 0).all()
    # verify error has approximately linearity in dK
    assert rel_errors[-1] / rel_errors[0] < 10 * dKs[-1] / dKs[0]

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(dKs, np.array(errors) / np.array(dKs))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_aspect(1)
        ax.grid()
        plt.show()

        # the linear approximation has errors in the absolute value of order
        # da**2
        # the K / dK has hence errors scaling linearly with dK
        # this is what we see in this plot

@pytest.mark.parametrize("cfclass", [SphereCrackFrontPenetration,
                                 SphereCrackFrontPenetrationLin])
def test_dump(cfclass):
    penetration = 0.
    n_rays = 2
    npx = 64

    w = 1 / np.pi
    Es = 3. / 4
    mean_Kc = np.sqrt(2 * Es * w)

    dK = 0.5
    lcor = 0.1
    errors = []

    def kc(radius, angle):
        return (1 + dK * np.cos(angle * n_rays) * np.cos(radius / lcor) ) * mean_Kc

    def dkc(radius, angle):
        return (1 - dK / lcor * np.cos(angle * n_rays) * np.sin(radius / lcor) ) * mean_Kc


    cf = cfclass(
        npx,
        kc=kc,
        dkc=dkc, )
    i = 0
    nc = NCStructuredGrid("test_dump.nc", "w", (npx,))
    a = np.ones(npx) * JKR.contact_radius(penetration=penetration) * 0.25
    sol = trustregion_newton_cg(
        x0=a, gradient=lambda a: cf.gradient(a, penetration),
        hessian_product=lambda a, p:
        cf.hessian_product(p, radius=a, penetration=penetration),
        trust_radius=0.25 * min(lcor, np.min(a)),
        maxiter=3000,
        gtol=1e-11)
    assert sol.success
    print(sol.nit)
    cf.dump(nc[i], penetration, sol)

@pytest.mark.parametrize("cfclass", [SphereCrackFrontPenetration,
                                     SphereCrackFrontPenetrationLin])
def test_hessp_and_hessian_equivalent(cfclass):
    n_rays = 1
    npx = 16

    w = 1 / np.pi
    Es = 3. / 4
    mean_Kc = np.sqrt(2 * Es * w)

    dK = 0.4
    penetration=0

    def kc(radius, angle):
        return (1 + dK * np.cos(angle * n_rays)) * mean_Kc

    def dkc(radius, angle):
        return np.zeros_like(radius)

    cf = cfclass(
        npx,
        kc=kc,
        dkc=dkc, )

    a = 1 + np.random.uniform(-0.5, 0.5, size=npx)
    p = np.random.uniform(-0.5, 0.5, size=npx)
    bruteforce = cf.hessian(radius=a, penetration=penetration) @ p
    hessp = cf.hessian_product(p, radius=a, penetration=penetration)

    np.testing.assert_allclose(hessp, bruteforce)