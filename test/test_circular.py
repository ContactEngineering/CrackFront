from Adhesion.ReferenceSolutions import JKR
from CrackFront.Optimization import trustregion_newton_cg
from CrackFront.Circular import SphereCrackFrontPenetration
import numpy as np
import pytest


@pytest.mark.parametrize("lin", [True, False])
def test_circular_front_vs_jkr(lin):
    """
    assert we recover the JKR solution for an uniform distribution of
    work adhesion
    """
    _plot = False
    n = 8
    w = 1 / np.pi
    Es = 3. / 4
    Kc = np.sqrt(2 * Es * w)
    cf = SphereCrackFrontPenetration(n,
                                     kc=lambda a, theta: np.ones_like(a) * Kc,
                                     dkc=lambda a, theta: np.zeros_like(a),
                                     lin=lin)

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
@pytest.mark.parametrize("n_rays", [1, 8])
@pytest.mark.parametrize("npx", [2, 3, 128])
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

    cf = SphereCrackFrontPenetration(npx,
                                     kc=kc,
                                     dkc=dkc,
                                     lin=True)
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
            + JKR.stress_intensity_factor(a0, penetration) / (2 * a0)
    ) * np.cos(cf.angles) + a0

    np.testing.assert_allclose(radii_cf, radii_lin_by_hand)


def test_elastic_hessp_vs_brute_force_elastic_hessian():
    npx = 32
    cf = SphereCrackFrontPenetration(npx, lambda x: None, lambda x: None)
    a_test = np.random.normal(size=npx)
    np.testing.assert_allclose(cf.elastic_hessp(a_test),
                               cf.elastic_jacobian @ a_test)


def test_converges_to_linear():
    r"""
    asserts the less linearized model converges to the linearized one as the
    amplitude of sif fluctuations decrease.
    """
    pass
