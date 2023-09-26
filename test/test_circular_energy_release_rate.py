#
# Copyright 2021-2022 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import pytest
import numpy as np
from Adhesion.ReferenceSolutions import JKR
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationLin,
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc, SphereCrackFrontERRPenetrationFull
    )
from CrackFront.Optimization import trustregion_newton_cg
from Adhesion.ReferenceSolutions import JKR

@pytest.mark.parametrize("cfclass", [
    SphereCrackFrontERRPenetrationLin,
    SphereCrackFrontERRPenetrationFull,
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc
    ])
def test_circular_front_vs_jkr(cfclass):
    """
    assert we recover the JKR solution for an uniform distribution of
    work adhesion
    """
    _plot = False
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    cf = cfclass(n,
                 w=lambda a, theta: np.ones_like(a) * w,
                 dw=lambda a, theta: np.zeros_like(a), )

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
            hessian_product=lambda a, p: cf.hessian_product(p, a, penetration),
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
            plt.pause(0.0011)
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
    For a sinusoidal work of adhesion distribution,
    the shape of the crack front can be solved by hand (using the fully
    linearized version of the equaiton)

    For the work of adhesion distribution
    .. math ::

        w(\theta) = \left<w\right> (1 + dw \cos(n \theta))

    The contact radius takes the form

    .. math ::

        a(\theta) = a_0 + da \cos(\theta)

    with

    .. math ::

        da =  \frac{a_0 dw}{n\left<w\right>}

    and :math:`a_0` the solution of

    .. math ::

        \bar w = G(a_0, \Delta)

    """
    w = 1 / np.pi
    Es = 3. / 4

    w_amplitude = 0.4

    def w_landscape(radius, angle):
        return (1 + w_amplitude * np.cos(angle * n_rays)) * w

    def dw_landscape(radius, angle):
        return np.zeros_like(radius)

    cf = SphereCrackFrontERRPenetrationLin(npx,
                                           w=w_landscape,
                                           dw=dw_landscape)
    # initial guess:
    a = np.ones(npx) * JKR.contact_radius(penetration=penetration)
    sol = trustregion_newton_cg(
        x0=a, gradient=lambda a: cf.gradient(a, penetration),
        hessian_product=lambda a, p: cf.hessian_product(p, a, penetration),
        trust_radius=0.25 * np.min(a),
        maxiter=3000,
        gtol=1e-11)
    assert sol.success
    assert (abs(cf.gradient(sol.x, penetration)) < 1e-11).all()  #
    radii_cf = sol.x

    # Reference
    a0 = JKR.contact_radius(penetration=penetration)
    K = JKR.stress_intensity_factor(penetration=penetration, contact_radius=a0)
    dK = JKR.stress_intensity_factor(penetration=penetration, contact_radius=a0, der="1_a")

    da = w_amplitude * w / (K * dK / Es + n_rays * w / a0)

    radii_lin_by_hand = da * np.cos(n_rays * cf.angles) + a0

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(radii_lin_by_hand, "o", label="by hand")
        ax.plot(radii_cf, "+", label="general model")
        plt.show()
    np.testing.assert_allclose(radii_cf, radii_lin_by_hand)

# TODO: Test that all three models converge together in the small delta w limit

@pytest.mark.parametrize("cfclass", [
    SphereCrackFrontERRPenetrationLin,
    SphereCrackFrontERRPenetrationFull,
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc])
def test_hessian_product(cfclass):
    penetration = 0

    w = 1 / np.pi
    Es = 3. / 4

    w_amplitude = 0.4

    n_rays = 8
    npx = 32

    # l_waviness = 0.1
    q_waviness = 2 * np.pi / 0.1

    def w_landscape(radius, angle):
        return (1 + w_amplitude * np.cos(radius * q_waviness) * np.cos(angle * n_rays)) * w

    def dw_landscape(radius, angle):
        return - w * w_amplitude * q_waviness * np.sin(radius * q_waviness) * np.cos(angle * n_rays)

    cf = cfclass(npx,
                 w=w_landscape,
                 dw=dw_landscape)

    a = np.ones(npx) * JKR.contact_radius(penetration=penetration)
    da = np.random.normal(size=npx) * np.mean(a) / 10

    grad = cf.gradient(a, penetration)
    if False:
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            grad_d = cf.gradient(a + h * da, penetration)
            dgrad = grad_d - grad
            dgrad_from_hess = cf.hessian_product(h * da, a, penetration)
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
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        plt.show()

    hs = np.array([1e-2, 1e-3, 1e-4])
    rms_errors = []
    for h in hs:
        grad_d = cf.gradient(a + h * da, penetration)
        dgrad = grad_d - grad
        dgrad_from_hess = cf.hessian_product(h * da, a, penetration)
        rms_errors.append(np.sqrt(np.mean((dgrad_from_hess - dgrad) ** 2)))
        rms_errors.append(
            np.sqrt(
                np.mean(
                    (dgrad_from_hess.reshape(-1) - dgrad.reshape(-1)) ** 2)))

    rms_errors = np.array(rms_errors)
    assert rms_errors[-1] / rms_errors[0] < 1.5 * (hs[-1] / hs[0]) ** 2

@pytest.mark.parametrize("cfclass", [
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc
    ])
def test_jkr_elastic_energy(cfclass):
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    cf = cfclass(n,
                 w_radius_integral=lambda a, theta: a**2 / 2 * w * 2 * np.pi / n,
                 w_radius=lambda a, theta: a * w * 2 * np.pi / n,
                 dw_radius=lambda a, theta: w * 2 * np.pi / n, )

    penetration = 1.
    contact_radius = 0.6
    np.testing.assert_allclose(
        cf.elastic_energy(np.ones(n) * contact_radius, penetration),
        JKR.elastic_energy(contact_radius=contact_radius, penetration=penetration)
    )

@pytest.mark.parametrize("n", (8, 15))
@pytest.mark.parametrize("cfclass", [
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc
    ])
def test_surface_energy(cfclass, n):
    w = 1 / np.pi
    Es = 3. / 4

    cf = cfclass(n,
                 w_radius_integral=lambda a, theta: a**2 / 2 * w * 2 * np.pi / n,
                 w_radius=lambda a, theta: a * w * 2 * np.pi / n,
                 dw_radius=lambda a, theta: w * 2 * np.pi / n, )
    contact_radius = 0.6
    np.testing.assert_allclose(
        cf.surface_energy(np.ones(n) * contact_radius,),
        - contact_radius ** 2 *w * np.pi)


@pytest.mark.parametrize("cfclass", [
    SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationEnergyConstGc
    ])
def test_total_energy(cfclass):
    n = 8
    w = 1 / np.pi
    Es = 3. / 4

    cf = cfclass(n,
                 w_radius_integral=lambda a, theta: a**2 / 2 * w * 2 * np.pi / n,
                 w_radius=lambda a, theta: a * w * 2 * np.pi / n,
                 dw_radius=lambda a, theta: w * 2 * np.pi / n, )
    penetration = 1.
    contact_radius = 0.6
    np.testing.assert_allclose(
        cf.energy(np.ones(n) * contact_radius, penetration),
        - contact_radius ** 2 *w * np.pi
        + JKR.elastic_energy(contact_radius=contact_radius, penetration=penetration)
    )



#def test_circular_front_vs_jkr(cfclass):

def test_energy_vs_gradient():
    pass  # TODO: need to implement the surface energy term before.

@pytest.mark.parametrize("n",[8,9,512])
def test_n_an2_random(n):
    n = 8
    w = 1 / np.pi
    Es = 3. / 4
    cf = SphereCrackFrontERRPenetrationEnergy(n,
        w_radius_integral=lambda a, theta: a ** 2 / 2 * w * 2 * np.pi / n,
        w_radius=lambda a, theta: a * w * 2 * np.pi / n,
        dw_radius=lambda a, theta: w * 2 * np.pi / n, )
    a = 0.6 + np.random.normal(size=n)
    k = np.fft.fftfreq(len(a),1 / len(a))
    ak = np.fft.fft(a,norm="forward")
    ref = np.sum((abs(k) * ak * ak.conj() ))
    test = cf._n_an_2(a)
    np.testing.assert_allclose(test,ref)

    # This test fails
    # But the following doesn't, meaning that the error is in the high frequency components.
    # This also explains why this error has never been critical
@pytest.mark.parametrize("n", [8, 9, 512])
def test_n_an2_sinusoidal(n):
    n = 8
    w = 1 / np.pi
    Es = 3. / 4
    cf = SphereCrackFrontERRPenetrationEnergy(n,
                                              w_radius_integral=lambda a, theta: a ** 2 / 2 * w * 2 * np.pi / n,
                                              w_radius=lambda a, theta: a * w * 2 * np.pi / n,
                                              dw_radius=lambda a, theta: w * 2 * np.pi / n, )
    angle = np.arange(n) * 2 * np.pi/ n
    a = 0.6 + np.sin(3*angle)
    k = np.fft.fftfreq(len(a), 1 / len(a))
    ak = np.fft.fft(a, norm="forward")
    ref = np.sum((abs(k) * ak * ak.conj()))
    test = cf._n_an_2(a)
    np.testing.assert_allclose(test, ref)
