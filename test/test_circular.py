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
