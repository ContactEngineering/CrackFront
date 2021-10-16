import numpy as np
from ContactMechanics.Tools.Logger import Logger

from CrackFront.CircularEnergyReleaseRate import SphereCrackFrontERRPenetrationEnergyConstGc, RadiusTooLowError
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced

from Adhesion.ReferenceSolutions import JKR

w = 1 / np.pi
Es = .75
R = 1


def test_JKR_single():
    L = 100
    # L can't be too low because there is a nonlinear part in the elasticity (coming from the JKR solution)
    # that we ignore
    #
    # Having more pixels increases the stiffness associated with moving one pixel at a time, so we make more
    # careful steps where the nonlinearity is not a problem

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx=L, w=lambda x: x, dw = lambda x: x, wm=w)

    sample_radii = np.linspace(0.1, 2.5, 20)

    values = - w * np.ones((L, len(sample_radii)))

    pinning_field = linear_interpolated_pinning_field_equaly_spaced(
        values * sample_radii * 2 * np.pi / cf.npx,
        sample_radii)

    cf.pinning_field = pinning_field
    a = np.ones(L) * 0.1

    penetration = 0.5
    sol = cf.rosso_krauth(a, penetration, gtol=1e-10, maxit=1000, dir=1, logger=Logger("RK.log", outevery=1))
    assert sol.success
    np.testing.assert_allclose(sol.x, JKR.contact_radius(penetration=penetration))


def test_JKR_curve():
    L = 100
    # L can't be too low because there is a nonlinear part in the elasticity (coming from the JKR solution)
    # that we ignore
    #
    # Having more pixels increases the stiffness associated with moving one pixel at a time, so we make more
    # careful steps where the nonlinearity is not a problem

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx=L, w=lambda x: x, dw = lambda x: x, wm=w)

    sample_radii = np.linspace(0.1, 2.5, 20)

    values = - w * np.ones((L, len(sample_radii)))

    pinning_field = linear_interpolated_pinning_field_equaly_spaced(
        values * sample_radii * 2 * np.pi / cf.npx,
        sample_radii)

    cf.pinning_field = pinning_field
    a = np.ones(L) *1.
    penetration = 0.0
    dpen = 0.1
    dir=1
    maxpen = 0.5
    while True:
        print(penetration)
        try:
            sol = cf.rosso_krauth(a, penetration, gtol=1e-10, maxit=100000, dir=dir, logger=Logger("RK.log", outevery=1))
        except RadiusTooLowError:
            print("lost contact")
            break
        assert sol.success
        a = sol.x
        np.testing.assert_allclose(sol.x, JKR.contact_radius(penetration=penetration))

        if penetration > maxpen:
            dir = -1
        penetration += dir * dpen
