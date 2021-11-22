import numpy as np
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced

from Adhesion.ReferenceSolutions import JKR

from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator

w = 1 / np.pi
Es = .75
R = 1


def test_JKR_single():
    npx_front = 100
    # npx_front can't be too low because there is a nonlinear part in the elasticity (coming from the JKR solution)
    # that we ignore
    #
    # Having more pixels increases the stiffness associated with moving one pixel at a time, so we make more
    # careful steps where the nonlinearity is not a problem

    sample_radii = np.linspace(0.1, 2.5, 20)

    values = w * np.ones((npx_front, len(sample_radii)))

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        values * sample_radii * 2 * np.pi / npx_front,
        sample_radii)

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    a = np.ones(npx_front) * 0.1

    penetration = 0.5
    sol = cf.rosso_krauth(a, penetration, gtol=1e-10, maxit=1000, direction=1, logger=Logger("RK.log", outevery=1))
    assert sol.success
    np.testing.assert_allclose(sol.x, JKR.contact_radius(penetration=penetration))


def test_JKR_curve():
    npx_front = 100
    # npx_front can't be too low because there is a nonlinear part in the elasticity (coming from the JKR solution)
    # that we ignore
    #
    # Having more pixels increases the stiffness associated with moving one pixel at a time, so we make more
    # careful steps where the nonlinearity is not a problem

    sample_radii = np.linspace(0.1, 2.5, 20)

    values = w * np.ones((npx_front, len(sample_radii)))

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        values * sample_radii * 2 * np.pi / npx_front,
        sample_radii)

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    cf.propagate_rosso_krauth(
        penetrations=penetrations_generator(0.1, 0.5),
        gtol=1e-10,
        maxit=10000,
        file="RK_numpy.nc",
        )

    nc = NCStructuredGrid("RK_numpy.nc")
    np.testing.assert_allclose(nc.mean_radius,
                               [JKR.contact_radius(penetration=nc.penetration[i]) for i in range(len(nc))],
                               rtol=1e-7)
