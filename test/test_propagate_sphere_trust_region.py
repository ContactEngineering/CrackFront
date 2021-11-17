import numpy as np
import pytest
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid

from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationEnergyConstGc,
    SphereCFPenetrationEnergyConstGcPiecewiseLinearField
    )
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator, simulate_crack_front

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.
maugis_K = 1.

def test_propagate_sphere_trust_region_vs_JKR():
    # nondimensional units following Maugis Book:
    Es = 3 / 4
    w = 1 / np.pi
    R = 1.
    maugis_K = 1.

    npx = 64

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(npx,
                                                     w=lambda radius, angle: np.ones_like(radius) * w,
                                                     dw=lambda radius, angle: np.zeros_like(radius) * w,)

    pulloff_radius = (np.pi * w * R**2 / 6 * maugis_K)**(1/3)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(0.1, 1.),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=0.5,
        dump_fields=False,
        gtol=1e-08
    )

    nc = NCStructuredGrid("trust.nc")
    np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    np.testing.assert_allclose(nc.mean_radius, [JKR.contact_radius(penetration=p) for p in nc.penetration[:]], rtol=1e-06)

@pytest.mark.skip("just plotting")
def test_linear_interpolated_pinning_field_derivative():
    npx_front = 16
    n_radii = 100

    sample_radii = np.linspace(0.5, 6, n_radii)
    w_values = np.ones((npx_front, 1)) * sample_radii.reshape(1, -1)
    w_values = np.random.normal(size=(npx_front, n_radii))

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(w_values * sample_radii * 2 * np.pi / npx_front,
                                                                        sample_radii)
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w)

    a = 3 + np.random.uniform(0, 1, size=npx_front)
    da = np.random.uniform(0, 1, size=npx_front)

    w = cf.piecewise_linear_w(a)

    if True:
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            dw = cf.piecewise_linear_w(a + h * da) - w
            dw_from_derivative = cf.piecewise_linear_w(a, der="1") * h * da
            rms_errors.append(np.sqrt(np.mean((dw_from_derivative - dw) ** 2)))

        # Visualize the quadratic convergence of the taylor expansion
        # What to expect:
        # Taylor expansion: g(x + h ∆x) - g(x) = Hessian * h * ∆x + O(h^2)
        # We should see quadratic convergence as long as h^2 > g epsmach,
        # the precision with which we are able to determine ∆g.
        # What is the precision with which the hessian product is made ?
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hs, rms_errors# / hs ** 2
                , "+-")
        print(rms_errors)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        plt.show()



def test_linear_interpolated_pinning_field_vs_JKR():
    """
    Here again we simply compute the JKR contact,
    but we check that the linearly interpolated pinning field is not buggy
    """

    npx_front = 64
    n_radii = 23

    pulloff_radius = (np.pi * w * R**2 / 6 * maugis_K)**(1/3)

    sample_radii = np.linspace(0.5 * pulloff_radius, 6, n_radii)
    w_values = np.ones((npx_front, n_radii)) * w
    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(w_values * sample_radii * 2 * np.pi / npx_front,
                                                                        sample_radii)
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(0.1, 1.),
        filename="trust.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=0.5,
        dump_fields=False,
        gtol=1e-08
    )

    nc = NCStructuredGrid("trust.nc")
    np.testing.assert_allclose(nc.force, [JKR.force(penetration=p) for p in nc.penetration[:]], rtol=1e-06)
    np.testing.assert_allclose(nc.mean_radius, [JKR.contact_radius(penetration=p) for p in nc.penetration[:]], rtol=1e-06)