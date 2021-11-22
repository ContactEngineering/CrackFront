
import numpy as np
import pytest
from Adhesion.ReferenceSolutions import JKR
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography import Topography

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationEnergyConstGc,
    SphereCFPenetrationEnergyConstGcPiecewiseLinearField, SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationLin
    )
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_pytorch import propagate_rosso_krauth
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator, simulate_crack_front

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.
maugis_K = 1.
mean_Kc = np.sqrt(2 * Es * w)


def test_axissymmetric_sinewave_linear_interp():
    n_radii = 400
    npx_front = 8

    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    sample_radii = np.linspace(0.5 * pulloff_radius, 6, n_radii)

    wavelength = 0.1

    dw = 0.2

    def local_w(radius):
        return w * (1 + dw * np.sin(2 * np.pi / wavelength * radius))

    def local_dw(radius):
        return w * dw * 2 * np.pi / wavelength * np.cos(2 * np.pi / wavelength * radius)

    w_values = local_w(sample_radii.reshape(1, -1)) * np.ones((npx_front, 1))

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        w_values * sample_radii * 2 * np.pi / npx_front, sample_radii)

    penetration_increment = wavelength / 4
    max_penetration = 1.

    # %%

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(penetration_increment, max_penetration),
        filename="trust_lin_interp.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=wavelength / 4,
        dump_fields=False,
        gtol=1e-08
        )

    # %% Reference case that the linear interpolation works properly

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(
        npx_front,
        w=lambda radius, angle: local_w(radius),
        dw=lambda radius, angle: local_dw(radius),
        wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(penetration_increment, max_penetration),
        filename="trust_direct.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=wavelength / 4,
        dump_fields=False,
        gtol=1e-08
        )

    # %%

    if False:

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        a = np.linspace(0.001, 2, 300)
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=local_w(a)), JKR.force(contact_radius=a, work_of_adhesion=local_w(a)), "-k")
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=w), JKR.force(contact_radius=a, work_of_adhesion=w), "--k")

        nc_interp = NCStructuredGrid("trust_lin_interp.nc")
        nc_direct = NCStructuredGrid("trust_direct.nc")

        ax.plot(nc_interp.penetration, nc_interp.force, "+", label="lin. interp")
        ax.plot(nc_direct.penetration, nc_direct.force, "x", label="analytical")

        plt.show()


    # %% TODO, eventually
    # I can check bicubic vs. linear that almost all penetrations are nearly equal



def test_axissymmetric_sinewave_rosso_krauth():
    n_radii = 400
    npx_front = 64  # Warning ! This needs to be reasonably high for the RK solvers to work !

    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    sample_radii = np.linspace(0.5 * pulloff_radius, 6, n_radii)

    wavelength = 0.1

    dw = 0.2

    def local_w(radius):
        return w * (1 + dw * np.sin(2 * np.pi / wavelength * radius))

    def local_dw(radius):
        return w * dw * 2 * np.pi / wavelength * np.cos(2 * np.pi / wavelength * radius)

    w_values = local_w(sample_radii.reshape(1, -1)) * np.ones((npx_front, 1))

    piecewise_linear_w_radius = linear_interpolated_pinning_field_equaly_spaced(
        w_values * sample_radii * 2 * np.pi / npx_front, sample_radii)

    penetration_increment = wavelength / 4
    max_penetration = 1.

    # %% Reference case that the linear interpolation works properly

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(
        npx_front,
        w=lambda radius, angle: local_w(radius),
        dw=lambda radius, angle: local_dw(radius),
        wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(penetration_increment, max_penetration),
        filename="trust_direct.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=wavelength / 4,
        dump_fields=False,
        gtol=1e-08
        )

    # %%

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w_radius, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(penetration_increment, max_penetration),
        filename="trust_lin_interp.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=wavelength / 4,
        dump_fields=False,
        gtol=1e-08
        )

    # %% numpy Rosso Krauth propagation
    print("Numpy implementation of Rosso-Krauth")

    cf.propagate_rosso_krauth(
        penetrations=penetrations_generator(penetration_increment, max_penetration),
        gtol=1e-08,
        maxit=10000,
        file="RK_numpy.nc",
        logger=Logger("RK_numpy.log", outevery=100),
        )

    # %% Pytorch Rosso-Krauth implementation

    propagate_rosso_krauth(
        cf.piecewise_linear_w_radius,
        penetration_increment=penetration_increment,
        max_penetration=max_penetration,
        initial_a=np.ones(cf.npx) * cf.piecewise_linear_w_radius.kinks[0],
        gtol=1e-08,
        dump_fields=True,
        filename="RK_torch.nc"
    )


    # %%

    nc_interp = NCStructuredGrid("trust_lin_interp.nc")
    nc_direct = NCStructuredGrid("trust_direct.nc")
    nc_rk_torch = NCStructuredGrid("RK_torch.nc")
    nc_rk_numpy = NCStructuredGrid("RK_numpy.nc")

    # %%

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        a = np.linspace(0.001, 2, 300)
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=local_w(a)), JKR.force(contact_radius=a, work_of_adhesion=local_w(a)), "-k")
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=w), JKR.force(contact_radius=a, work_of_adhesion=w), "--k")


        ax.plot(nc_direct.penetration, nc_direct.force, "o", label="analytical")
        ax.plot(nc_interp.penetration, nc_interp.force, "+", label="lin. interp")
        ax.plot(nc_rk_numpy.penetration, nc_rk_numpy.force, "x", label="rk_numpy")
        ax.plot(nc_rk_torch.penetration, nc_rk_torch.force, "--", label="rk_torch")

        plt.show()

    # %%

    # same interpolation, rk solver versus trust region
    np.testing.assert_allclose(nc_rk_numpy.force, nc_interp.force, rtol=1e-05)
    np.testing.assert_allclose(nc_rk_torch.force, nc_interp.force, rtol=1e-05)
