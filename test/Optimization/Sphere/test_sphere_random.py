
import numpy as np
import pytest
from Adhesion.ReferenceSolutions import JKR
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography import Topography
from SurfaceTopography.Generation import fourier_synthesis

from CrackFront.Circular import Interpolator, pol2cart
from CrackFront.CircularEnergyReleaseRate import (
    SphereCrackFrontERRPenetrationEnergyConstGc,
    SphereCFPenetrationEnergyConstGcPiecewiseLinearField, SphereCrackFrontERRPenetrationEnergy,
    SphereCrackFrontERRPenetrationLin
    )
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_pytorch import propagate_rosso_krauth
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator, simulate_crack_front

from CrackFront.CircularEnergyReleaseRate import Es, w, R, maugis_K, generate_random_work_of_adhesion


def test_random_linear_interp():
    params = dict(
        # pixel_size_radial=0.1,
        n_pixels_front=512,
        rms=.5,
        max_penetration=1.,
        penetration_increment=0.05,
        shortcut_wavelength=0.08,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-8,
        maxit=10000000,
        n_pixels=256,
        # n_pixels_fourier_interpolation=128,
        pixel_size=0.02,
    )
    npx_front = params["n_pixels_front"]
    assert params["shortcut_wavelength"] > 2 * params["pixel_size"]
    params.update(dict(pixel_size_radial=params["shortcut_wavelength"] / 16))
    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    minimum_radius = pulloff_radius / 10

    # maximum radius
    physical_sizes = params["pixel_size"] * params["n_pixels"]

    maximum_radius = physical_sizes / 2

    n_pixels_radial = np.floor((maximum_radius - minimum_radius) / params["pixel_size_radial"])

    sample_radii = np.arange(n_pixels_radial) * params["pixel_size_radial"] + minimum_radius
    cf_angles = np.arange(params["n_pixels_front"]) * 2 * np.pi / npx_front

    w_topography = generate_random_work_of_adhesion(**params)

    interpolator = Interpolator(w_topography)

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        interpolator.field_polar(sample_radii.reshape(1, -1), cf_angles.reshape(-1, 1)) * sample_radii.reshape(1, -1) * 2 * np.pi / npx_front, sample_radii)

    # %%

    print("############# TRUST REGION on LINEAR interpolated field ################")
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        filename="trust_lin_interp.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=params["shortcut_wavelength"] / 16,
        dump_fields=False,
        gtol=params["gtol"]
        )

    # %% Reference case that the linear interpolation works properly
    print("############# TRUST REGION on SPLINE interpolated field ################")

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(
        npx_front,
        w=lambda radius, angle: interpolator.field_polar(radius, cf_angles),
        dw=lambda radius, angle: interpolator.dfield_dr_polar(radius, cf_angles),
        wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        filename="trust_direct.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=params["shortcut_wavelength"] / 16,
        dump_fields=False,
        gtol=params["gtol"]
        )

    # %%

    nc_interp = NCStructuredGrid("trust_lin_interp.nc")
    nc_direct = NCStructuredGrid("trust_direct.nc")

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        a = np.linspace(0.001, 2, 300)
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=w), JKR.force(contact_radius=a, work_of_adhesion=w), "--k")
        ax.plot(nc_interp.penetration, nc_interp.force, "+", label="lin. interp")
        ax.plot(nc_direct.penetration, nc_direct.force, "x", label="analytical")

        plt.show()

    np.testing.assert_allclose(nc_interp.penetration, nc_direct.penetration)

    # we need a certain tolerance because discrepancies have to be expected due to the different interpolation.
    np.testing.assert_allclose(nc_interp.force, nc_direct.force, rtol=1e-2)
    np.testing.assert_allclose(nc_interp.mean_radius, nc_direct.mean_radius, rtol=1e-2)


def test_random_rosso_krauth():
    params = dict(
        # pixel_size_radial=0.1,
        n_pixels_front=512,
        rms=.5,
        max_penetration=1.,
        penetration_increment=0.05,
        shortcut_wavelength=0.08,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-8,
        maxit=10000,
        n_pixels=256,
        #n_pixels_fourier_interpolation=128,
        pixel_size=0.02,
    )
    npx_front = params["n_pixels_front"]
    assert params["shortcut_wavelength"] > 2 * params["pixel_size"]
    params.update(dict(pixel_size_radial = params["shortcut_wavelength"] / 16) )
    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    minimum_radius = pulloff_radius / 10

    # maximum radius
    physical_sizes = params["pixel_size"] * params["n_pixels"]

    maximum_radius = physical_sizes / 2

    n_pixels_radial = np.floor((maximum_radius - minimum_radius) / params["pixel_size_radial"])

    sample_radii = np.arange(n_pixels_radial) * params["pixel_size_radial"] + minimum_radius
    cf_angles = np.arange(params["n_pixels_front"]) * 2 * np.pi / npx_front

    w_topography = generate_random_work_of_adhesion(**params)

    interpolator = Interpolator(w_topography)

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        interpolator.field_polar(sample_radii.reshape(1, -1), cf_angles.reshape(-1, 1)) * sample_radii.reshape(1, -1) * 2 * np.pi / npx_front, sample_radii)

    # %% Reference case that the linear interpolation works properly
    print("############# TRUST REGION on SPLINE interpolated field ################")

    cf = SphereCrackFrontERRPenetrationEnergyConstGc(
        npx_front,
        w=lambda radius, angle: interpolator.field_polar(radius, cf_angles),
        dw=lambda radius, angle: interpolator.dfield_dr_polar(radius, cf_angles),
        wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        filename="trust_spline.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=params["shortcut_wavelength"] / 16,
        dump_fields=False,
        gtol=params["gtol"]
        )

    # %%

    print("############# TRUST REGION on LINEAR interpolated field ################")
    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    simulate_crack_front(
        cf,
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        filename="trust_lin_interp.nc",
        pulloff_radius=pulloff_radius * 0.5,
        initial_radius=JKR.contact_radius(penetration=0, work_of_adhesion=w) * 0.5,
        trust_radius=params["shortcut_wavelength"] / 16,
        dump_fields=False,
        gtol=params["gtol"]
        )

    # %%
    print("Numpy implementation of Rosso-Krauth")

    cf.propagate_rosso_krauth(
        penetrations=penetrations_generator(params["penetration_increment"], params["max_penetration"]),
        gtol=params["gtol"],
        maxit=params["maxit"],
        file="RK_numpy.nc",
        logger=Logger("RK_numpy.log", outevery=1),
        )


    # %% Pytorch Rosso-Krauth implementation
    print("Pytorch implementation of Rosso-Krauth")
    propagate_rosso_krauth(
        cf,
        penetration_increment=params["penetration_increment"],
        max_penetration=params["max_penetration"],
        initial_a=np.ones(cf.npx) * (cf.piecewise_linear_w_radius.kinks[0]),
        gtol=params["gtol"],
        dump_fields=False,
        maxit=params["maxit"],
        filename="RK_torch.nc",
        logger=Logger("RK_pytorch.log", outevery=1),
    )

    # %% Assert numpy and RK implementation have exactly the same convergence behavior ()

    conv_data_RK = np.loadtxt("RK_pytorch.log")
    conv_data_np = np.loadtxt("RK_numpy.log")


    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        imax = np.argwhere(conv_data_RK[:, 0] == 1)[1] - 1
        sl = slice(0, int(imax))
        ax.plot(conv_data_RK[sl, 0], conv_data_RK[sl, 1], label="torch")
        ax.plot(conv_data_np[sl, 0], conv_data_np[sl, 1], label="numpy")

        ax.legend()

        ax.set_yscale("log")
        ax.set_yscale("max(|grad|)")

    # first penetration
    sl = slice(0, int(np.argwhere(conv_data_RK[:, 0] == 1)[1] - 1))
    np.testing.assert_allclose(conv_data_RK[sl, 1], conv_data_np[sl, 1])
    # second penetration
    # It already runs away a little bit.
    #sl = slice(int(np.argwhere(conv_data_RK[:, 0] == 1)[1]), int(np.argwhere(conv_data_RK[:, 0] == 1)[2] - 1))
    # np.testing.assert_allclose(conv_data_RK[sl, 1], conv_data_np[sl, 1])
    # At some point they will run away from each other because of numerical errors.

    # %% Check the results against the trust region implementation

    nc_interp = NCStructuredGrid("trust_lin_interp.nc")
    nc_spline = NCStructuredGrid("trust_spline.nc")
    nc_rk_torch = NCStructuredGrid("RK_torch.nc")
    nc_rk_numpy = NCStructuredGrid("RK_numpy.nc")

    # %% Assert the results are the same for RK and the Trust-Region

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        a = np.linspace(0.001, 2, 300)
        ax.plot(JKR.penetration(contact_radius=a, work_of_adhesion=w), JKR.force(contact_radius=a, work_of_adhesion=w), "--k")

        ax.plot(nc_spline.penetration, nc_spline.force, "o", label="spline")
        ax.plot(nc_interp.penetration, nc_interp.force, "+", label="lin. interp")
        ax.plot(nc_rk_numpy.penetration, nc_rk_numpy.force, "x", label="rk_numpy")
        ax.plot(nc_rk_torch.penetration, nc_rk_torch.force, "--", label="rk_torch")

        plt.show()

    np.testing.assert_allclose(nc_rk_numpy.penetration, nc_interp.penetration, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_numpy.force, nc_interp.force, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_numpy.mean_radius, nc_interp.mean_radius, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_numpy.min_radius, nc_interp.min_radius, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_numpy.max_radius, nc_interp.max_radius, rtol=1e-5)

    np.testing.assert_allclose(nc_rk_torch.penetration, nc_interp.penetration, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_torch.force, nc_interp.force, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_torch.mean_radius, nc_interp.mean_radius, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_torch.min_radius, nc_interp.min_radius, rtol=1e-5)
    np.testing.assert_allclose(nc_rk_torch.max_radius, nc_interp.max_radius, rtol=1e-5)

    # %% compare global number of iterations of the two Rosso Krauth implementations
    assert np.count_nonzero(nc_rk_torch.nit != nc_rk_numpy.nit) < 0.05 * len(nc_rk_torch)
