import numpy as np
import torch
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid
from matplotlib import pyplot as plt

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import (
    generate_random_work_of_adhesion,
    SphereCFPenetrationEnergyConstGcPiecewiseLinearField, w, maugis_K, R
    )
from CrackFront.Optimization.propagate_sphere_pytorch import (
    LinearInterpolatedPinningFieldUniformFromFile,
    propagate_rosso_krauth
    )

disable_cuda = False

# TORCH code starts here
if torch.cuda.is_available() and not disable_cuda:
    accelerator = torch.device("cuda")
    print("CUDA detected, using CUDA")
else:
    if not disable_cuda:
        print("CUDA not available, fall back to torch on CPU")
    else:
        print("CUDA disabled, use CPU")
    accelerator = torch.device("cpu")


def test_pinning_field_from_file():
    npx_propagation = 25
    npx_front = 19

    values = np.random.normal(size=(npx_propagation, npx_front))

    grid_spacing = 4
    minimum_radius = 3

    # SAVE
    LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        values,
        grid_spacing=grid_spacing,
        filename="values_and_slopes.npy"
        )

    # Instantiate the file handler
    pf = LinearInterpolatedPinningFieldUniformFromFile(
        filename="values_and_slopes.npy",
        grid_spacing=grid_spacing,
        min_radius=minimum_radius,
        accelerator=accelerator
        )

    subdomain = [5, 21]
    collocation_points = torch.from_numpy(np.random.randint(*subdomain, size=npx_front))

    # Load all data
    pf.load_data(0, npx_propagation)

    values_and_slopes_from_full = pf.values_and_slopes(collocation_points).to(device="cpu").numpy()

    # Load part of the data
    pf.load_data(*subdomain)

    values_and_slopes_from_part = pf.values_and_slopes(collocation_points).to(device="cpu").numpy()

    # should yield the same values at the collocation point
    np.testing.assert_equal(values_and_slopes_from_part, values_and_slopes_from_full)

    # check the slopes are computed ok
    pf.load_data(0, npx_propagation)
    values_and_slopes_upper = pf.values_and_slopes(collocation_points + 1).to(device="cpu").numpy()
    np.testing.assert_equal(
        values_and_slopes_from_part[:, 1],
        (values_and_slopes_upper[:, 0] - values_and_slopes_from_part[:, 0]) / grid_spacing
        )

def test_propagate_rosso_krauth_with_partial_data():
    params = dict(
        # pixel_size_radial=0.1,
        n_pixels_front=512,
        rms=.5,
        max_penetration=1.,
        penetration_increment=0.2,
        shortcut_wavelength=0.08,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-8,
        maxit=10000,
        n_pixels=256,
        # n_pixels_fourier_interpolation=128,
        pixel_size=0.02,
        )
    npx_front = params["n_pixels_front"]
    assert params["shortcut_wavelength"] > 2 * params["pixel_size"]
    params.update(dict(pixel_size_radial=params["shortcut_wavelength"] / 16))
    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    minimum_radius = pulloff_radius / 3

    # maximum radius
    physical_sizes = params["pixel_size"] * params["n_pixels"]

    maximum_radius = physical_sizes / 2

    n_pixels_radial = np.floor((maximum_radius - minimum_radius) / params["pixel_size_radial"])

    sample_radii = np.arange(n_pixels_radial) * params["pixel_size_radial"] + minimum_radius
    cf_angles = np.arange(params["n_pixels_front"]) * 2 * np.pi / npx_front

    w_topography = generate_random_work_of_adhesion(**params)

    interpolator = Interpolator(w_topography)
    w_radius_values = np.ascontiguousarray(interpolator.field_polar(sample_radii.reshape(-1, 1), cf_angles.reshape(1, -1)) \
                      * sample_radii.reshape(-1, 1) * 2 * np.pi / npx_front)

    LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        w_radius_values,
        params["pixel_size_radial"],
        filename="values_and_slopes.npy")

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
        piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename="values_and_slopes.npy",
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=torch.device("cpu")
            ),
        wm=w)

    propagate_rosso_krauth(
        cf,
        initial_a=np.ones(cf.npx) * JKR.contact_radius(penetration=0) * 0.5,
        dump_fields=False,
        filename="all_loaded.nc",
        **params,
    )

    propagate_rosso_krauth(
        cf,
        initial_a=np.ones(cf.npx) * cf.piecewise_linear_w_radius.min_radius,
        dump_fields=False,
        filename="partial_loaded.nc",
        pinning_field_memory=int(cf.piecewise_linear_w_radius.npx_propagation * 0.25),
        **params,
    )

    # assert we have the same result
    nc_full = NCStructuredGrid("all_loaded.nc")
    nc_partial = NCStructuredGrid("partial_loaded.nc")

    np.testing.assert_allclose(nc_full.force, nc_partial.force, rtol=1e-09)
    np.testing.assert_allclose(nc_full.penetration, nc_partial.penetration, rtol=1e-09)

