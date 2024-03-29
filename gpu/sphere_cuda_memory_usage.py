# +
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath("../")) # Use local code and not the one in the singularity image
from CrackFront.Optimization.propagate_elastic_line_pytorch import propagate_rosso_krauth

import CrackFront
CrackFront.__file__

# +

# nondimensional units following Maugis Book:
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography.Generation import fourier_synthesis

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator
from CrackFront.Optimization.propagate_sphere_pytorch import propagate_rosso_krauth


Es = 3 / 4
w = 1 / np.pi
R = 1.
maugis_K = 1.
mean_Kc = np.sqrt(2 * Es * w)

def generate_random_field(
    pixel_size,
    n_pixels,
    shortcut_wavelength,
    seed,
    rms,
    n_pixels_fourier_interpolation=None,
    **kwargs):
    if n_pixels_fourier_interpolation is None:
        n_pixels_fourier_interpolation = n_pixels

    np.random.seed(seed)

    w_landscape = fourier_synthesis(
        (n_pixels, n_pixels),
        [n_pixels * pixel_size] * 2,
        long_cutoff=shortcut_wavelength,
        hurst=.5,  # doesn't matter
        short_cutoff=shortcut_wavelength,
        c0=1.
        ).interpolate_fourier((n_pixels_fourier_interpolation, n_pixels_fourier_interpolation))

    w_landscape = w_landscape.scale(w *  rms / w_landscape.rms_height_from_area()).squeeze()
    w_landscape._heights += w
    return w_landscape


# +
time_gpu = []
time_cpu = []
time_numpy = []

nit_gpu = []
nit_cpu = []
nit_numpy = []

line_lengths = []

max_memory = []

for refine  in [1, 2, 4, 8, 16]:
    
    params = dict(
            # pixel_size_radial=0.1,
            n_pixels_front=512 * refine,
            rms=.1,
            max_penetration=0.8,
            penetration_increment=0.4,
            shortcut_wavelength=0.08 / refine,
            # randomness:
            seed=0,
            # numerics:
            gtol=1e-8,
            maxit=10000000,
            n_pixels=256 * refine,
            # n_pixels_fourier_interpolation=128,
            pixel_size=0.02 / refine,
        )
    npx_front = params["n_pixels_front"]
    print("############ npx_front: ", npx_front)
    assert params["shortcut_wavelength"] > 2 * params["pixel_size"]
    params.update(dict(pixel_size_radial=params["shortcut_wavelength"] / 16))
    pulloff_radius = (np.pi * w * R ** 2 / 6 * maugis_K) ** (1 / 3)

    minimum_radius = pulloff_radius / 10

    line_lengths.append(npx_front)

    # maximum radius
    physical_sizes = params["pixel_size"] * params["n_pixels"]

    maximum_radius = physical_sizes / 2

    n_pixels_radial = np.floor((maximum_radius - minimum_radius) / params["pixel_size_radial"])

    sample_radii = np.arange(n_pixels_radial) * params["pixel_size_radial"] + minimum_radius
    cf_angles = np.arange(params["n_pixels_front"]) * 2 * np.pi / npx_front

    w_topography = generate_random_field(**params)

    interpolator = Interpolator(w_topography)

    piecewise_linear_w = linear_interpolated_pinning_field_equaly_spaced(
        interpolator.field_polar(sample_radii.reshape(1, -1), cf_angles.reshape(-1, 1)) * sample_radii.reshape(1, -1) * 2 * np.pi / npx_front, sample_radii)

    cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=w)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    propagate_rosso_krauth(
        cf,
        penetration_increment=params["penetration_increment"],
        max_penetration=params["max_penetration"],
        initial_a=np.ones(cf.npx) * (cf.piecewise_linear_w_radius.kinks[0]),
        gtol=params["gtol"],
        dump_fields=False,
        maxit=params["maxit"],
        filename="torch_gpu.nc",
        disable_cuda=False
    )
    time_gpu.append(time.time() - start_time)
    max_memory.append(torch.cuda.max_memory_allocated())

    nc = NCStructuredGrid("torch_gpu.nc")
    nit_gpu.append(np.sum(nc.nit[:]))

# +
fig, ax = plt.subplots()

ax.plot(line_lengths, time_gpu, "+")
ax.plot(line_lengths, np.array(line_lengths) /100, "k")

ax.set_ylabel("time")
ax.set_xlabel("line length")
ax.set_xscale("log")
ax.set_yscale("log")


# +
fig, ax = plt.subplots()

ax.plot(line_lengths, np.array(max_memory) / 1e6, "+")
ax.set_ylabel("max memory allocated (MBytes)")
ax.set_xlabel("line length")
ax.set_xscale("log")
ax.set_yscale("log")
# -

np.array(max_memory) / 1e9  # Memory in GB

line_lengths


