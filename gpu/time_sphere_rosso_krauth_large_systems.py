import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from NuMPI.IO.NetCDF import NCStructuredGrid

# +
import sys, os
sys.path.insert(0, os.path.abspath("../")) # Use local code and not the one in the singularity image
from CrackFront.Optimization.propagate_elastic_line_pytorch import propagate_rosso_krauth

import CrackFront
CrackFront.__file__

# -


# Configure path where to cache

try:
    # We are in a job, where TMPDIR is defined and points towards the local memory of the node
    tmpdir = os.environ["TMPDIR"] 
    tmpdir = "/tmp" # the actual TMPDIR was bindmouted as /tmp by singularity
except KeyError:
    # We are on the vis node, and there isn't such a thing
    tmpdir = os.environ["PWD"]

cache_filename = tmpdir + "/values_and_slopes.npy"
cache_filename

# +
# nondimensional units following Maugis Book:
from NuMPI.IO.NetCDF import NCStructuredGrid
from SurfaceTopography.Generation import fourier_synthesis

from CrackFront.Circular import Interpolator
from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field_equaly_spaced
from CrackFront.Optimization.propagate_sphere_trust_region import penetrations_generator
from CrackFront.Optimization.propagate_sphere_pytorch import propagate_rosso_krauth, LinearInterpolatedPinningFieldUniformFromFile


Es = 3 / 4
w = 1 / np.pi
R = 1.
maugis_K = 1.
mean_Kc = np.sqrt(2 * Es * w)


# +

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



# -

refine = 16

# +
if torch.cuda.is_available():
    print("CUDA detected, using CUDA")
    accelerator = torch.device("cuda")
else:
    print("CUDA not available or disabled, fall back to torch on CPU")
    accelerator = torch.device("cpu")
    
cpu = torch.device("cpu")

# +
params = dict(
            # pixel_size_radial=0.1,
            n_pixels_front=512 * refine,
            rms=.4,
            max_penetration=1.,
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
            dump_fields=False,
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

w_topography = generate_random_field(**params)

interpolator = Interpolator(w_topography)

w_radius_values = interpolator.field_polar(sample_radii.reshape(-1, 1), cf_angles.reshape(1, -1)) \
                       * sample_radii.reshape(-1, 1) * 2 * np.pi / npx_front

LinearInterpolatedPinningFieldUniformFromFile.save_values_and_slopes_to_file(
        values=w_radius_values,
        grid_spacing=params["pixel_size_radial"],
        filename=cache_filename,
)

initial_a = np.ones(params["n_pixels_front"]) * sample_radii[0]

# -

npx_front

w_topography.rms_height_from_area()

# %load_ext line_profiler

# ### All GPU, but with only part of the pinning field loaded

cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
    piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename=cache_filename,
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=accelerator, 
            data_device=accelerator,
            ),
    wm=w)
propagate_rosso_krauth(cf, initial_a=initial_a.copy(),filename="torch_timing.nc", pinning_field_memory=int(cf.piecewise_linear_w_radius.npx_propagation * .25), **params,)

# +
cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
    piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename=cache_filename,
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=accelerator, 
            data_device=accelerator,
            ),
    wm=w)

pinning_field_memory = int(cf.piecewise_linear_w_radius.npx_propagation * 0.25)
torch.cuda.reset_peak_memory_stats()
# %lprun -f propagate_rosso_krauth -T line_profile_sphere_rosso_krauth.txt propagate_rosso_krauth(cf, initial_a=initial_a.copy(),filename="torch_timing.nc", pinning_field_memory=pinning_field_memory, **params,)
print("max_memory_allocated:", torch.cuda.max_memory_allocated())
# -

# Full

# !cat line_profile_sphere_rosso_krauth.txt

# grep filtered showing only lines with more then 10%

# !grep -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9][0-9].[0-9]+[[:space:]]" line_profile_sphere_rosso_krauth.txt

# grep filtered output that shows only lines that have more then 0% time

# +
# #!grep -v -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+0.0[[:space:]]" line_profile_sphere_rosso_krauth.txt
# -

# ### All GPU

# +
cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
    piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename=cache_filename,
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=accelerator, 
            data_device=accelerator,
            ),
    wm=w)

# I run it once "for nothing" to awaken the GPU. Otherwise the first GPU operation might seem extremely slow. 
propagate_rosso_krauth(cf, initial_a=initial_a.copy(),filename="torch_timing.nc",**params,)

torch.cuda.reset_peak_memory_stats()
# %lprun -f propagate_rosso_krauth -T line_profile_sphere_rosso_krauth.txt propagate_rosso_krauth(cf, initial_a=initial_a.copy(),filename="torch_timing.nc",**params,)
print("max_memory_allocated:", torch.cuda.max_memory_allocated())
# -

# Full

# !cat line_profile_sphere_rosso_krauth.txt

# grep filtered showing only lines with more then 10%

# !grep -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9][0-9].[0-9]+[[:space:]]" line_profile_sphere_rosso_krauth.txt

# grep filtered output that shows only lines that have more then 0% time

# +
# #!grep -v -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+0.0[[:space:]]" line_profile_sphere_rosso_krauth.txt
# -

# ### GPU, but pinning field array on CPU

# +
cf = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(
    piecewise_linear_w_radius=LinearInterpolatedPinningFieldUniformFromFile(
            filename=cache_filename,
            min_radius=sample_radii[0],
            grid_spacing=params["pixel_size_radial"],
            accelerator=accelerator, 
            data_device=cpu,
            ),      
    wm=w)

torch.cuda.reset_peak_memory_stats()
# %lprun -f propagate_rosso_krauth -T line_profile_sphere_rosso_krauth.txt propagate_rosso_krauth(cf, initial_a=initial_a.copy(),filename="torch_timing.nc",**params,)
print("max_memory_allocated:", torch.cuda.max_memory_allocated())
# -

# Full

# !cat line_profile_sphere_rosso_krauth.txt

# grep filtered output that shows only lines that have more then 10% time

# !grep -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9][0-9].[0-9]+[[:space:]]" line_profile_sphere_rosso_krauth.txt

# grep filtered output that shows only lines that have more then 0% time

# +
# #!grep -v -E "^[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+0.0[[:space:]]" line_profile_sphere_rosso_krauth.txt
# -

# ## How does the simulation look like ? 
#
# this is meant to be a "real life" situation

nc = NCStructuredGrid("torch_timing.nc")
fig, ax = plt.subplots()
ax.plot(nc.penetration, nc.force)
ax.set_xlabel(r"Penetration $\Delta ^* $")
ax.set_ylabel(r"Normal Force $F^* $")


# ## Exectuting on a job and exporting as html: 

# +
# %%writefile time_sphere_rosso_krauth_large_systems_job.sh
#MSUB -l walltime=01:00:00
#MSUB -m ea                                                                    
#MSUB -q gpu
#MSUB -l nodes=1:ppn=1:gpus=1 
#MSUB -l pmem=32G

set -e

ml devel/cuda/11.3 
ml tools/singularity/3.5

PATH=$HOME/commandline:$PATH

WS=/work/ws/nemo/fr_as1412-2110_cf_gpu-0
IMAGE=$WS/gpu_jupyterlab.sif

# cd $MOAB_SUBMITDIR

# we expect that this script was copied in dataset/data and submitted from there

export KMP_AFFINITY=compact,1,0
export OMP_NUM_THREADS=1
# Add local, up to date CrackFront installation on top of path.
export PYTHONPATH=$WS/CrackFront:$PYTHONPATH

FILE=$WS/CrackFront/gpu/time_sphere_rosso_krauth_large_systems.py

# --nv: for GPU
# --home=PWD: for jupyter

singularity exec --nv --home=$WS --pwd $PWD -B $WS -B $TMPDIR:/tmp $IMAGE sh $WS/commandline/jupytext_to_html $FILE
