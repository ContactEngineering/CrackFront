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
# +
line_length = 8192
structural_length = line_length / 2

print("Line length", line_length)
params = dict(
line_length=line_length,
propagation_length=int(2 * structural_length),
rms=.5,
structural_length=structural_length,
n_steps=10,
# randomness:
seed=0,
# numerics:
gtol=1e-8,
maxit=10000000,
)
params.update(initial_a=- np.ones(params["line_length"]) * params["structural_length"] * params["rms"] ** 2 * 2)
np.random.seed(params["seed"])
pinning_forces = np.random.normal(size=(params["line_length"], params["propagation_length"])) * params["rms"]
# -

# %load_ext line_profiler

# %lprun -f propagate_rosso_krauth -T line_profile_propgate_rosso_krauth.txt propagate_rosso_krauth(**params,pinning_forces=pinning_forces,dump_fields=False,simulation_type="reciprocating parabolic potential",disable_cuda=False,filename="torch_gpu_profiling.nc")


# !cat line_profile_propgate_rosso_krauth.txt

# visualizing with snakewiz

# + active=""
# %load_ext snakeviz

# + active=""
# %%prun -D rosso_krauth
# propagate_rosso_krauth(
#         **params,
#         pinning_forces=pinning_forces,
#         dump_fields=False,
#         simulation_type="reciprocating parabolic potential",
#         # logger=Logger("simulation.log", outevery=100),
#         disable_cuda=False,
#         filename="torch_gpu_profiling.nc"
#         )
