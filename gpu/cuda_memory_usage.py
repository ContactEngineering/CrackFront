import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from NuMPI.IO.NetCDF import NCStructuredGrid

import sys, os
sys.path.insert(0, os.path.abspath("../")) # Use local code and not the one in the singularity image
from CrackFront.Optimization.propagate_elastic_line_pytorch import propagate_rosso_krauth

import CrackFront

CrackFront.__file__

16384 * 2 

# +
torch.cuda.reset_peak_memory_stats()

time_gpu = []
time_cpu = []
time_numpy = []

nit_gpu = []
nit_cpu = []
nit_numpy = []

max_memory = []

line_lengths = [256, 1024, 4096, 16384,]
# measure scaling of time
for line_length in line_lengths:
    print("Line length", line_length)
    params = dict(
    line_length=line_length,
    propagation_length=line_length,
    rms=.5,
    structural_length=128,
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

    start_time = time.time()
    propagate_rosso_krauth(
            **params,
            pinning_forces=pinning_forces,
            dump_fields=False,
            simulation_type="reciprocating parabolic potential",
            # logger=Logger("simulation.log", outevery=100),
            disable_cuda=False,
            filename="torch_gpu.nc"
            )
    time_gpu.append(time.time() - start_time)
    max_memory.append(torch.cuda.max_memory_allocated())
    nc = NCStructuredGrid("torch_gpu.nc")
    nit_gpu.append(np.sum(nc.nit[:]))

# +
fig, ax = plt.subplots()

ax.plot(line_lengths, time_gpu, "+")
ax.set_ylabel("time")
ax.set_xlabel("line length")
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("time.svg")

# +
fig, ax = plt.subplots()

ax.plot(line_lengths, np.array(max_memory) / 1e6, "+")
ax.set_ylabel("max memory allocated (MBytes)")
ax.set_xlabel("line length")
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("memory_usage.svg")
# -

# The GPU on the NVidia node has 32 GB https://wiki.bwhpc.de/e/BwForCluster_NEMO_Hardware_and_Architecture#Compute_and_Special_Purpose_Nodes
#
# On the vis node and the AMD node it has 16GB only
#

np.array(max_memory) / 1e9  # Memory in GB

# This means that if I use the GPU node I should be able to do a 32k simulation.
#
# If I only store 1 of values or slopes, 65k could be possible as well. 


