import time
from CrackFront.Optimization.propagate_elastic_line import propagate_rosso_krauth
from CrackFront.Optimization.propagate_elastic_line_pytorch import propagate_rosso_krauth as propagate_rosso_krauth_torch

import numpy as np
from NuMPI.IO.NetCDF import NCStructuredGrid


def time_vs_line_length_scaling():

    time_gpu = []
    time_cpu = []
    time_numpy = []

    nit_gpu = []
    nit_cpu = []
    nit_numpy = []

    line_lengths = [256, 1024, 4096, 16384, 65536]
    # measure scaling of time
    for line_length in line_lengths:
        print("Line length", line_length)
        params = dict(
        line_length=line_length,
        propagation_length=256,
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
            #logger=Logger("simulation.log", outevery=100),
            filename="numpy.nc"
            )
        time_numpy.append(time.time() - start_time)

        start_time = time.time()
        propagate_rosso_krauth_torch(
            **params,
            pinning_forces=pinning_forces,
            dump_fields=False,
            simulation_type="reciprocating parabolic potential",
            # logger=Logger("simulation.log", outevery=100),
            disable_cuda=True,
            filename="torch_cpu.nc"
            )
        time_cpu.append(time.time() - start_time)

        start_time = time.time()
        propagate_rosso_krauth_torch(
            **params,
            pinning_forces=pinning_forces,
            dump_fields=False,
            simulation_type="reciprocating parabolic potential",
            # logger=Logger("simulation.log", outevery=100),
            disable_cuda=False,
            filename="torch_gpu.nc"
            )
        time_gpu.append(time.time() - start_time)

        nc = NCStructuredGrid("torch_gpu.nc")
        nit_gpu.append(np.sum(nc.nit[:]))
        nc = NCStructuredGrid("torch_cpu.nc")
        nit_cpu.append(np.sum(nc.nit[:]))
        nc = NCStructuredGrid("numpy.nc")
        nit_numpy.append(np.sum(nc.nit[:]))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(line_lengths, time_cpu, "+", label="cpu")
    ax.plot(line_lengths, time_gpu, "x", label="gpu")
    ax.plot(line_lengths, time_numpy, "o", label="numpy")

    ax.legend()

    ax.set_ylabel("total time")
    ax.set_xlabel("line length")

    ax.set_xscale("log")
    ax.set_yscale("log")  
    
    fig, ax = plt.subplots()

    ax.plot(line_lengths, np.array(time_cpu) / np.array(nit_cpu), "+", label="cpu")
    ax.plot(line_lengths, np.array(time_gpu) / np.array(nit_gpu), "x", label="gpu")
    ax.plot(line_lengths, np.array(time_numpy) / np.array(nit_numpy), "o", label="numpy")

    ax.legend()
    ax.set_ylabel("time per iteration")
    ax.set_xlabel("line length")
    ax.set_xscale("log")
    ax.set_yscale("log")    

if __name__ == "__main__":
    time_vs_line_length_scaling()








