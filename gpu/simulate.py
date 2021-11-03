import time

import numpy as np
from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
import sys



from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field, brute_rosso_krauth_other_spacing
import torch


def propagate_rosso_krauth(line_length, propagation_length, structural_length,
                           gtol,
                           n_steps,
                           initial_a,
                           pinning_forces,
                           dump_fields=True,
                           simulation_type="pulling parabolic potential",
                           maxit=10000,
                           compute_smallest_eigenvalue=False,
                           rosso_krauth=brute_rosso_krauth_other_spacing,
                           logger=None,
                           filename="data.nc",
                           **kwargs):
    """

    This algorithm has issue when the solution is not ahead of the initial configuration.
    Mind that when you choose the initial configuration

    """
    L = line_length

    # axev.axhline(2 * np.pi / structural_length)  # no disorder limit of the smallest eigenvalue
    interpolator = linear_interpolated_pinning_field(pinning_forces)
    line = ElasticLine(L, structural_length, pinning_field=interpolator)

    if simulation_type == "pulling parabolic potential":
        driving_positions = np.arange(n_steps) * propagation_length / n_steps
    elif simulation_type == "reciprocating parabolic potential":
        driving_positions = np.arange(n_steps) * propagation_length / n_steps
        driving_positions = np.concatenate((driving_positions, driving_positions[:-1][::-1]))
    else:
        raise ValueError

    a = initial_a

    nc = NCStructuredGrid(filename, "w", (L,))
    driving_a_prev = driving_positions[0] - 1
    for i in range(len(driving_positions)):

        driving_a = driving_positions[i]

        direction = np.sign(driving_a - driving_a_prev)
        print(f"position: {driving_a}")
        sol = rosso_krauth(a, driving_a, line, direction=direction, maxit=maxit, gtol=gtol, logger=logger)
        assert sol.success
        a = sol.x
        assert sol.success, f"solution failed: {sol.message}"

        line.dump(nc[i], driving_a, a, dump_fields=dump_fields)
        if compute_smallest_eigenvalue:
            eigval, eigvec = line.eigenvalues(a)
            nc[i].eigenvalue = eigval
            if dump_fields:
                nc[i].eigenvector = eigvec

        nc[i].nit = sol.nit
        print("nit {}".format(sol.nit))
        driving_a_prev = driving_a
        nc.sync()
        sys.stdout.flush()

    nc.close()
    return True


def propagate_rosso_krauth_torch(line_length, propagation_length, structural_length,
                                 gtol,
                                 n_steps,
                                 pinning_forces,
                                 initial_a,
                                 dump_fields=True,
                                 simulation_type="pulling parabolic potential",
                                 maxit=10000,
                                 compute_smallest_eigenvalue=False,
                                 rosso_krauth=brute_rosso_krauth_other_spacing,
                                 logger=None,
                                 disable_cuda=False,
                                 filename="data.nc",
                                 **kwargs):
    L = line_length

    # axev.axhline(2 * np.pi / structural_length)  # no disorder limit of the smallest eigenvalue
    interpolator = linear_interpolated_pinning_field(pinning_forces)
    line = ElasticLine(L, structural_length, pinning_field=interpolator)

    if simulation_type == "pulling parabolic potential":
        driving_positions = np.arange(n_steps) * propagation_length / n_steps
    elif simulation_type == "reciprocating parabolic potential":
        driving_positions = np.arange(n_steps) * propagation_length / n_steps
        driving_positions = np.concatenate((driving_positions, driving_positions[:-1][::-1]))
    else:
        raise ValueError

    grid_spacing = 1.
    npx_front = line_length
    npx_propagation = propagation_length
    values = pinning_forces
    slopes = (np.roll(pinning_forces, -1, axis=-1) - pinning_forces) / grid_spacing

    # TORCH code starts here
    if torch.cuda.is_available() and not disable_cuda:
        accelerator = torch.device("cuda")
    else:
        print("CUDA not available, fall back to torch on CPU")
        accelerator = torch.device("cpu")

    kwargs_array_creation = dict(device=accelerator)

    indexes = torch.arange(L, dtype=int)
    grid_spacing = torch.tensor(grid_spacing, dtype=torch.double)
    values_and_slopes = torch.from_numpy(np.stack([values, slopes], axis=2)).to(device=accelerator)

    q_front_rfft = 2 * np.pi * torch.fft.rfftfreq(npx_front, L / npx_front, **kwargs_array_creation)

    qk = 2 * np.pi / structural_length
    a_test = torch.zeros(npx_front, **kwargs_array_creation)
    a_test[0] = 1
    elastic_stiffness_individual = torch.fft.irfft(q_front_rfft * torch.fft.rfft(a_test), n=npx_front)[0] + qk

    colloc_point_above = np.zeros_like(initial_a, dtype=int)
    colloc_point_above = np.ceil(initial_a / line.pinning_field.grid_spacing, casting="unsafe", out=colloc_point_above)
    colloc_point_above += colloc_point_above == initial_a
    colloc_point_above = torch.from_numpy(colloc_point_above).to(**kwargs_array_creation)

    a = torch.from_numpy(initial_a).to(**kwargs_array_creation)

    nc = NCStructuredGrid(filename, "w", (L,))
    driving_a_prev = driving_positions[0] - 1
    for i in range(len(driving_positions)):

        driving_a_cpu = driving_positions[i]
        print(driving_a_cpu)
        driving_a = torch.tensor(driving_a_cpu).to(device=accelerator)
        # whether the crack is expected to move forward or backward
        direction = torch.sign(driving_a - driving_a_prev)
        # print(direction)
        nit = 0
        while nit < maxit:
            ###
            # Nullify the force on each pixel, assuming each pixel moves individually

            current_value_and_slope = values_and_slopes[indexes, (colloc_point_above - 1) % npx_propagation, :]

            # Here I splitted the evalation of grad in sevaral lines using add_ just in order to time these expressions
            # individually.
            grad = torch.fft.irfft(q_front_rfft * torch.fft.rfft(a), n=npx_front)
            grad.add_(qk * (a - driving_a))
            grad.add_(current_value_and_slope[:, 0])
            grad.add_(current_value_and_slope[:, 1] * (a - grid_spacing * (colloc_point_above - 1)))

            if logger:
                logger.st(["it", "max. residual"], [nit, torch.max(abs(grad))])

            # TODO: Optimization: I don't to evaluate this every iteration
            if (torch.max(torch.abs(grad)) < gtol):
                break

            stiffness = current_value_and_slope[:, 1] + elastic_stiffness_individual
            increment = - grad / stiffness
            ###

            a_new = a + increment
            mask_negative_stiffness = stiffness <= 0

            if direction == 1:
                # We let the line advance only until the boundary to the next pixel.
                # This is because the step length was based on the pinning curvature
                # which is erroneous as soon as we meet the next pixel
                #
                # Additionally, when the curvature is negative, the increment is negative but the front should actually move forward.
                # In this case as well we advance the front until the edge of the next pixel
                mask_new_pixel = torch.logical_or(a_new >= colloc_point_above * grid_spacing, mask_negative_stiffness)
                a_new = torch.where(mask_new_pixel, grid_spacing * colloc_point_above, a_new)

                colloc_point_above.add_(mask_new_pixel)

                # because of numerical errors it can be that the gradient points in the wrong
                # direction on some pixels, but is very small.
                # We just make sure these points do not move backwards
                a_new = torch.maximum(a_new, a)

            elif direction == -1:
                mask_new_pixel = torch.logical_or(a_new <= grid_spacing * (colloc_point_above - 1), mask_negative_stiffness)
                a_new = torch.where(mask_new_pixel, grid_spacing * (colloc_point_above - 1 ), a_new)

                colloc_point_above.add_(mask_new_pixel, alpha=-1) # alpha is a scalar prefactor for mask_new_pixel
                # Why not just -= ?

                a_new = torch.minimum(a_new, a)

            a = a_new

            nit += 1
        assert nit < maxit

        line.dump(nc[i], driving_a_cpu, a.to(device=torch.device("cpu")).numpy(), dump_fields=dump_fields)
        if compute_smallest_eigenvalue:
            eigval, eigvec = line.eigenvalues(a)
            nc[i].eigenvalue = eigval
            if dump_fields:
                nc[i].eigenvector = eigvec

        nc[i].nit = nit
        print("nit: {}".format(nit))

        driving_a_prev = driving_a
        nc.sync()
        sys.stdout.flush()

    nc.close()
    return True


def check_same_result():
    params = dict(
        line_length=256,  # starting from 8000 pix numpy starts to slower then cuda
        propagation_length=256,
        rms=.1,
        structural_length=64,
        n_steps=10,
        # randomness:
        seed=0,
        # numerics:
        gtol=1e-10,
        maxit=10000000,
        )
    params.update(initial_a=- np.ones(params["line_length"]) * params["structural_length"] * params["rms"] ** 2 * 2)
    np.random.seed(params["seed"])
    pinning_forces = np.random.normal(size=(params["line_length"], params["propagation_length"])) * params["rms"]

    propagate_rosso_krauth(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=True,
        simulation_type="reciprocating parabolic potential",
        #logger=Logger("simulation.log", outevery=100),
        filename="numpy.nc"
        )

    propagate_rosso_krauth_torch(
        **params,
        pinning_forces=pinning_forces,
        dump_fields=True,
        simulation_type="reciprocating parabolic potential",
        #logger=Logger("simulation.log", outevery=100),
        filename="torch.nc"
        )

    nc_torch = NCStructuredGrid("torch.nc")
    nc_numpy = NCStructuredGrid("numpy.nc")

    # fig, ax = plt.subplots()
    #
    # ax.plot(nc_torch.driving_position, nc_torch.driving_force, "+")
    # ax.plot(nc_numpy.driving_position, nc_numpy.driving_force, "x")
    #
    # fig, ax = plt.subplots()
    # for i in range(len(nc_torch)):
    #     ax.plot(nc_torch.position[i] - nc_numpy.position[i], "-")

    np.testing.assert_allclose(nc_torch.position_mean, nc_numpy.position_mean)
    np.testing.assert_allclose(nc_torch.position, nc_numpy.position, atol=1e-5)
    # 1e-5 is already very small compared to the heterogeneity spacing
    np.testing.assert_allclose(nc_torch.position_rms, nc_numpy.position_rms, )
    np.testing.assert_allclose(nc_torch.driving_force, nc_numpy.driving_force, )
    np.testing.assert_allclose(nc_torch.elastic_potential, nc_numpy.elastic_potential, )
    np.testing.assert_allclose(nc_torch.nit, nc_numpy.nit,)

def time_vs_line_length_scaling():

    time_gpu = []
    time_cpu = []
    time_numpy = []

    nit_gpu = []
    nit_cpu = []
    nit_numpy = []

    line_lengths = [256, 1024, 4096, 16384]
    # measure scaling of time
    for line_length in line_lengths:
        params = dict(
        line_length=line_length,  # starting from 8000 pix numpy starts to slower then cuda
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

    fig, ax = plt.subplots()

    ax.plot(line_lengths, np.array(time_cpu) / np.array(nit_cpu), "+", label="cpu")
    ax.plot(line_lengths, np.array(time_gpu) / np.array(nit_gpu), "x", label="gpu")
    ax.plot(line_lengths, np.array(time_numpy) / np.array(nit_numpy), "o", label="numpy")

    ax.legend()
    ax.set_ylabel("time per iteration")
    ax.set_xlabel("line length")

if __name__ == "__main__":
    check_same_result()
    time_vs_line_length_scaling()