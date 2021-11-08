
import time

import numpy as np
#from ContactMechanics.Tools.Logger import Logger
from NuMPI.IO.NetCDF import NCStructuredGrid
import sys

from CrackFront.GenericElasticLine import ElasticLine
from CrackFront.Optimization.RossoKrauth import linear_interpolated_pinning_field, brute_rosso_krauth_other_spacing
import torch


def propagate_rosso_krauth(line_length, propagation_length, structural_length,
                                 gtol,
                                 n_steps,
                                 pinning_forces,
                                 initial_a,
                                 dump_fields=True,
                                 simulation_type="pulling parabolic potential",
                                 maxit=10000,
                                 compute_smallest_eigenvalue=False,
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
        print("CUDA detected, using CUDA")
    else:
        if not disable_cuda:
            print("CUDA not available, fall back to torch on CPU")
        else:
            print("CUDA disabled, use CPU")
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
