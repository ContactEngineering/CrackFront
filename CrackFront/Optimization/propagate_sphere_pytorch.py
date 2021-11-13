import time

import numpy as np
# from ContactMechanics.Tools.Logger import Logger
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid
import sys

from CrackFront.Circular import RadiusTooLowError
from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
import torch

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)


def penetrations(dpen, max_pen):
    i = 0  # integer penetration value
    pen = dpen * i
    yield pen
    while pen < max_pen:
        i += 1
        pen = dpen * i
        yield pen
    while True:
        i -= 1
        pen = dpen * i
        yield pen


def propagate_rosso_krauth(piecewise_linear_w,
                           gtol,
                           penetration_increment,
                           max_penetration,
                           initial_a,  # can also be an initial configuration from a restart
                           wm=1/np.pi,
                           dump_fields=True,
                           simulation_type="pulling parabolic potential",
                           maxit=10000,
                           logger=None,
                           disable_cuda=False,
                           filename="data.nc",
                           handle_signals=False,
                           restart=False,
                           **kwargs):
    there_is_enough_time_left = True
    if handle_signals:
        import signal
        def recieve_signal(signum, stack):
            nonlocal there_is_enough_time_left
            there_is_enough_time_left = False

        signal.signal(signal.SIGUSR1, recieve_signal)
    n_pixels = piecewise_linear_w.npx_front

    # axev.axhline(2 * np.pi / structural_length)  # no disorder limit of the smallest eigenvalue

    line = SphereCFPenetrationEnergyConstGcPiecewiseLinearField(piecewise_linear_w, wm=wm)

    npx_front = line.npx

    values = line.piecewise_linear_w.values
    grid_spacing = line.piecewise_linear_w.grid_spacing
    min_radius = line.piecewise_linear_w.kinks[0]

    slopes = (np.roll(values, -1, axis=-1) - values) / grid_spacing

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

    indexes = torch.arange(npx_front, dtype=int)
    grid_spacing = torch.tensor(grid_spacing, dtype=torch.double)
    values_and_slopes = torch.from_numpy(np.stack([values, slopes], axis=2)).to(device=accelerator)

    nq_front_rfft = torch.fft.rfftfreq(npx_front, 1, **kwargs_array_creation)

    #qk = 2 * np.pi / structural_length
    #a_test = torch.zeros(npx_front, **kwargs_array_creation)
    #a_test[0] = 1
    #elastic_stiffness_individual = torch.fft.irfft(q_front_rfft * torch.fft.rfft(a_test), n=npx_front)[0] + qk

    a_test = np.zeros(line.npx)
    a_test[0] = 1
    line_stiffness_individual = 2 * np.pi / line.npx * line.wm * line.elastic_hessp(a_test)[0]
    # TODO: move to GPU

    colloc_point_above = np.zeros_like(initial_a, dtype=int)
    colloc_point_above = np.ceil(initial_a / grid_spacing, casting="unsafe", out=colloc_point_above)
    colloc_point_above += colloc_point_above == initial_a
    colloc_point_above = torch.from_numpy(colloc_point_above).to(**kwargs_array_creation)

    a = torch.from_numpy(initial_a).to(**kwargs_array_creation)

    nc = NCStructuredGrid(filename, "a" if restart else "w", (npx_front,))
    i = len(nc)
    integer_penetration = 0
    direction = 1
    if i > 0: # It is a restart and the direction is not yet clear
        raise NotImplementedError("TODO")
        # I think I can deduce it from the previous penetrations
    else:
        direction = 1


    try:
        while True:
            penetration_cpu = integer_penetration * penetration_increment
            print("penetation:", penetration_cpu)
            penetration = torch.tensor(penetration_cpu).to(device=accelerator)

            nit = 0
            while nit < maxit:
                # Nullify the force on each pixel

                current_value_and_slope = values_and_slopes[indexes, (colloc_point_above - 1), :]

                # grad = line.elastic_gradient(a, penetration) \
                eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
                    contact_radius=a,
                    penetration=penetration,
                    **_jkrkwargs)
                outer_eastic_stifness = 2 * np.pi / npx_front * eerr_j

                grad = a * outer_eastic_stifness
                grad.add_(2 * np.pi / npx_front * wm * torch.fft.irfft(nq_front_rfft * torch.fft.rfft(a), n=npx_front))

                # Note: here we have the opposite sign compared to the elastic line code because values is the work of adhesion
                grad.add_(- current_value_and_slope[:, 0])
                grad.add_(- current_value_and_slope[:, 1] * (a - (min_radius + grid_spacing * (colloc_point_above - 1))))

                max_abs_grad = torch.max(torch.abs(grad))
                # TODO: Optimization: I don't need to evaluate this every iteration
                if max_abs_grad < gtol:
                    break

                if logger:
                    logger.st(["it", "max. residual"], [nit, max_abs_grad])


                # strictly speaking I should take into account that this is nonlinear
                # But in practice with a fine discretisation the stiffness associated
                # with moving one pixel leads to contact area increments small enough so that this nonlinearity
                # doesn't matter
                stiffness = - current_value_and_slope[:, 1] + outer_eastic_stifness + line_stiffness_individual

                increment = - grad / stiffness

                a_new = a + increment
                mask_negative_stiffness = stiffness <= 0

                if direction == 1:
                    # We let the line advance only until the boundary to the next pixel.
                    # This is because the step length was based on the pinning curvature
                    # which is erroneous as soon as we meet the next pixel
                    #
                    # Additionally, when the curvature is negative, the increment is negative
                    # but the front should actually move forward.
                    # In this case as well we advance the front until the edge of the next pixel
                    mask_new_pixel = torch.logical_or(a_new >= min_radius + colloc_point_above * grid_spacing, mask_negative_stiffness)
                    a_new = torch.where(mask_new_pixel, min_radius + grid_spacing * colloc_point_above, a_new)

                    colloc_point_above.add_(mask_new_pixel)

                    # because of numerical errors it can be that the gradient points in the wrong
                    # direction on some pixels, but is very small.
                    # We just make sure these points do not move backwards
                    a = torch.maximum(a_new, a)
                elif direction == -1:
                    mask_new_pixel = torch.logical_or(a_new <= min_radius + grid_spacing * (colloc_point_above - 1),
                                                      mask_negative_stiffness)
                    a_new = torch.where(mask_new_pixel, min_radius + grid_spacing * (colloc_point_above - 1), a_new)

                    colloc_point_above.add_(mask_new_pixel, alpha=-1)
                    # Why not just -= ? -> -= is not allowed for boolean arrays

                    a = torch.minimum(a_new, a)

                if (colloc_point_above < 1).any():
                    raise RadiusTooLowError

                nit += 1
            assert nit < maxit

            nc[i].nit = nit
            print("nit: {}".format(nit))
            a_cpu = a.to(device=torch.device("cpu")).numpy()
            # if compute_smallest_eigenvalue:
            #     eigval, eigvec = line.eigenvalues(a)
            #     nc[i].eigenvalue = eigval
            #     if dump_fields:
            #         nc[i].eigenvector = eigvec
            line.dump(nc[i], penetration_cpu, a_cpu, dump_fields=dump_fields)

            if not there_is_enough_time_left:
                # if not dump_fields: # this is actually useless when we dumpfields, but who cares ?
                np.save("restart_position.npy", a_cpu)
                break

            nc.sync()
            sys.stdout.flush()
            if penetration_cpu >= max_penetration:
                direction = -1
            integer_penetration += direction
            i += 1
    except RadiusTooLowError:
        print("lost contact")

    nc.close()
    return there_is_enough_time_left
