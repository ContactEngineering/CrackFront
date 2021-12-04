import time

import numpy as np
# from ContactMechanics.Tools.Logger import Logger
from Adhesion.ReferenceSolutions import JKR
from NuMPI.IO.NetCDF import NCStructuredGrid
import sys
from NuMPI.IO import load_npy, make_mpi_file_view
from NuMPI import MPI

from CrackFront.Circular import RadiusTooLowError
from CrackFront.CircularEnergyReleaseRate import SphereCFPenetrationEnergyConstGcPiecewiseLinearField
import torch

# nondimensional units following Maugis Book:
Es = 3 / 4
w = 1 / np.pi
R = 1.

_jkrkwargs = dict(contact_modulus=Es, radius=R)


class LinearInterpolatedPinningFieldUniformFromFile:
    def __init__(self, filename, min_radius, grid_spacing, accelerator, data_device=torch.device("cpu")):
        """
        Linearly interpolates the pinning field in crack propagation direction

        Parameters:
        -----------
        values: np.ndarray of shape (npx_front, npx_propagation)
            values of the pinning field at the kinks
        kinks: np.ndarray of shape (npx_propagation)
            equidistantly spaced points representing the grid of the piecewise linear interpolation

        """

        self.filename = filename
        self.accelerator = accelerator # TODO: this is actually never used. I think I will leave the transfer to the accelerator to the rosso Krauth script
        self.data_device = data_device
        self.file = make_mpi_file_view(filename, MPI.COMM_SELF, format="npy")

        Lx, L = self.file.nb_grid_pts
        L = L // 2

        self.npx_front = L
        self.npx_propagation = Lx

        self.grid_spacing = grid_spacing
        self.min_radius = min_radius
        self.indexes = np.arange(L, dtype=int)

    def kink_position(self, collocation_point):
        # TODO: should I shift these guys to the accelerator ?
        return self.min_radius + self.grid_spacing * collocation_point

    @property
    def kinks(self):
        return self.kink_position(np.arange(self.npx_propagation))

    def values_and_slopes(self, collocation_point,):
        r"""
        Parameters:
        -----------
        collocation_point: np.array(size=npx_front, dtype=int )
            index of the outmost collocation point within the contact area
        Returns:
        --------
        values_and_slopes: np.array(size=(npx_front, 2), dtype=float)
           array containing for collocation_point the pinning force value and the slope towards the next outward pixel.
        """
        local_indexes = collocation_point - int(self.subdomain[0])
        return self.subdomain_data[local_indexes, self.indexes, :]

    @property
    def nb_subdomain_grid_pts(self):
        return self.subdomain[1] - self.subdomain[0], self.npx_front, 2

    @property
    def subdomain_locations(self):
        return self.subdomain[0], 0, 0

    @property
    def nb_domain_grid_pts(self):
        return self.npx_propagation, self.npx_front, 2

    def load_data(self, colloc_min, colloc_max):
        self.subdomain = torch.tensor([colloc_min, colloc_max], device=self.data_device)
        n_subdomain = int(self.subdomain[1] - self.subdomain[0])
        self.subdomain_data = torch.from_numpy(
            self.file.read([int(self.subdomain[0]), 0],
                           (n_subdomain, self.npx_front * 2),
                           ).reshape(n_subdomain, self.npx_front, 2)).to(device=self.data_device)

    @staticmethod
    def save_values_and_slopes_to_file(values, grid_spacing, filename="values_and_slopes.npy"):
        """
        Parameters:
        -----------
        values: np.array(size=(npx_propagation, npx_front))
        """

        slopes = (np.roll(values, -1, axis=0) - values) / grid_spacing

        # Workaround because NuMPI has no 3D data support
        values_and_slopes = np.zeros((values.shape[0], values.shape[1] * 2))
        values_and_slopes[:, ::2] = values
        values_and_slopes[:, 1::2] = slopes

        # [[values[0,0], slope[0,0], value[0,1], slope[0,1]]]
        # [[values[1,0], slope[1,0], value[1,1], slope[1,1]]]
        # [[values[2,0], slope[2,0], value[2,1], slope[2,1]]]

        np.save(filename, values_and_slopes)

    def __call__(self, a, der="0"):

        a_above = np.searchsorted(self.kinks, a, side="right")
        a_below = a_above - 1
        # TODO:  Wrapping periodic boundary conditions

        # print(a_below)

        values_and_slopes = self.values_and_slopes(a_below).to(device=torch.device("cpu"))

        value_below = values_and_slopes[:, 0]
        slope = values_and_slopes[:, 1]

        if der == "0":
            ret = value_below + slope * (a - self.kink_position(a_below))
        elif der == "1":
            ret = slope

        if isinstance(a, np.ndarray):
            return ret.to(device=torch.device("cpu")).numpy()
        else:
            return ret



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


def propagate_rosso_krauth(line,
                           gtol,
                           penetration_increment,
                           max_penetration,
                           initial_a,  # can also be an initial configuration from a restart
                           dump_fields=True,
                           maxit=10000,
                           logger=None,
                           disable_cuda=False,
                           filename="data.nc",
                           handle_signals=False,
                           restart=False,
                           pinning_field_memory=None,
                           **kwargs):
    """
    Parameters:
    -----------

    """

    # TODO: rethink a good function signature.

    there_is_enough_time_left = True
    if handle_signals:
        import signal
        def recieve_signal(signum, stack):
            nonlocal there_is_enough_time_left
            there_is_enough_time_left = False
            print("Recieved signal, interupting simulation at the next penetration step")

        signal.signal(signal.SIGUSR1, recieve_signal)

    npx_front = line.npx

    grid_spacing = line.piecewise_linear_w_radius.grid_spacing

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

    grid_spacing_cpu = grid_spacing

    nq_front_rfft = torch.fft.rfftfreq(npx_front, 1 / npx_front, **kwargs_array_creation)


    a_test = np.zeros(line.npx)
    a_test[0] = 1
    line_stiffness_individual = 2 * np.pi / line.npx * line.wm * line.elastic_hessp(a_test)[0]
    # TODO: move to GPU

    colloc_point_above = np.zeros_like(initial_a, dtype=int)
    colloc_point_above = np.ceil((initial_a - line.piecewise_linear_w_radius.min_radius) / grid_spacing_cpu, casting="unsafe", out=colloc_point_above)
    colloc_point_above += colloc_point_above * grid_spacing_cpu + line.piecewise_linear_w_radius.min_radius == initial_a
    colloc_point_above = torch.from_numpy(colloc_point_above).to(**kwargs_array_creation)


    a = torch.from_numpy(initial_a).to(**kwargs_array_creation)


    nc = NCStructuredGrid(filename, "a" if restart else "w", (npx_front,))
    i = len(nc)
    integer_penetration = 0
    direction = 1
    if i > 0:  # It is a restart and the direction is not yet clear
        # simply go through all previous iterations to find again what the previous penetration was.
        for j in range(i):
            penetration_cpu = integer_penetration * penetration_increment
            if penetration_cpu >= max_penetration:
                direction = -1
            integer_penetration += direction
    else:
        direction = 1
    max_loaded_colloc_point = 0
    try:
        if pinning_field_memory is None:
            # then we load all data, simply.
            line.piecewise_linear_w_radius.load_data(0, line.piecewise_linear_w_radius.npx_propagation)
        while True:
            penetration_cpu = integer_penetration * penetration_increment
            print("penetation:", penetration_cpu)
            penetration = torch.tensor(penetration_cpu).to(device=accelerator)


            nit = 0
            while nit < maxit:

                # load the needed pinning field values
                if pinning_field_memory:
                    if direction == 1:
                        if max_loaded_colloc_point < line.piecewise_linear_w_radius.npx_propagation - 1:
                            # else we have already loaded all the data we need

                            # New collocation points exceed the range, we need to refresh the memory
                            if torch.any((colloc_point_above - 1) >= max_loaded_colloc_point):

                                min_loaded_colloc_point = torch.min(colloc_point_above - 1)
                                max_loaded_colloc_point = min(min_loaded_colloc_point + pinning_field_memory,
                                                              line.piecewise_linear_w_radius.npx_propagation - 1)
                                line.piecewise_linear_w_radius.load_data(min_loaded_colloc_point, max_loaded_colloc_point)
                    else:
                        if min_loaded_colloc_point > 0:
                            # else we have already loaded all the data we need

                            # New collocation points exceed the range, we need to refresh the memory
                            if torch.any((colloc_point_above - 1) <= min_loaded_colloc_point):
                                max_loaded_colloc_point = torch.max(colloc_point_above)
                                min_loaded_colloc_point = max(0, max_loaded_colloc_point - pinning_field_memory)
                                line.piecewise_linear_w_radius.load_data(min_loaded_colloc_point, max_loaded_colloc_point)

                current_value_and_slope = line.piecewise_linear_w_radius.values_and_slopes(colloc_point_above - 1).to(
                    device=accelerator)

                # Nullify the force on each pixel

                # grad = line.elastic_gradient(a, penetration) \
                eerr_j = JKR.nonequilibrium_elastic_energy_release_rate(
                    contact_radius=a,
                    penetration=penetration,
                    **_jkrkwargs)
                outer_eastic_stifness = 2 * np.pi / npx_front * eerr_j

                grad = a * outer_eastic_stifness
                grad.add_(2 * np.pi / npx_front * line.wm * torch.fft.irfft(nq_front_rfft * torch.fft.rfft(a), n=npx_front))
                # TODO: I have some scalars here that live on CPU, but it doesn't seem to make a big difference.

                # Note: here we have the opposite sign compared to the elastic line code because values is the work of adhesion
                grad.add_(- current_value_and_slope[:, 0])
                grad.add_(- current_value_and_slope[:, 1] * (a - line.piecewise_linear_w_radius.kink_position(colloc_point_above - 1)))

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
                    mask_new_pixel = torch.logical_or(
                        a_new >= line.piecewise_linear_w_radius.kink_position(colloc_point_above),
                        mask_negative_stiffness)
                    a_new = torch.where(mask_new_pixel,
                                        line.piecewise_linear_w_radius.kink_position(colloc_point_above).to(torch.float64), a_new)

                    colloc_point_above.add_(mask_new_pixel)

                    # because of numerical errors it can be that the gradient points in the wrong
                    # direction on some pixels, but is very small.
                    # We just make sure these points do not move backwards
                    a = torch.maximum(a_new, a)
                elif direction == -1:
                    mask_new_pixel = torch.logical_or(
                        a_new <= line.piecewise_linear_w_radius.kink_position(colloc_point_above - 1).to(torch.float64),
                        mask_negative_stiffness)
                    a_new = torch.where(mask_new_pixel,
                                        line.piecewise_linear_w_radius.kink_position(colloc_point_above - 1).to(torch.float64), a_new)

                    colloc_point_above.add_(mask_new_pixel, alpha=-1)
                    # Why not just -= ? -> -= is not allowed for boolean arrays

                    a = torch.minimum(a_new, a)

                if (colloc_point_above < 1).any():
                    raise RadiusTooLowError

                nit += 1
            assert nit < maxit

            nc[i].nit = nit
            print("nit: {}".format(nit))
            # pinning_field_values_cpu = a_cpu = a.to(device=torch.device("cpu")).numpy()
            a_cpu = a.to(device=torch.device("cpu")).numpy()
            # if compute_smallest_eigenvalue:
            #     eigval, eigvec = line.eigenvalues(a)
            #     nc[i].eigenvalue = eigval
            #     if dump_fields:
            #         nc[i].eigenvector = eigvec
            line.dump(nc[i], penetration_cpu, a_cpu, dump_fields=dump_fields)

            if not there_is_enough_time_left:
                # if not dump_fields: # this is actually useless when we dumpfields, but who cares ?
                print("Saving front position for restart")
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
