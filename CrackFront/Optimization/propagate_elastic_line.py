
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